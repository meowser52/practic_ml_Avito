"""Этапы пайплайна Olist: PostgreSQL -> CSV-снимок -> проверка -> разбиение -> MLflow.

Адреса и параметры берутся из окружения: DATABASE_URL, MLFLOW_TRACKING_URI,
OLIST_WORKDIR, OLIST_EPOCHS, OLIST_SEED, OLIST_CUTOFF, OLIST_LR, OLIST_EXPERIMENT.
Файлы одного запуска лежат в OLIST_WORKDIR/runs/<run_key>/; между этапами
передаются только пути, идентификаторы и короткие метаданные.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_COLUMNS",
    "Settings",
    "SnapshotValidationError",
    "extract",
    "prepare",
    "train",
    "validate",
]

logger = logging.getLogger(__name__)

PathLike = str | os.PathLike[str]

REQUIRED_COLUMNS = (
    "order_id", "order_purchase_timestamp", "price", "freight", "weight", "n_items",
    "installments", "est_days", "customer_state", "seller_state", "category",
    "payment_type", "late",
)
NUMERIC_COLUMNS = ("price", "freight", "weight", "n_items", "installments", "est_days")
CATEGORICAL_COLUMNS = ("customer_state", "seller_state", "category", "payment_type")

SNAPSHOT_NAME = "snapshot.csv"
_EXPORT_SQL = (
    "COPY (SELECT {columns} FROM public.orders "
    'ORDER BY order_purchase_timestamp, order_id COLLATE "C") '
    "TO STDOUT WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
)


@dataclass(frozen=True)
class Settings:
    """Параметры запуска из окружения контейнера Airflow."""

    workdir: Path
    database_url: str | None
    tracking_uri: str | None
    experiment: str
    epochs: int
    seed: int
    lr: float
    cutoff: str
    db_timeout_s: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            workdir=Path(env.get("OLIST_WORKDIR", "/opt/olist/work")),
            database_url=env.get("DATABASE_URL"),
            tracking_uri=env.get("MLFLOW_TRACKING_URI"),
            experiment=env.get("OLIST_EXPERIMENT", "olist_training"),
            epochs=int(env.get("OLIST_EPOCHS", "5")),
            seed=int(env.get("OLIST_SEED", "42")),
            lr=float(env.get("OLIST_LR", "0.003")),
            cutoff=env.get("OLIST_CUTOFF", "2017-07-31"),
            db_timeout_s=int(env.get("OLIST_DB_TIMEOUT_S", "120")),
        )


class SnapshotValidationError(ValueError):
    """Снимок нарушает контракт данных из data/README.md."""


def _require(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


def _run_dir(workdir: Path, run_key: str) -> Path:
    """Каталог запуска: читаемое имя из run_key и короткий хеш, чтобы разные run_key не совпали."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", run_key).strip("._") or "run"
    suffix = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:8]
    return workdir / "runs" / f"{safe}-{suffix}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- extract

def extract(run_key: str) -> str:
    """Выгрузить public.orders в UTF-8 CSV каталога запуска; вернуть абсолютный путь.

    Повтор того же run_key перезаписывает снимок атомарно, поэтому строки не копятся.
    """
    import psycopg

    if not isinstance(run_key, str) or not run_key:
        raise ValueError("run_key must be a nonempty string")
    settings = Settings.from_env()
    url = _require(settings.database_url, "DATABASE_URL")
    conninfo = re.sub(r"^postgres(?:ql)?\+\w+://", "postgresql://", url)

    run_dir = _run_dir(settings.workdir, run_key)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / SNAPSHOT_NAME
    sql = _EXPORT_SQL.format(columns=", ".join(REQUIRED_COLUMNS))
    fd, tmp_name = tempfile.mkstemp(prefix=".snapshot-", suffix=".part", dir=run_dir)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle, psycopg.connect(
            conninfo,
            connect_timeout=10,
            application_name="olist_extract",
            options=f"-c statement_timeout={settings.db_timeout_s * 1000}",
        ) as conn:
            conn.read_only = True
            with conn.cursor() as cursor:
                with cursor.copy(sql) as copy:
                    for chunk in copy:
                        handle.write(chunk)
                rows = cursor.rowcount
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("Extracted %s rows from public.orders to %s", rows, target)
    return str(target.resolve())


# ---------------------------------------------------------------------------- validate

def _read_raw(path: Path) -> pd.DataFrame:
    """Прочитать CSV как строки без автозамены пропусков; первая строка — заголовок."""
    try:
        return pd.read_csv(path, header=None, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.EmptyDataError as exc:
        raise SnapshotValidationError(f"Snapshot {path} is empty") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise SnapshotValidationError(f"Snapshot {path} is not a readable UTF-8 CSV: {exc}") from exc


def _missing(series: pd.Series) -> np.ndarray:
    return (series.isna() | (series.astype(str).str.strip() == "")).to_numpy()


def _as_float(series: pd.Series) -> np.ndarray:
    numbers = pd.to_numeric(series.astype(object), errors="coerce")
    return np.asarray(numbers, dtype=np.float64)


def _check_frame(frame: pd.DataFrame) -> list[str]:
    problems = []
    ids = frame["order_id"]
    missing_ids = int(_missing(ids).sum())
    if missing_ids:
        problems.append(f"order_id has {missing_ids} empty values")
    duplicated = ids[ids.duplicated(keep=False) & ~_missing(ids)]
    if len(duplicated):
        examples = ", ".join(sorted(duplicated.unique())[:3])
        problems.append(f"order_id is not unique: {len(duplicated)} rows share ids ({examples})")

    stamps = pd.to_datetime(frame["order_purchase_timestamp"].astype(object), format="ISO8601", errors="coerce", utc=True)
    bad_stamps = int(stamps.isna().sum())
    if bad_stamps:
        problems.append(f"order_purchase_timestamp has {bad_stamps} missing or non-ISO values")

    for column in NUMERIC_COLUMNS:
        bad = int((~np.isfinite(_as_float(frame[column]))).sum())
        if bad:
            problems.append(f"{column} has {bad} missing, non-numeric or infinite values")

    labels = _as_float(frame["late"])
    bad_labels = int((~np.isin(labels, (0.0, 1.0))).sum())
    if bad_labels:
        problems.append(f"late has {bad_labels} values outside 0/1")

    for column in CATEGORICAL_COLUMNS:
        empty = int(_missing(frame[column]).sum())
        if empty:
            problems.append(f"{column} has {empty} empty values")
    return problems


def validate(snapshot: PathLike) -> str:
    """Проверить CSV-снимок только чтением; вернуть путь или вызвать SnapshotValidationError.

    Проверки: файл не пустой, есть все 13 столбцов без дублей, order_id уникален,
    даты в ISO 8601, числовые признаки конечны, late равен 0 или 1, категории заполнены.
    """
    path = Path(snapshot)
    if not path.is_file():
        raise SnapshotValidationError(f"Snapshot {path} is not an existing file")
    raw = _read_raw(path)
    header = [str(name) for name in raw.iloc[0].tolist()]
    problems = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing_columns:
        problems.append(f"missing columns: {', '.join(missing_columns)}")
    repeated = sorted({name for name in header if header.count(name) > 1})
    if repeated:
        problems.append(f"repeated column names: {', '.join(repeated)}")
    if len(raw) < 2:
        problems.append("no data rows")
    if not problems:
        frame = raw.iloc[1:].reset_index(drop=True)
        frame.columns = header
        problems = _check_frame(frame)
    if problems:
        raise SnapshotValidationError(f"Snapshot {path} failed validation: " + "; ".join(problems))
    logger.info("Snapshot %s passed validation: %d rows", path, len(raw) - 1)
    return os.fspath(snapshot)


# ----------------------------------------------------------------------------- prepare

def _write_csv_atomic(frame: pd.DataFrame, target: Path) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".part", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _output_dir(source: Path, digest: str, workdir: Path) -> Path:
    """Каталог запуска снимка; для файла вне OLIST_WORKDIR/runs — отдельный каталог по хешу."""
    runs_root = (workdir / "runs").resolve()
    if source.parent.parent == runs_root:
        return source.parent
    return runs_root / f"adhoc-{digest[:16]}"


def prepare(snapshot: PathLike) -> dict[str, Any]:
    """Разделить снимок по времени (как в ноутбуке) и вернуть пути и метаданные для train.

    SHA256 считается по байтам исходного CSV extract, разбиение — по тем же байтам.
    """
    from olist_model import split_by_time

    settings = Settings.from_env()
    source = Path(snapshot).resolve(strict=True)
    data = source.read_bytes()
    digest = _sha256_bytes(data)
    frame = pd.read_csv(io.BytesIO(data), encoding="utf-8")
    train_df, valid_df = split_by_time(frame, cutoff=settings.cutoff)

    out_dir = _output_dir(source, digest, settings.workdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = _write_csv_atomic(train_df, out_dir / "train.csv")
    valid_path = _write_csv_atomic(valid_df, out_dir / "validation.csv")
    prepared = {
        "snapshot_path": str(source),
        "snapshot_sha256": digest,
        "snapshot_rows": len(frame),
        "cutoff": settings.cutoff,
        "run_dir": str(out_dir),
        "train_path": str(train_path),
        "validation_path": str(valid_path),
        "train_rows": len(train_df),
        "validation_rows": len(valid_df),
    }
    logger.info("Prepared split: %s", prepared)
    return prepared


# ------------------------------------------------------------------------------- train

def _check_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return value


def _read_split(path: str, expected_rows: int, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(f"{name} has {len(frame)} rows, prepare reported {expected_rows}")
    return frame


def _save_model(predictor: Any, valid_df: pd.DataFrame, model_dir: Path) -> None:
    """Сохранить модель с преобразованиями и убедиться, что она загружается обратно."""
    from olist_model import load_predictor, save_predictor

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-", dir=model_dir.parent))
    try:
        save_predictor(predictor, staging)
        sample = valid_df.drop(columns="late").head(256)
        if not np.array_equal(predictor.predict(sample), load_predictor(staging).predict(sample)):
            raise RuntimeError("Reloaded model gives different predictions")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        os.replace(staging, model_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def train(
    prepared: Mapping[str, Any],
    *,
    dag_run_id: str,
    epochs: int | None = None,
    seed: int | None = None,
) -> str:
    """Обучить выданную модель, записать run в MLflow и вернуть его run_id.

    Тег dag_run_id — run_id запуска Airflow без изменений; snapshot_sha256 — хеш
    исходного CSV из prepared. Модель с преобразованиями лежит в артефакте model/.
    """
    import mlflow

    from olist_model import fit_model

    if not isinstance(dag_run_id, str) or not dag_run_id:
        raise ValueError("dag_run_id must be a nonempty string")
    settings = Settings.from_env()
    tracking_uri = _require(settings.tracking_uri, "MLFLOW_TRACKING_URI")
    epochs = _check_int("epochs", settings.epochs if epochs is None else epochs)
    seed = _check_int("seed", settings.seed if seed is None else seed)

    train_df = _read_split(prepared["train_path"], prepared["train_rows"], "train")
    valid_df = _read_split(prepared["validation_path"], prepared["validation_rows"], "validation")
    model_dir = Path(prepared.get("run_dir") or Path(prepared["train_path"]).parent) / "model"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings.experiment)
    tags = {"dag_run_id": dag_run_id, "snapshot_path": prepared["snapshot_path"]}
    with mlflow.start_run(run_name=dag_run_id, tags=tags) as run:
        mlflow.log_params({
            "epochs": epochs,
            "seed": seed,
            "lr": settings.lr,
            "cutoff": prepared["cutoff"],
            "train_rows": len(train_df),
            "validation_rows": len(valid_df),
            "snapshot_sha256": prepared["snapshot_sha256"],
        })
        predictor, history = fit_model(train_df, valid_df, epochs=epochs, seed=seed, lr=settings.lr)
        for row in history:
            mlflow.log_metrics(
                {"train_loss": row["train_loss"], "validation_pr_auc": row["validation_pr_auc"]},
                step=row["epoch"],
            )
        _save_model(predictor, valid_df, model_dir)
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        run_id = run.info.run_id
    logger.info("MLflow run %s finished, last epoch: %s", run_id, history[-1])
    return run_id
