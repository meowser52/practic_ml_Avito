"""Supplied offline CPU baseline; students orchestrate it, not reimplement it."""
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn


@contextmanager
def _cpu_threads():
    """Avoid oversubscribing tiny CPU batches; restore the caller's setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)

NUMERIC_COLUMNS = ["price", "freight", "weight", "n_items", "installments", "est_days"]
CATEGORICAL_COLUMNS = ["customer_state", "seller_state", "category", "payment_type"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def _validate_frame(df, *, target=False, name="data"):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{name} must be a nonempty pandas DataFrame")
    required = FEATURE_COLUMNS + (["late"] if target else [])
    if not df.columns.is_unique or not set(required).issubset(df.columns):
        raise ValueError(f"{name} requires unique columns including {required}")
    try:
        numbers = df[NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} numeric features must be numbers") from exc
    if not np.isfinite(numbers).all():
        raise ValueError(f"{name} numeric features must be finite (no missing/NaN/Inf)")
    for column in CATEGORICAL_COLUMNS:
        if not df[column].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{name}.{column} must contain strings or missing values")
    if target:
        try:
            labels = df["late"].to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}.late must contain binary numeric labels") from exc
        if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{name}.late must contain finite binary 0/1 labels")
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{name}.late must contain both classes")


def split_by_time(df, cutoff="2017-07-31"):
    """Purchase time < cutoff is train; cutoff and later are validation."""
    _validate_frame(df, target=True)
    if "order_purchase_timestamp" not in df.columns:
        raise ValueError("order_purchase_timestamp is required for the time split")
    dates = pd.to_datetime(df["order_purchase_timestamp"], format="ISO8601", errors="raise", utc=True)
    boundary = pd.to_datetime(cutoff, errors="raise", utc=True)
    if not isinstance(boundary, pd.Timestamp) or pd.isna(boundary) or dates.isna().any():
        raise ValueError("Purchase dates and cutoff must not be missing/NaT")
    mask = dates < boundary
    train, valid = df.loc[mask].copy(), df.loc[~mask].copy()
    _validate_frame(train, target=True, name="train")
    _validate_frame(valid, target=True, name="validation")
    return train, valid


class Predictor:
    """Complete CPU model plus train-fitted numerical/categorical transforms.

    Unknown categories are represented by all-zero one-hot blocks. Missing
    strings use an explicit sentinel. Numeric missing values are not imputed.
    """

    def __init__(self, model, preprocessing):
        self.model = model.cpu().eval()
        self.preprocessing = preprocessing

    def _transform(self, df):
        numbers = df[NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
        mean = np.asarray(self.preprocessing["mean"], dtype=np.float64)
        scale = np.asarray(self.preprocessing["scale"], dtype=np.float64)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                blocks = [((numbers - mean) / scale).astype(np.float32)]
            except FloatingPointError as exc:
                raise ValueError("Numeric features exceed finite float32 preprocessing range") from exc
        for column in CATEGORICAL_COLUMNS:
            values = df[column].fillna("<missing>").astype(str).to_numpy()
            categories = self.preprocessing["categories"][column]
            block = np.zeros((len(df), len(categories)), dtype=np.float32)
            for index, value in enumerate(categories):
                block[:, index] = values == value
            blocks.append(block)
        return torch.from_numpy(np.concatenate(blocks, axis=1))

    def predict(self, df):
        _validate_frame(df)
        with _cpu_threads(), torch.inference_mode():
            result = torch.sigmoid(self.model(self._transform(df)).squeeze(1)).numpy().copy()
        if not np.isfinite(result).all():
            raise ValueError("Model produced nonfinite probabilities")
        return result


def _network(input_size, hidden):
    return nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def fit_model(train_df, valid_df, epochs=5, seed=42, lr=0.003, hidden=32):
    """Train a small MLP on CPU. Validation is used for metrics only.

    Report average precision as validation_pr_auc (stepwise PR integral).
    Loss is class-weighted BCE; outputs are positive scores, not calibrated risks.
    """
    _validate_frame(train_df, target=True, name="train")
    _validate_frame(valid_df, target=True, name="validation")
    for name, value in (("epochs", epochs), ("hidden", hidden)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer < 2**63")
    if not isinstance(lr, (int, float)) or not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    numbers = train_df[NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            mean, scale = numbers.mean(axis=0), numbers.std(axis=0)
        except FloatingPointError as exc:
            raise ValueError("Numeric training statistics overflow the finite preprocessing range") from exc
    scale[scale == 0] = 1.0
    preprocessing = {
        "format_version": 1, "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS, "mean": mean.tolist(),
        "scale": scale.tolist(), "hidden": hidden,
        "categories": {column: sorted(train_df[column].fillna("<missing>").astype(str).unique().tolist()) for column in CATEGORICAL_COLUMNS},
    }
    _validate_preprocessing(preprocessing)
    input_size = len(NUMERIC_COLUMNS) + sum(len(values) for values in preprocessing["categories"].values())
    with _cpu_threads(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        predictor = Predictor(_network(input_size, hidden), preprocessing)
        x = predictor._transform(train_df)
        y = torch.tensor(train_df.late.to_numpy(dtype=np.float32))
        optimizer = torch.optim.Adam(predictor.model.parameters(), lr=lr)
        positive_weight = (len(y) - y.sum()) / y.sum()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        history = []
        for epoch in range(1, epochs + 1):
            predictor.model.train()
            total_loss = 0.0
            for batch in torch.randperm(len(y), generator=generator).split(256):
                optimizer.zero_grad()
                loss = loss_fn(predictor.model(x[batch]).squeeze(1), y[batch])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            predictor.model.eval()
            score = float(average_precision_score(valid_df.late, predictor.predict(valid_df)))
            history.append({"epoch": epoch, "train_loss": total_loss / len(y), "validation_pr_auc": score})
    return predictor, history


def _validate_preprocessing(metadata):
    """Validate the small, explicit JSON schema before allocating a network."""
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        raise ValueError("Unsupported preprocessing format_version")
    if metadata.get("numeric_columns") != NUMERIC_COLUMNS or metadata.get("categorical_columns") != CATEGORICAL_COLUMNS:
        raise ValueError("Preprocessing feature columns do not match the supplied model")
    hidden = metadata.get("hidden")
    if type(hidden) is not int or not 1 <= hidden <= 4096:
        raise ValueError("hidden must be an integer in [1, 4096]")
    for field in ("mean", "scale"):
        try:
            value = np.asarray(metadata[field], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid preprocessing {field}") from exc
        if value.shape != (len(NUMERIC_COLUMNS),) or not np.isfinite(value).all():
            raise ValueError(f"Preprocessing {field} must have six finite values")
        if field == "scale" and (value <= 0).any():
            raise ValueError("Preprocessing scales must be positive")
    categories = metadata.get("categories")
    if not isinstance(categories, dict) or set(categories) != set(CATEGORICAL_COLUMNS):
        raise ValueError("Preprocessing requires all categorical vocabularies")
    for column, values in categories.items():
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values) or len(set(values)) != len(values):
            raise ValueError(f"Invalid category vocabulary for {column}")
    if sum(map(len, categories.values())) > 20000:
        raise ValueError("Categorical vocabulary exceeds the small-model limit")


def save_predictor(predictor, directory):
    """Save tensors only, plus JSON describing training-fitted preprocessing.

    This is an artifact directory, not a whole-object Python pickle. Load only
    expected local/downloaded course artifacts: weights_only is not a sandbox
    against malicious resource exhaustion, and hashes are not authentication.
    """
    import json
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(predictor.model.cpu().state_dict(), directory / "weights.pt")
    (directory / "preprocessing.json").write_text(json.dumps(predictor.preprocessing, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_predictor(directory):
    """Rebuild the supplied architecture, read JSON, load only tensor weights."""
    import json
    from pathlib import Path

    directory = Path(directory)
    preprocessing = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
    _validate_preprocessing(preprocessing)
    input_size = len(NUMERIC_COLUMNS) + sum(len(values) for values in preprocessing["categories"].values())
    with torch.random.fork_rng(devices=[]):
        model = _network(input_size, preprocessing["hidden"])
    state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Expected a finite tensor-only model state_dict")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Weights do not match the declared model architecture") from exc
    return Predictor(model, preprocessing)
