"""Еженедельное обучение модели опоздания доставки Olist: extract -> validate -> prepare -> train.

Через XCom идут только путь к снимку, словарь с путями и SHA256 и MLflow run_id;
таблицы и модель лежат в общем каталоге /opt/olist/work/runs/<run_id>/.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from airflow.sdk import Param, dag, get_current_context, task

DEFAULT_ARGS = {
    "owner": "olist",
    "retries": 0,
    "execution_timeout": timedelta(minutes=10),
}


@dag(
    dag_id="olist_training",
    description="PostgreSQL orders -> validated CSV snapshot -> time split -> MLflow run with model",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=1),
    default_args=DEFAULT_ARGS,
    params={
        "epochs": Param(None, type=["null", "integer"], minimum=1, description="Empty: OLIST_EPOCHS"),
        "seed": Param(None, type=["null", "integer"], minimum=0, description="Empty: OLIST_SEED"),
    },
    tags=["olist", "training"],
)
def olist_training() -> None:
    @task(retries=3, retry_delay=timedelta(seconds=30), execution_timeout=timedelta(minutes=5))
    def extract() -> str:
        import olist_pipeline

        return olist_pipeline.extract(get_current_context()["run_id"])

    @task(retries=0, trigger_rule="all_success", execution_timeout=timedelta(minutes=5))
    def validate(snapshot: str) -> str:
        import olist_pipeline

        return olist_pipeline.validate(snapshot)

    @task(trigger_rule="all_success", multiple_outputs=False, execution_timeout=timedelta(minutes=10))
    def prepare(snapshot: str) -> dict[str, Any]:
        import olist_pipeline

        return olist_pipeline.prepare(snapshot)

    @task(trigger_rule="all_success", execution_timeout=timedelta(minutes=30))
    def train(prepared: dict[str, Any]) -> str:
        import olist_pipeline

        context = get_current_context()
        params = context["params"]
        return olist_pipeline.train(
            prepared,
            dag_run_id=context["run_id"],
            epochs=params.get("epochs"),
            seed=params.get("seed"),
        )

    train(prepare(validate(extract())))


olist_training()
