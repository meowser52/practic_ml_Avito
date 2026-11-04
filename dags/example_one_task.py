"""Supplied example, not the solution. Implement olist_training separately."""
from datetime import datetime, timedelta, timezone
from airflow.sdk import dag, task

@dag(schedule=None, start_date=datetime(2026, 1, 1, tzinfo=timezone.utc), catchup=False, tags=['example'])
def example_one_task():
    @task(execution_timeout=timedelta(minutes=1))
    def hello():
        print('Airflow works. Replace this example with your four-task DAG.')
    hello()

example_one_task()
