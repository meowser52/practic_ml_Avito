FROM apache/airflow:3.3.2-python3.12
USER root
RUN mkdir -p /opt/olist/work /opt/olist/src /opt/olist/data && chown -R airflow:root /opt/olist
USER airflow
COPY requirements.txt /opt/olist/requirements.txt
RUN pip install --no-cache-dir -r /opt/olist/requirements.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
COPY --chown=airflow:root src/ /opt/olist/src/
COPY --chown=airflow:root dags/ /opt/airflow/dags/
ENV PYTHONPATH=/opt/olist/src
