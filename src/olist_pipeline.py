"""Заготовки решения: смотрите prototype.ipynb и ASSIGNMENT.md.

Адреса берите из DATABASE_URL, MLFLOW_TRACKING_URI, OLIST_WORKDIR.
Через XCom передавайте только пути, идентификаторы и короткие метаданные,
включая SHA256 снимка, но не таблицы, модели или содержимое файлов.
"""

def extract(run_key):
    """Прочитать public.orders, сохранить обычный UTF-8 CSV; вернуть абсолютный путь строкой."""
    raise NotImplementedError('Implement PostgreSQL extraction')


def validate(snapshot):
    """Проверить файл snapshot до обучения; вернуть путь или вызвать исключение.

    Только чтение CSV: без записей, отчётов, mkdir/chdir, процессов, сети и БД.
    Обязательны все 13 столбцов
    из data/README.md, включая order_purchase_timestamp. Оставьте функцию
    доступной для прямой проверки на испорченных копиях файла.
    """
    raise NotImplementedError('Implement input validation')


def prepare(snapshot):
    """Разделить данные по времени, вернуть пути и метаданные для train.

    Сохраните путь и SHA256 байтов исходного snapshot, не подготовленных таблиц.
    Внутренний формат prepared выбираете сами: обязательного набора ключей нет.
    """
    raise NotImplementedError('Implement preparation')


def train(prepared, *, dag_run_id, epochs=None, seed=None):
    """Обучить модель и сохранить параметры, метрики и полную модель в MLflow.

    DAG передаёт dag_run_id из фактического run_id контекста задачи Airflow.
    Запишите его без изменений в тег dag_run_id, а SHA256 исходного файла extract
    из prepared — в параметр snapshot_sha256. Верните идентификатор MLflow run.
    Остальные имена параметров: epochs, seed, lr, cutoff, train_rows, validation_rows;
    метрики по эпохам: train_loss и validation_pr_auc. Полная модель: model/.
    epochs/seed — необязательный бонус для Params или dag_run.conf; при None
    используйте рабочие значения по умолчанию. Эту сигнатуру можно адаптировать
    вместе с DAG, сохраняя обязательные метаданные и публичную validate(snapshot).
    """
    raise NotImplementedError('Implement training and MLflow tracking')
