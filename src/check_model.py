"""Supplied check: download the model artifact and predict, without unsafe pickle.

Usage: python -m check_model --run-id RUN_ID
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


def validate_probabilities(values):
    probabilities = [float(value) for value in values]
    if not probabilities or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Expected nonempty finite probabilities in [0, 1]')
    return probabilities


def predict(run_id: str, dataset: Path, rows: int = 5) -> dict:
    import mlflow
    import pandas as pd
    from olist_model import load_predictor

    if not 1 <= rows <= 100:
        raise ValueError('rows must be between 1 and 100')
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise ValueError('Expected a 32-character lowercase hexadecimal MLflow run_id')
    frame = pd.read_csv(dataset).head(rows)
    with tempfile.TemporaryDirectory(prefix='olist-model-') as directory:
        model_path = mlflow.artifacts.download_artifacts(artifact_uri=f'runs:/{run_id}/model', dst_path=directory)
        predictor = load_predictor(Path(model_path))
        probabilities = validate_probabilities(predictor.predict(frame))
    if len(probabilities) != len(frame):
        raise ValueError('Prediction count does not match input rows')
    return {'run_id': run_id, 'rows': len(frame), 'probabilities': probabilities}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--data', type=Path, default=Path(os.environ.get('OLIST_DATASET', '/opt/olist/data/olist_orders.csv.gz')))
    parser.add_argument('--rows', type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(predict(args.run_id, args.data, args.rows), allow_nan=False))


if __name__ == '__main__':
    main()
