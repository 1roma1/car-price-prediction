import json
import yaml
import mlflow
import pandas as pd

from pathlib import Path
from typing import Dict


def load_json(filename: str):
    """Load json data from file"""

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_configuration(config_file: str) -> Dict:
    """Load configuration from yaml file"""

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_schema(dictionary):
    result = {}
    for value in dictionary.values():
        result.update(value)
    return result


def load_dataframe(path: Path, cols: dict):
    df = pd.read_csv(path)
    return df.astype(get_schema(cols))


def get_X_y(df: pd.DataFrame, schema: dict, target: str):
    df = df.astype(schema)
    return df.drop(labels=target, axis=1), df[target]


def get_or_create_experiment(experiment_name) -> str:
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
