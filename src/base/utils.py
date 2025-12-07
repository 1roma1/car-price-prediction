import json
import yaml
import mlflow
import pandas as pd


def load_json(filename: str) -> dict:
    """Load json data from file"""

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_yaml(config_file: str) -> dict:
    """Load configuration from yaml file"""

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_X_y(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Divide dataframe on features and target"""

    return df.drop(labels=target, axis=1), df[target]


def get_or_create_experiment(experiment_name: str) -> str:
    """Create a mlflow experiment and return its id"""

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
