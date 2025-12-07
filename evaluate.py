import os
import argparse
import mlflow
from pathlib import Path
from dotenv import load_dotenv

import src.metrics
from src.base.utils import (
    load_configuration,
    load_dataframe,
    get_X_y,
    get_schema,
)
from src.model import ModelRegistry, get_model_class
from src.base.evaluator import Evaluator


def get_argv():
    parser = argparse.ArgumentParser(prog="Model Training")
    parser.add_argument(
        "-e", "--experiment", type=str, help="MLflow experiment name"
    )
    parser.add_argument(
        "--train-config", type=str, help="Traininig configuration"
    )
    parser.add_argument(
        "-est",
        "--estimator",
        type=str,
        help="Estimator id",
    )
    parser.add_argument(
        "-tr",
        "--transformer",
        type=str,
        default=None,
        help="Transformer id",
    )
    return vars(parser.parse_args())


def evaluate():
    argv = get_argv()
    config = load_configuration("configs/config.yaml")
    train_config = load_configuration(argv.get("train_config"))

    train_data_path = Path(
        config["preprocessed_data_dir"], config["test_data"]
    )
    df = load_dataframe(train_data_path, config["cols"])

    X_test, y_test = get_X_y(
        df,
        get_schema(config["cols"]),
        list(config["cols"]["target"].keys())[0],
    )

    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_tracking_uri("http://127.0.0.1:8000")

    evaluator = Evaluator(X_test, y_test, train_config)
    evaluator.evaluate(
        argv["experiment"], argv.get("estimator"), argv.get("transformer")
    )


if __name__ == "__main__":
    load_dotenv()
    evaluate()
