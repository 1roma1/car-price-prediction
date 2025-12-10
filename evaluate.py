import os
import argparse
import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import src.metrics
from src.base.utils import load_yaml, get_X_y
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
    config = load_yaml("configs/config.yaml")
    train_config = load_yaml(argv.get("train_config"))

    df = pd.read_csv(
        Path(config["preprocessed_data_dir"], config["test_data"])
    )
    X_test, y_test = get_X_y(df, config["target"])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    evaluator = Evaluator(X_test, y_test, train_config)
    evaluator.evaluate(
        argv["experiment"], argv.get("estimator"), argv.get("transformer")
    )


if __name__ == "__main__":
    load_dotenv()
    evaluate()
