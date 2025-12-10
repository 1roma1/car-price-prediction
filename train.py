import os
import argparse
import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from src.base.trainer import Trainer, CrossValidation
from src.base.utils import load_yaml, get_X_y


def get_argv():
    parser = argparse.ArgumentParser(prog="Model Training")
    parser.add_argument(
        "-e", "--experiment", type=str, help="MLflow experiment name"
    )
    parser.add_argument(
        "--train-config", type=str, help="Traininig configuration"
    )
    return vars(parser.parse_args())


def train():
    argv = get_argv()
    config = load_yaml("configs/config.yaml")
    train_config = load_yaml(argv["train_config"])

    df = pd.read_csv(
        Path(config["preprocessed_data_dir"], config["train_data"])
    )

    X_train, y_train = get_X_y(df, config["target"])
    validator = CrossValidation(
        X_train,
        y_train,
        n_folds=config["n_folds"],
        random_state=config["random_state"],
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    trainer = Trainer(
        X_train=X_train,
        y_train=y_train,
        config=train_config,
        validator=validator,
    )
    trainer.train(
        argv["experiment"],
    )


if __name__ == "__main__":
    load_dotenv()
    train()
