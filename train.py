import os
import argparse
import mlflow
from pathlib import Path
from dotenv import load_dotenv

import src.metrics
import src.hyperparameters
from src.base.trainer import Trainer, CrossValidation
from src.models import linear
from src.base.utils import (
    load_configuration,
    load_dataframe,
    get_X_y,
    get_schema,
)


def get_argv():
    parser = argparse.ArgumentParser(prog="Model Training")
    parser.add_argument(
        "-e", "--experiment", type=str, help="MLflow experiment name"
    )
    parser.add_argument(
        "--train-config", type=str, help="Traininig configuration"
    )
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     choices=list(ModelRegistry.registry.keys()),
    #     help="Model name",
    # )
    # parser.add_argument(
    #     "-tr",
    #     "--transformer",
    #     choices=list(Transformers.registry.keys()),
    #     default=None,
    #     help="Transformer name",
    # )
    # parser.add_argument(
    #     "--logtr",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Whether apply log transformation to target or not",
    # )
    # parser.add_argument(
    #     "--optimize",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Train model with hyperparameter tuning",
    # )
    # parser.add_argument(
    #     "-d",
    #     "--direction",
    #     choices=["minimize", "maximize"],
    #     help="Direction for optuna optimizator",
    # )
    # parser.add_argument(
    #     "-t",
    #     "--trials",
    #     type=int,
    #     help="Number of trials",
    # )
    # parser.add_argument(
    #     "-om",
    #     "--optimization-metric",
    #     default="rmse",
    #     help="Metric that will be optimized by optuna",
    # )
    # parser.add_argument(
    #     "-s",
    #     "--save",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Whether save model or not",
    # )
    return vars(parser.parse_args())


def train():
    argv = get_argv()
    config = load_configuration("configs/config.yaml")
    train_config = load_configuration(argv["train_config"])

    train_data_path = (
        Path(config["preprocessed_data_dir"]) / config["train_data"]
    )
    df = load_dataframe(train_data_path, config["cols"])

    X_train, y_train = get_X_y(
        df,
        get_schema(config["cols"]),
        list(config["cols"]["target"].keys())[0],
    )
    validator = CrossValidation(
        X_train,
        y_train,
        n_folds=config["n_folds"],
        random_state=config["random_state"],
    )

    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_tracking_uri("http://127.0.0.1:8000")

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
