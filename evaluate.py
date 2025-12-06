import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.base.trainer import MlflowManager
from src.base.utils import load_configuration, load_dataframe, get_X_y
from src.model import ModelRegistry, get_model_class
from src.evaluator import Evaluator


def get_argv():
    parser = argparse.ArgumentParser(prog="Model Training")
    parser.add_argument(
        "-e", "--experiment", type=str, help="MLflow experiment name"
    )
    parser.add_argument(
        "--train-config", type=str, help="Traininig configuration"
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=list(ModelRegistry.registry.keys()),
        help="Model name",
    )
    parser.add_argument(
        "--logtr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether apply log transformation to target or not",
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
    parser.add_argument(
        "-s",
        "--save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether save model or not",
    )
    return vars(parser.parse_args())


def evaluate():
    argv = get_argv()
    config = load_configuration("configs/config.yaml")
    train_config = load_configuration(argv.get("train_config"))

    train_data_path = (
        Path(config["preprocessed_data_dir"]) / config["test_data"]
    )
    df = load_dataframe(train_data_path, config["cols"])

    X_test, y_test = get_X_y(df, list(config["cols"]["target"].keys())[0])

    tracking_manager = MlflowManager(os.getenv("MLFLOW_TRACKING_URI"))
    evaluator = Evaluator(X_test, y_test, tracking_manager)

    model = get_model_class(train_config["model_name"])(
        train_config["model_name"],
        train_config.get("transformer_name"),
        train_config["log_transform"],
        schema=train_config.get("schema"),
        cat_features=train_config.get("cat_features"),
    )
    model.load(argv["estimator"], argv["transformer"])

    evaluator.evaluate(
        argv["experiment"], model, config["metrics"], argv["save"]
    )


if __name__ == "__main__":
    load_dotenv()
    evaluate()
