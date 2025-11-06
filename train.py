import argparse
from pathlib import Path

from src.model import TrainRegistry
from src.model import Model
from src.features import transfomrers_map
from src.trainer import Trainer, MlflowManager, Optimizer, CrossValidation
from src.utils import load_configuration, load_dataframe, get_X_y


def get_argv():
    parser = argparse.ArgumentParser(prog="Model Training")
    parser.add_argument("-e", "--experiment", type=str, help="MLflow experiment name")
    parser.add_argument(
        "-m",
        "--model",
        choices=list(TrainRegistry.registry.keys()),
        help="Model name",
    )
    parser.add_argument(
        "-tr",
        "--transformer",
        type=str,
        default=None,
        help="Transformer name",
    )
    parser.add_argument(
        "--logtr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether apply log transformation to target or not",
    )
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train model with hyperparameter tuning",
    )
    parser.add_argument(
        "-d",
        "--direction",
        choices=["minimize", "maximize"],
        help="Direction for optuna optimizator",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        help="Number of trials",
    )
    parser.add_argument(
        "-om",
        "--optimization-metric",
        default="rmse",
        help="Metric that will be optimized by optuna",
    )
    parser.add_argument(
        "-s",
        "--save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether save model or not",
    )
    return vars(parser.parse_args())


def train():
    argv = get_argv()
    config = load_configuration("config.yaml")

    train_data_path = Path(config["preprocessed_data_dir"]) / config["train_data"]
    df = load_dataframe(train_data_path, config["cols"])

    X_train, y_train = get_X_y(df, list(config["cols"]["target"].keys())[0])

    transformer = transfomrers_map[argv["transformer"]]() if argv["transformer"] else None

    validator = CrossValidation(
        X_train,
        y_train,
        n_folds=config["n_folds"],
        stratify=True,
        stratify_col=X_train["price_usd_bin"],
        random_state=config["random_state"],
    )

    tracking_manager = MlflowManager(config["mlflow_uri"])
    optimizer = Optimizer(
        direction=argv["direction"], n_trials=argv["trials"] if argv["optimize"] else None
    )
    model = Model(
        argv["model"],
        transformer,
        argv["logtr"],
        cat_features=list(config["cols"]["mul_cols"].keys()),
    )

    trainer = Trainer(
        X_train=X_train,
        y_train=y_train,
        validator=validator,
        tracking_manager=tracking_manager,
        optimizer=optimizer,
    )
    trainer.train(
        argv["experiment"],
        model,
        config["metrics"],
        optimize=argv["optimize"],
        optimization_metric=argv["optimization_metric"],
        save=argv["save"],
    )


if __name__ == "__main__":
    train()
