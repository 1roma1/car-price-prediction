import numpy as np
from optuna.trial import Trial

from src.base.registries import HyperparameterRegistry


@HyperparameterRegistry.register("ridge")
def ridge_hyperparameters(trial: Trial) -> dict:
    return {
        "alpha": trial.suggest_categorical("alpha", np.logspace(-4, 2, 50)),
    }


@HyperparameterRegistry.register("xgb")
def get_hyperparameters(trial: Trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [50, 100, 150, 200, 250, 300, 400, 500]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [3, 6, 9, 12]),
        "max_leaves": trial.suggest_categorical(
            "max_leaves", [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.01, 0.02, 0.03, 0.5, 0.8, 0.1, 0.12, 0.15, 0.2]
        ),
        "gamma": trial.suggest_categorical(
            "gamma", [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        ),
        "min_child_weight": trial.suggest_categorical(
            "min_child_weight", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ),
    }
