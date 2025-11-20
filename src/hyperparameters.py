from typing import Any, Callable

import numpy as np
from optuna.trial import Trial


class Hyperparameters:
    registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_func: Any) -> Any:
            cls.registry[name] = wrapped_func
            return wrapped_func

        return inner_wrapper

    @classmethod
    def get_hyperparameters(cls, name: str, trial: Trial) -> Any:
        if name in cls.registry:
            return cls.registry[name](trial)
        else:
            raise ValueError(f"There is no model: {name}, available: {cls.registry.keys()}")


@Hyperparameters.register("ridge")
def ridge_hyperparameters(trial: Trial) -> dict:
    return {
        "alpha": trial.suggest_categorical("alpha", np.logspace(-4, 2, 50)),
    }


@Hyperparameters.register("xgb")
def get_hyperparameters(trial: Trial) -> dict:
    return {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [50, 100, 150, 200, 250, 300, 400, 500]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [3, 6, 9, 12]),
        "max_leaves": trial.suggest_categorical("max_leaves", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.01, 0.05, 0.1, 0.2, 0.25, 0.3]
        ),
        "enable_categorical": trial.suggest_categorical("enable_categorical", [True]),
    }
