from abc import ABC, abstractmethod

import numpy as np
from optuna.trial import Trial


class BaseHyperparameters(ABC):
    @classmethod
    @abstractmethod
    def get_hyperparameters(cls, trial: Trial) -> dict:
        pass


class RidgeHyperparameters(BaseHyperparameters):
    @classmethod
    def get_hyperparameters(cls, trial: Trial) -> dict:
        return {
            "alpha": trial.suggest_categorical("alpha", np.logspace(-4, 2, 50)),
        }


class XGBoostHyperparameters(BaseHyperparameters):
    @classmethod
    def get_hyperparameters(cls, trial: Trial) -> dict:
        return {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [50, 100, 150, 200, 250, 300, 400, 500]
            ),
            "max_depth": trial.suggest_categorical("max_depth", [3, 6, 9, 12]),
            "max_leaves": trial.suggest_categorical("max_leaves", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.01, 0.05, 0.1, 0.2, 0.25, 0.3]
            ),
            # "enable_categorical": trial.suggest_categorical("enable_categorical", [True]),
        }


spaces = {
    "ridge": RidgeHyperparameters,
    "xgb": XGBoostHyperparameters,
}
