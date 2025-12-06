import mlflow
import optuna
import numpy as np
import pandas as pd

from optuna import Study
from sklearn.model_selection import KFold

from src.model import BaseModel, get_model_class
from src.callbacks import MlflowCallback

from src.base.registries import (
    MetricRegistry,
    HyperparameterRegistry,
)
from src.base.utils import get_or_create_experiment


class Optimizer:
    def __init__(self, direction, n_trials, optimization_metric):
        self.direction = direction
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric

    def optimize(self, objective_fn, callbacks=[]) -> Study:
        study = optuna.create_study(direction=self.direction)
        study.optimize(
            objective_fn, n_trials=self.n_trials, callbacks=callbacks
        )
        return study


class CrossValidation:
    def __init__(self, X, y, n_folds, random_state=1) -> None:
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.random_state = random_state

        validator = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        self.idxs = [
            (train_idx, val_idx) for train_idx, val_idx in validator.split(X)
        ]

    def validate(self, model: BaseModel, metrics: dict):
        scores = {metric_name: [] for metric_name in metrics.keys()}
        for train_idx, val_idx in self.idxs:
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
            X_train, y_train, X_val, y_val = (
                self.X.iloc[train_idx],
                self.y.iloc[train_idx],
                self.X.iloc[val_idx],
                self.y.iloc[val_idx],
            )
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            y_pred = model.predict(X_val)

            for metric_name, metric_func in metrics.items():
                scores[metric_name].append(metric_func(y_pred, y_val))
        return scores


class Trainer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        config: dict,
        validator: CrossValidation,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.config = config

        self.validator = validator
        # self.artifact_manager = ArtifactManager()
        self.optimizer = (
            Optimizer(**config.get("optimizer_params"))
            if config.get("optimizer_params")
            else None
        )

    def _process_metrics(self, metrics: dict) -> dict:
        return {
            metric_name: np.mean(metrics[metric_name])
            for metric_name in metrics.keys()
        }

    def _get_cross_val_score(self, model: BaseModel, metrics) -> dict:
        scores = self.validator.validate(model, metrics)
        return self._process_metrics(scores)

    def _objective(self, trial, model: BaseModel, metrics):
        params = HyperparameterRegistry.get_hyperparameters(
            model.estimator_name, trial
        )
        model.set_params(params)

        metrics = self._get_cross_val_score(model, metrics)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return metrics[self.optimizer.optimization_metric]

    def train(self, experiment_name: str) -> None:
        model_class = get_model_class(self.config["model"])
        model = model_class(**self.config.get("model_params"))
        metrics_fn = {
            metric: MetricRegistry.get(metric)
            for metric in self.config["metrics"]
        }

        experiment = mlflow.set_experiment(
            experiment_id=get_or_create_experiment(experiment_name)
        )
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            if self.optimizer:
                study = self.optimizer.optimize(
                    lambda trial: self._objective(trial, model, metrics_fn),
                    [MlflowCallback()],
                )
                params = study.best_params
                metrics = study.best_trial.user_attrs
            else:
                params = {}
                model.set_params(params)
                metrics = self._get_cross_val_score(model, metrics_fn)

            model.set_params(params)
            model.fit(self.X_train, self.y_train)

            # self.artifact_manager.log_artifacts(
            #     model, self.X_train, self.y_train
            # )

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            model.save(self.X_train.iloc[:2, :])
