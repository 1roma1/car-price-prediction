import mlflow
import optuna
import numpy as np
import pandas as pd

from optuna import Study
from optuna.trial import FrozenTrial
from sklearn.model_selection import KFold, StratifiedKFold

from src.model import Model
from src.metrics import get_metrics
from src.models.hyperparam_spaces import spaces
from src.components import ArtifactManager, MlflowManager


class Optimizer:
    def __init__(self, direction, n_trials):
        self.direction = direction
        self.n_trials = n_trials

    def optimize(self, objective_fn, callbacks=[]) -> Study:
        study = optuna.create_study(direction=self.direction)
        study.optimize(objective_fn, n_trials=self.n_trials, callbacks=callbacks)
        return study


class MlflowCallback:
    def __init__(
        self,
        tracking_manager: MlflowManager,
        experiment_id: str,
        nested: bool = True,
        parent_run_id: str = None,
    ):
        self.tracking_manager = tracking_manager
        self.experiment_id = experiment_id
        self.nested = nested
        self.parent_run_id = parent_run_id

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        with self.tracking_manager.start_run(
            experiment_id=self.experiment_id,
            nested=self.nested,
            parent_run_id=self.parent_run_id,
        ):
            mlflow.log_metrics(trial.user_attrs)
            mlflow.log_params(trial.params)


class CrossValidation:
    def __init__(self, X, y, n_folds, stratify=None, stratify_col=None, random_state=1) -> None:
        self.X = X.drop(labels=stratify_col.name, axis=1) if stratify_col is not None else X
        self.y = y
        self.n_folds = n_folds
        self.random_state = random_state

        validator = self._get_validator(stratify)
        self.idxs = [
            (train_idx, val_idx) for train_idx, val_idx in validator.split(X, stratify_col)
        ]

    def _get_validator(self, stratify):
        if stratify:
            validator = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
        else:
            validator = KFold(
                n_folds=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        return validator

    def validate(self, model: Model, metrics: dict):
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
                scores[metric_name] = metric_func(y_pred, y_val)
        return scores


class Trainer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        validator: CrossValidation,
        tracking_manager: MlflowManager,
        optimizer: Optimizer,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train

        self.validator = validator
        self.tracking_manager = tracking_manager
        self.optimizer = optimizer
        self.artifact_manager = ArtifactManager()

    def _process_metrics(self, metrics: dict) -> dict:
        return {metric_name: np.mean(metrics[metric_name]) for metric_name in metrics.keys()}

    def _get_cross_val_score(self, model: Model, metrics) -> dict:
        scores = self.validator.validate(model, metrics)
        return self._process_metrics(scores)

    def _objective(
        self,
        trial,
        model: Model,
        metrics,
        optimization_metric,
    ):
        params = spaces[model.model_name].get_hyperparameters(trial)
        model.set_params(params)

        metrics = self._get_cross_val_score(model, metrics)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return metrics[optimization_metric]

    def train(
        self,
        experiment_name: str,
        model: Model,
        metrics: list,
        optimize: bool = False,
        optimization_metric="rmse",
        save: bool = False,
    ) -> None:
        metrics_fn = get_metrics(metrics)
        experiment = self.tracking_manager.set_experiment(experiment_name)
        with self.tracking_manager.start_run(experiment.experiment_id) as root_run:
            if optimize:
                mlflow_callback = MlflowCallback(
                    self.tracking_manager,
                    experiment.experiment_id,
                    nested=True,
                    parent_run_id=root_run.info.run_id,
                )
                study = self.optimizer.optimize(
                    lambda trial: self._objective(trial, model, metrics_fn, optimization_metric),
                    [mlflow_callback],
                )
                params = study.best_params
                metrics = study.best_trial.user_attrs
            else:
                params = {}
                model.set_params(params)
                metrics = self._get_cross_val_score(model, metrics_fn)

            model.set_params(params)
            model.fit(self.X_train, self.y_train)

            self.artifact_manager.log_artifacts(model, self.X_train, self.y_train)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if save:
                model.save(
                    self.X_train[:5],
                    f"estimator_{root_run.info.run_name}",
                    f"transformer_{root_run.info.run_name}",
                )
