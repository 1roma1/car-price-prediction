import mlflow
import tempfile
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from mlflow.entities import Experiment


class MlflowManager:
    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)

    def _get_or_create_experiment(self, experiment_name) -> str:
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)

    def set_experiment(self, experiment_name: str) -> Experiment:
        experiment_id = self._get_or_create_experiment(experiment_name=experiment_name)
        return mlflow.set_experiment(experiment_id=experiment_id)

    def start_run(self, experiment_id, nested=False, parent_run_id=None):
        return mlflow.start_run(
            experiment_id=experiment_id,
            nested=nested,
            parent_run_id=parent_run_id,
        )


class ArtifactManager:
    LINEAR_MODELS = ["linreg", "ridge", "lasso"]
    XGB = ["xgb"]

    def log_artifacts(self, model, X, y):
        y_pred = model.predict(X)
        if model.estimator_name in ArtifactManager.LINEAR_MODELS:
            self._plot_feature_importance_lr(
                model.transformer.get_feature_names_out(), model.estimator.coef_
            )
        self._plot_prediction_error(y, y_pred)
        self._get_prediction_sample(y, y_pred)

    def _plot_feature_importance_lr(self, features, values):
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.barplot(x=values, y=features, ax=ax)
        ax.set_title("Feature Importance")
        plt.close(fig)
        mlflow.log_figure(fig, "feature_importance.png")

    def _plot_prediction_error(self, y_test, y_pred, plot_size=(10, 8)):
        fig, ax = plt.subplots(figsize=plot_size)
        ax.scatter(y_pred, y_test - y_pred)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_title("Prediction Error Plot", fontsize=14)
        ax.set_xlabel("Predicted Values", fontsize=12)
        ax.set_ylabel("Errors", fontsize=12)
        plt.tight_layout()
        plt.close(fig)
        mlflow.log_figure(fig, "prediction_error.png")

    def _get_prediction_sample(self, y_test, y_pred):
        residuals = y_test - y_pred
        df = pd.DataFrame(
            {
                "True Values": y_test[:5],
                "Predicted Values": y_pred[:5],
                "Residuals": residuals[:5],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "prediction_sample.csv")
            df.to_csv(Path(tmp_dir, "prediction_sample.csv"), index=False)

            mlflow.log_artifact(path)
