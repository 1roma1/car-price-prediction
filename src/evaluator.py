import mlflow
import pandas as pd

from src.model import Model
from src.metrics import get_metrics
from src.components import ArtifactManager, MlflowManager


class Evaluator:
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, tracking_manager: MlflowManager):
        self.X_test = X_test
        self.y_test = y_test

        self.tracking_manager = tracking_manager
        self.artifact_manager = ArtifactManager()

    def evaluate(self, experiment_name: str, model: Model, metrics: list, save: bool):
        metrics_fn = get_metrics(metrics)
        experiment = self.tracking_manager.set_experiment(experiment_name)
        with self.tracking_manager.start_run(experiment.experiment_id) as run:
            y_pred = model.predict(self.X_test)

            scores = {}
            for metric_name, metric_func in metrics_fn.items():
                scores[metric_name] = metric_func(y_pred, self.y_test)

            self.artifact_manager.log_artifacts(model, self.X_test, self.y_test)
            mlflow.log_metrics(scores)

            if save:
                model.save_onnx(
                    self.X_test[:1],
                    f"estimator_{run.info.run_name}",
                    f"transformer_{run.info.run_name}",
                )
