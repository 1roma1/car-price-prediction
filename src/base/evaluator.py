import mlflow
import pandas as pd

from src.model import get_model_class
from src.base.registries import MetricRegistry
from src.base.utils import get_or_create_experiment


class Evaluator:
    def __init__(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame, config: dict
    ):
        self.X_test = X_test
        self.y_test = y_test
        self.config = config

    def evaluate(
        self,
        experiment_name: str,
        estimator_id: str,
        transformer_id: str = None,
    ):
        model_class = get_model_class(self.config["model"])
        model = model_class(**self.config.get("model_params"))
        model.load(estimator_id, transformer_id)
        metrics_fn = {
            metric: MetricRegistry.get(metric)
            for metric in self.config["metrics"]
        }

        experiment = mlflow.set_experiment(
            experiment_id=get_or_create_experiment(experiment_name)
        )
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            y_pred = model.predict(self.X_test)

            scores = {}
            for metric_name, metric_func in metrics_fn.items():
                scores[metric_name] = metric_func(y_pred, self.y_test)

            mlflow.log_metrics(scores)
            model.save(self.X_test[:1])
