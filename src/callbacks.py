import mlflow
from optuna import Study
from optuna.trial import FrozenTrial


class MlflowCallback:
    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        with mlflow.start_run(nested=True):
            mlflow.log_metrics(trial.user_attrs)
            mlflow.log_params(trial.params)
