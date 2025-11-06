import mlflow
import numpy as np

from typing import Any, Dict, Callable

from mlflow.models import infer_signature
from catboost import CatBoostRegressor
from skl2onnx import to_onnx


class TrainRegistry:
    registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Any) -> Any:
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_model(cls, name: str) -> Any:
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise ValueError(f"There is no model: {name}, available: {cls.registry.keys()}")


class Model:
    def __init__(self, model_name, transformer=None, log_transform=False, **kwargs):
        self.model_name = model_name
        self.log_transform = log_transform
        self.estimator = TrainRegistry.get_model(model_name)()
        self.transformer = transformer

        self.cat_features = kwargs.get("cat_features")

    def fit(self, X, y, **kwargs):
        X_val, y_val = kwargs.get("X_val"), kwargs.get("y_val")
        if self.transformer:
            self.transformer.fit(X, y.values.reshape(-1))
            X = self.transformer.transform(X)
            if X_val is not None:
                X_val = self.transformer.transform(kwargs.get("X_val"))

        if self.log_transform:
            y = np.log1p(y)

        if self.model_name == "xgb":
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.estimator.fit(X, y, eval_set=eval_set, verbose=kwargs.get("verbose", False))
        elif self.model_name == "cb":
            self.estimator.fit(X, y, cat_features=self.cat_features)
        else:
            self.estimator.fit(X, y)

    def predict(self, X):
        if self.transformer:
            X = self.transformer.transform(X)
        y_pred = self.estimator.predict(X)
        return np.expm1(y_pred) if self.log_transform else y_pred

    def set_params(self, params):
        if self.model_name == "cb":
            self.estimator = CatBoostRegressor(**params)
        else:
            self.estimator = self.estimator.set_params(**params)

    def save(self, X, estimator_name, transformer_name=None) -> None:
        if self.transformer:
            X_tr = self.transformer.transform(X)
            signature = infer_signature(X, X_tr)

            mlflow.sklearn.log_model(
                self.transformer,
                name=transformer_name,
                pyfunc_predict_fn="transform",
                signature=signature,
            )
            X = self.transformer.transform(X)

        y_pred = self.estimator.predict(X)
        signature = infer_signature(X, y_pred)

        if self.model_name == "xgb":
            mlflow.xgboost.log_model(
                self.estimator, name=estimator_name, model_format="ubj", signature=signature
            )
        elif self.model_name == "cb":
            mlflow.catboost.log_model(self.estimator, name=estimator_name)
        else:
            mlflow.sklearn.log_model(self.estimator, name=estimator_name, signature=signature)

    def save_onnx(self, X, estimator_name, transformer_name=None):
        if self.transformer:
            onnx_transformer = to_onnx(self.transformer, X)
            mlflow.onnx.log_model(onnx_transformer, name=transformer_name)
            X = self.transformer.transform(X)
        onnx_estimator = to_onnx(self.estimator, X)
        mlflow.onnx.log_model(onnx_estimator, name=estimator_name)

    def load(self, model_id, transformer_id=None):
        if transformer_id:
            self.transformer = mlflow.sklearn.load_model(f"models:/{transformer_id}")

        if self.model_name == "xgb":
            self.estimator = mlflow.xgboost.load_model(f"models:/{model_id}")
        else:
            self.estimator = mlflow.sklearn.load_model(f"models:/{model_id}")
