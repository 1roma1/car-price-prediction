import mlflow
import numpy as np
import pandas as pd

from typing import Any, Callable, Optional
from abc import ABC, abstractmethod

from sklearn.compose import ColumnTransformer
from mlflow.models import infer_signature
from catboost import CatBoostRegressor

from src.features import Transformers


class ModelRegistry:
    registry: dict[str, Any] = {}

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


class BaseModel(ABC):
    def __init__(
        self,
        estimator_name: str,
        transformer_name: Optional[str] = None,
        log_transform: bool = False,
    ) -> None:
        self.estimator_name = estimator_name
        self.transformer_name = transformer_name
        self.log_transform = log_transform
        self.estimator = ModelRegistry.get_model(estimator_name)()
        self.transformer = (
            Transformers.get_transformer(transformer_name) if transformer_name is not None else None
        )

    def _prepare_X(
        self, X: np.ndarray | pd.DataFrame, X_val: Optional[np.ndarray | pd.DataFrame] = None
    ) -> tuple:
        if self.transformer:
            X = self.transformer.transform(X)
            X = pd.DataFrame(X, columns=self.transformer.get_feature_names_out())
            if hasattr(self, "schema"):
                X = X.astype(self.schema)
            if X_val is not None:
                X_val = self.transformer.transform(X_val)
                X_val = pd.DataFrame(X_val, columns=self.transformer.get_feature_names_out())
                if hasattr(self, "schema"):
                    X_val = X_val.astype(self.schema)
        return X, X_val

    def _prepare_y(
        self, y: np.ndarray | pd.Series, y_val: Optional[np.ndarray | pd.Series] = None
    ) -> tuple:
        if self.log_transform:
            y = np.log1p(y)
            y_val = np.log1p(y_val) if y_val is not None else y_val
        return y, y_val

    def fit_transformer(self, X: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> None:
        if self.transformer:
            self.transformer.fit(X, y)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
    ) -> None:
        self.fit_transformer(X, y)

        X, X_val = self._prepare_X(X, X_val)
        y, y_val = self._prepare_y(y, y_val)

        self.fit_estimator(X, y, X_val, y_val)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X, _ = self._prepare_X(X)
        y_pred = self.estimator.predict(X)
        return np.expm1(y_pred) if self.log_transform else y_pred

    @abstractmethod
    def fit_estimator(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
    ) -> None:
        pass

    @abstractmethod
    def set_params(self, params: dict) -> None:
        pass

    @abstractmethod
    def save(self, X_sample: np.ndarray | pd.DataFrame) -> None:
        pass

    @abstractmethod
    def load(self, model_id: str, transformer_id: Optional[str] = None) -> None:
        pass


class SklearnModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        transformer: ColumnTransformer = None,
        log_transform: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model_name, transformer, log_transform)

    def fit_estimator(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
    ) -> None:
        X = X.astype(np.float32)
        self.estimator.fit(X, y)

    def set_params(self, params: dict) -> None:
        self.estimator = self.estimator.set_params(**params)

    def save(self, X_sample: np.ndarray | pd.DataFrame) -> None:
        if self.transformer:
            X = self.transformer.transform(X_sample)
            signature = infer_signature(X_sample, X)

            mlflow.sklearn.log_model(
                self.transformer,
                self.transformer_name,
                pyfunc_predict_fn="transform",
                signature=signature,
            )
            X = pd.DataFrame(X, columns=self.transformer.get_feature_names_out()).astype(np.float32)

        y_pred = self.predict(X)
        signature = infer_signature(X, y_pred)

        mlflow.sklearn.log_model(self.estimator, self.estimator_name, signature=signature)

    def load(self, model_id: str, transformer_id: Optional[str] = None) -> None:
        if transformer_id:
            self.transformer = mlflow.sklearn.load_model(f"runs:/{transformer_id}")

        self.estimator = mlflow.sklearn.load_model(f"runs:/{model_id}")


class XGBModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        transformer: ColumnTransformer = None,
        log_transform: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model_name, transformer, log_transform)
        self.schema = kwargs.get("schema")

    def fit_estimator(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
    ) -> None:
        X = X.astype(self.schema)
        if X_val is not None:
            X_val = X_val.astype(self.schema)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.estimator.fit(X, y, eval_set=eval_set, verbose=False)

    def set_params(self, params: dict) -> None:
        self.estimator = self.estimator.set_params(**params)

    def save(self, X_sample: np.ndarray | pd.DataFrame) -> None:
        if self.transformer:
            X_tr = self.transformer.transform(X_sample)
            signature = infer_signature(X_sample, X_tr)

            mlflow.sklearn.log_model(
                self.transformer,
                self.transformer_name,
                pyfunc_predict_fn="transform",
                signature=signature,
            )

        y_pred = self.estimator.predict(X_tr)
        signature = infer_signature(X_tr, y_pred)

        mlflow.xgboost.log_model(
            self.estimator, self.estimator_name, model_format="ubj", signature=signature
        )

    def load(self, model_id: str, transformer_id: Optional[str] = None) -> None:
        if transformer_id:
            self.transformer = mlflow.sklearn.load_model(f"runs:/{transformer_id}")

        self.estimator = mlflow.xgboost.load_model(f"runs:/{model_id}")


class CBModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        transformer: ColumnTransformer = None,
        log_transform: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model_name, transformer, log_transform)
        self.cat_features = kwargs.get("cat_features")
        self.schema = kwargs.get("schema")

    def fit_estimator(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.Series] = None,
    ) -> None:
        self.estimator.fit(X, y, cat_features=self.cat_features)

    def set_params(self, params: dict) -> None:
        self.estimator = CatBoostRegressor(**params)

    def save(self, X_sample: np.ndarray | pd.DataFrame) -> None:
        if self.transformer:
            X_tr = self.transformer.transform(X_sample)
            signature = infer_signature(X_sample, X_tr)

            mlflow.sklearn.log_model(
                self.transformer,
                self.transformer_name,
                pyfunc_predict_fn="transform",
                signature=signature,
            )

        y_pred = self.estimator.predict(X_tr)
        signature = infer_signature(X_tr, y_pred)

        mlflow.catboost.log_model(self.estimator, self.estimator_name, signature=signature)

    def load(self, model_id: str, transformer_id: Optional[str] = None) -> None:
        if transformer_id:
            self.transformer = mlflow.sklearn.load_model(f"runs:/{transformer_id}")

        self.estimator = mlflow.catboost.load_model(f"runs:/{model_id}")


def get_model_class(model_name: str) -> BaseModel:
    match model_name:
        case "cb":
            return CBModel
        case "xgb":
            return XGBModel
        case _:
            return SklearnModel
