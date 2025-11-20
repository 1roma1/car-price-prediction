import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Any, Callable


class Transformers:
    registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_func: Any) -> Any:
            cls.registry[name] = wrapped_func
            return wrapped_func

        return inner_wrapper

    @classmethod
    def get_transformer(cls, name: str) -> Any:
        if name in cls.registry:
            return cls.registry[name]()
        else:
            raise ValueError(f"There is no model: {name}, available: {cls.registry.keys()}")


class CountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, delimiter=" "):
        self.delimiter = delimiter

    def fit(self, X, y=None):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            self.feature_names = list(X.columns)
        return self

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values

        X = X.tolist()

        if isinstance(X[0], list):
            X = [[len(str(item).split(self.delimiter)) for item in row] for row in X]
        else:
            X = [len(str(item).split(self.delimiter)) for item in X]

        return np.array(X, dtype=np.int32)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names, dtype=object)


@Transformers.register("ss")
def get_standard_scaler_transformer():
    return ColumnTransformer(
        [
            (
                "brand_enc",
                Pipeline(
                    [
                        ("target_enc", TargetEncoder(target_type="continuous")),
                        ("ss", StandardScaler()),
                    ]
                ),
                ["brand"],
            ),
            (
                "options_enc",
                Pipeline(
                    [
                        ("count_enc", CountTransformer(delimiter="|")),
                        ("ss", StandardScaler()),
                    ]
                ),
                ["options"],
            ),
            (
                "scaler",
                StandardScaler(),
                [
                    "year",
                    "mileage_km",
                    "engine_capacity",
                    "engine_power",
                    "mixed_drive_fuel_consumption",
                ],
            ),
            (
                "oh",
                OneHotEncoder(
                    min_frequency=500, handle_unknown="infrequent_if_exist", sparse_output=False
                ),
                [
                    "engine_type",
                    "transmission_type",
                    "interior_material",
                    "body_type",
                    "drive_type",
                ],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@Transformers.register("ct")
def get_column_transformer():
    return ColumnTransformer(
        [
            ("options_enc", CountTransformer(delimiter="|"), ["options"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
