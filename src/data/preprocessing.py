import os
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.base.utils import load_yaml, load_json
from src.base.registries import TransformerRegistry


def extract_categories(
    string: str, categories: list, other: str = "другой"
) -> str:
    for category in categories:
        if category in string:
            return category
    return other


def split_and_take_first(series: pd.Series) -> pd.Series:
    return series.str.split().str[0]


def replace(column: pd.Series, old: str, new: str) -> pd.Series:
    return column.str.replace(old, new)


class Preprocessor:
    def __init__(self, features, config=None):
        self.features = features
        self.config = config

    def _calculate_means(self, df: pd.DataFrame, columns) -> dict:
        return {column: df[column].mean() for column in columns}

    def _calculate_fillna_values(self, df: pd.DataFrame) -> dict:
        self.fillna_values = self._calculate_means(
            df,
            [
                "engine_capacity",
                "mixed_drive_fuel_consumption",
                "engine_power",
            ],
        )
        self.fillna_values["options"] = " "

    def _outlier_delete(
        self, df: pd.DataFrame, test: bool = False
    ) -> pd.DataFrame:
        df = df[df.year > self.config["min_year"]]
        df = df[df.mileage_km < self.config["max_mileage"]]
        if not test:
            self.anomaly_detector = Pipeline(
                [
                    (
                        "preprocessor",
                        TransformerRegistry.get_transformer("ss"),
                    ),
                    (
                        "estimator",
                        IsolationForest(n_estimators=500, contamination=0.01),
                    ),
                ]
            )
            self.anomaly_detector.fit(df, df["price_usd"])

        df["is_anomaly"] = self.anomaly_detector.predict(df)
        return df[df["is_anomaly"] == 1].drop(labels="is_anomaly", axis=1)

    def preprocess(self, df: pd.DataFrame, test: bool = False) -> pd.DataFrame:
        df = df[self.config["cols"]]

        df = df.assign(
            engine_type=df.engine_type.apply(
                lambda x: extract_categories(x, self.features["engine_types"])
            ),
            body_type=df.body_type.apply(
                lambda x: extract_categories(x, self.features["body_types"])
            ),
            engine_capacity=replace(df.engine_capacity, ",", ".").astype(
                "float32"
            ),
            mixed_drive_fuel_consumption=replace(
                split_and_take_first(df.mixed_drive_fuel_consumption),
                ",",
                ".",
            ).astype("float32"),
        )

        if not test:
            self._calculate_fillna_values(df)
        df = df.fillna(self.fillna_values)
        df = df.drop_duplicates()
        df = self._outlier_delete(df, test)

        return df


if __name__ == "__main__":
    config = load_yaml("configs/config.yaml")
    preprocessing_config = load_yaml("configs/preprocessing_config.yaml")
    features = load_json("data/features.json")

    raw_train = pd.read_csv(Path(config["raw_data_dir"], config["train_data"]))
    raw_test = pd.read_csv(Path(config["raw_data_dir"], config["test_data"]))

    print(f"Raw data shape: Train - {raw_train.shape} Test - {raw_test.shape}")

    preprocessor = Preprocessor(features, preprocessing_config)
    preprocessed_train = preprocessor.preprocess(raw_train)
    preprocessed_test = preprocessor.preprocess(raw_test, test=True)

    print(
        f"Preprocessed data shape: Train - {preprocessed_train.shape} "
        f"Test - {preprocessed_test.shape}"
    )

    os.makedirs(config["preprocessed_data_dir"], exist_ok=True)
    preprocessed_train.to_csv(
        Path(config["preprocessed_data_dir"]) / config["train_data"],
        index=False,
    )
    preprocessed_test.to_csv(
        Path(config["preprocessed_data_dir"]) / config["test_data"],
        index=False,
    )
