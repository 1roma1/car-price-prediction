import os
import pandas as pd

from pathlib import Path

from utils import load_configuration, get_schema


def _extract_categories(x, categories, other="другой"):
    for category in categories:
        if category in x:
            return category
    return other


def _split_and_take_first(column):
    return column.str.split().str[0]


def _replace(column, pat, repl):
    return column.str.replace(pat, repl)


class Preprocessor:
    def __init__(self, config=None):
        self.config = config

    def _calculate_means(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        return {column: df[column].mean() for column in columns}

    def _calculate_fillna_values(self, df: pd.DataFrame) -> dict:
        self.fillna_values = self._calculate_means(
            df,
            ["engine_capacity", "mixed_drive_fuel_consumption", "engine_power"],
        )
        self.fillna_values["options"] = " "

    def _outlier_delete(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.year > self.config["min_year"]]
        df = df[df.mileage_km < self.config["max_mileage"]]
        return df

    def preprocess(self, df: pd.DataFrame, test: bool = False) -> pd.DataFrame:
        df = df[self.config["cols"].keys()]

        df = df.assign(
            engine_type=df.engine_type.apply(
                lambda x: _extract_categories(x, self.config["engine_types"])
            ),
            body_type=df.body_type.apply(
                lambda x: _extract_categories(x, self.config["body_types"])
            ),
            engine_capacity=_replace(df.engine_capacity, ",", ".").astype("float32"),
            mixed_drive_fuel_consumption=_replace(
                _split_and_take_first(df.mixed_drive_fuel_consumption), ",", "."
            ).astype("float32"),
        )

        df = self._outlier_delete(df)
        if not test:
            self._calculate_fillna_values(df)
        df = df.fillna(self.fillna_values)

        df = df.drop_duplicates()
        df = df.astype(self.config["cols"])

        return df


if __name__ == "__main__":
    config = load_configuration("configs/config.yaml")
    preprocessing_config = load_configuration("configs/preprocessing_config.yaml")

    raw_train = pd.read_csv(Path(config["raw_data_dir"]) / config["train_data"])
    raw_test = pd.read_csv(Path(config["raw_data_dir"]) / config["test_data"])

    print(f"Raw data shape: Train - {raw_train.shape} Test - {raw_test.shape}")

    schema = get_schema(config["cols"])

    preprocessor = Preprocessor(preprocessing_config)
    preprocessed_train = preprocessor.preprocess(raw_train)
    preprocessed_test = preprocessor.preprocess(raw_test, test=True)

    print(
        f"Preprocessed data shape: Train - {preprocessed_train.shape} Test - {preprocessed_test.shape}"
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
