import pandas as pd

from typing import Literal
from pathlib import Path

from pydantic import BaseModel
from pandantic import Pandantic

from utils import load_configuration


class DataSchema(BaseModel):
    year: int
    mileage_km: int
    engine_capacity: float
    engine_power: float
    mixed_drive_fuel_consumption: float

    brand: str
    options: str
    engine_type: Literal[
        "бензин",
        "дизель",
        "электро",
        "другой",
    ]
    transmission_type: Literal[
        "механика",
        "автомат",
        "вариатор",
        "робот",
    ]
    interior_material: Literal[
        "ткань",
        "натуральная кожа",
        "комбинированные материалы",
        "искусственная кожа",
        "велюр",
        "алькантара",
    ]
    body_type: Literal[
        "внедорожник",
        "седан",
        "универсал",
        "хэтчбек",
        "минивэн",
        "лифтбек",
        "купе",
        "другой",
    ]
    drive_type: Literal[
        "передний привод",
        "подключаемый полный привод",
        "постоянный полный привод",
        "задний привод",
    ]

    price_usd: int


def validate_data(df: pd.DataFrame):
    validator = Pandantic(schema=DataSchema)
    validator.validate(df, errors="raise")


if __name__ == "__main__":
    config = load_configuration("configs/config.yaml")

    train_data_path = Path(config["preprocessed_data_dir"]) / config["train_data"]
    test_data_path = Path(config["preprocessed_data_dir"]) / config["test_data"]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    for path, df in zip((train_data_path, test_data_path), (train, test)):
        try:
            validate_data(df)
            print(f"{path} is successfully validated")
        except ValueError as e:
            print(f"Validation error: {e}")
