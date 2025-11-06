import os
import sqlite3
import pandas as pd

from pathlib import Path


from utils import load_configuration


def fetch_data_from_db() -> None:
    config = load_configuration("config.yaml")

    sql_stmt = """
    SELECT price_usd, brand, engine_capacity, engine_type,
        transmission_type, interior_material, body_type,
        drive_type, year, mileage_km, mixed_drive_fuel_consumption,
        engine_power, options
    FROM car_ad
    """

    with sqlite3.connect(config["data_source"]) as conn:
        data = pd.read_sql(sql_stmt, conn)
    os.makedirs(config["raw_data_dir"], exist_ok=True)
    data.to_csv(
        Path(config["raw_data_dir"]) / config["raw_data"],
        index=False,
    )


if __name__ == "__main__":
    fetch_data_from_db()
