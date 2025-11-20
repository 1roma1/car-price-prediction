import os
import sqlite3
import argparse
import pandas as pd

from pathlib import Path

from utils import load_configuration


def fetch_data_from_db(date: str) -> None:
    config = load_configuration("configs/config.yaml")

    sql_stmt = """
    SELECT date(substr(refreshed_at, 0, 11)) as date, price_usd, brand, engine_capacity, engine_type,
        transmission_type, interior_material, body_type,
        drive_type, year, mileage_km, mixed_drive_fuel_consumption,
        engine_power, coalesce(options, '') as options
    FROM car_ad
    WHERE date(substr(refreshed_at, 0, 11)) < date("{date}")
    """.format(date=date)

    with sqlite3.connect(config["data_source"]) as conn:
        data = pd.read_sql(sql_stmt, conn)
    os.makedirs(config["raw_data_dir"], exist_ok=True)
    data.to_csv(
        Path(config["raw_data_dir"]) / config["raw_data"],
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Data Ingestion")
    parser.add_argument("--date", type=str)

    fetch_data_from_db(vars(parser.parse_args())["date"])
