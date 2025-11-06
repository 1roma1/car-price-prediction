import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_configuration


def split(
    raw_data_path: str,
    train_data_path: str,
    test_data_path: str,
    test_size: float,
    random_state: int,
):
    df = pd.read_csv(raw_data_path)
    df = df.assign(
        price_usd_bin=pd.Categorical(pd.qcut(df.price_usd, q=10)).codes
    )
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["price_usd_bin"],
    )

    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)
    os.remove(raw_data_path)


if __name__ == "__main__":
    config = load_configuration("config.yaml")
    data_path = Path(config["raw_data_dir"])

    split(
        data_path / config["raw_data"],
        data_path / config["train_data"],
        data_path / config["test_data"],
        config["test_size"],
        config["random_state"],
    )
