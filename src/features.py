from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline


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
                "scaler",
                StandardScaler(),
                [
                    "year",
                    "mileage_km",
                    "engine_capacity",
                    "engine_power",
                    "mixed_drive_fuel_consumption",
                    "options",
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


transfomrers_map = {"ss": get_standard_scaler_transformer}
