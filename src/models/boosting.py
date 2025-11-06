from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.model import TrainRegistry

TrainRegistry.registry["xgb"] = XGBRegressor
TrainRegistry.registry["cb"] = CatBoostRegressor
