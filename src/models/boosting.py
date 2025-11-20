from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.model import ModelRegistry

ModelRegistry.registry["xgb"] = XGBRegressor
ModelRegistry.registry["cb"] = CatBoostRegressor
