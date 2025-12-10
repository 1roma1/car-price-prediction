from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from src.model import ModelRegistry

ModelRegistry.registry["xgb"] = XGBRegressor
ModelRegistry.registry["cb"] = CatBoostRegressor
ModelRegistry.registry["forest"] = RandomForestRegressor
