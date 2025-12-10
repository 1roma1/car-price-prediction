from sklearn.tree import DecisionTreeRegressor

from src.base.registries import ModelRegistry

ModelRegistry.registry["tree"] = DecisionTreeRegressor
