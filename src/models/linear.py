from sklearn.linear_model import (
    Ridge,
    Lasso,
    LinearRegression,
    BayesianRidge,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor,
)

from src.base.registries import ModelRegistry

ModelRegistry.registry["linreg"] = LinearRegression
ModelRegistry.registry["ridge"] = Ridge
ModelRegistry.registry["lasso"] = Lasso
ModelRegistry.registry["bayesian"] = BayesianRidge
ModelRegistry.registry["huber"] = HuberRegressor
ModelRegistry.registry["ransac"] = RANSACRegressor
ModelRegistry.registry["theilsen"] = TheilSenRegressor
