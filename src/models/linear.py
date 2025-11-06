from sklearn.linear_model import (
    Ridge,
    Lasso,
    LinearRegression,
    BayesianRidge,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor,
)

from src.model import TrainRegistry

TrainRegistry.registry["linreg"] = LinearRegression
TrainRegistry.registry["ridge"] = Ridge
TrainRegistry.registry["lasso"] = Lasso
TrainRegistry.registry["bayesian"] = BayesianRidge
TrainRegistry.registry["huber"] = HuberRegressor
TrainRegistry.registry["ransac"] = RANSACRegressor
TrainRegistry.registry["theilsen"] = TheilSenRegressor
