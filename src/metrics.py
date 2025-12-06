from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

from src.base.registries import MetricRegistry


MetricRegistry.registry["r2_score"] = r2_score
MetricRegistry.registry["mae"] = mean_absolute_error
MetricRegistry.registry["rmse"] = root_mean_squared_error
