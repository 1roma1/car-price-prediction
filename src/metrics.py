from sklearn.metrics import root_mean_squared_error


def get_metrics(metric_names):
    metrics_map = {"rmse": root_mean_squared_error}

    return {
        metric_name: metrics_map[metric_name] for metric_name in metric_names
    }
