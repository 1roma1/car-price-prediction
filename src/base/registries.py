from typing import Any, Callable
from optuna.trial import Trial


class BaseRegistry:
    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Any) -> Any:
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Any:
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise ValueError(
                f"There is no element: {name}, "
                f"available: {cls.registry.keys()}"
            )


class ModelRegistry(BaseRegistry):
    registry: dict[str, Any] = {}


class TransformerRegistry:
    registry: dict[str, Any] = {}


class MetricRegistry:
    registry: dict[str, Any] = {}


class HyperparameterRegistry:
    registry: dict[str, Any] = {}

    @classmethod
    def get(cls, name: str, trial: Trial) -> Any:
        if name in cls.registry:
            return cls.registry[name](trial)
        else:
            raise ValueError(
                f"There is no element: {name}, "
                f"available: {cls.registry.keys()}"
            )
