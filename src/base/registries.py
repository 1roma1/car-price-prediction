from typing import Any, Callable, Optional
from optuna.trial import Trial


class ModelRegistry:
    registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Any) -> Any:
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_model(cls, name: str) -> Any:
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise ValueError(
                f"There is no model: {name}, available: {cls.registry.keys()}"
            )


class TransformerRegistry:
    registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_func: Any) -> Any:
            cls.registry[name] = wrapped_func
            return wrapped_func

        return inner_wrapper

    @classmethod
    def get_transformer(cls, name: str) -> Any:
        if name in cls.registry:
            return cls.registry[name]()
        else:
            raise ValueError(
                f"There is no model: {name}, available: {cls.registry.keys()}"
            )


class HyperparameterRegistry:
    registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_func: Any) -> Any:
            cls.registry[name] = wrapped_func
            return wrapped_func

        return inner_wrapper

    @classmethod
    def get_hyperparameters(cls, name: str, trial: Trial) -> Any:
        if name in cls.registry:
            return cls.registry[name](trial)
        else:
            raise ValueError(
                f"There is no model: {name}, available: {cls.registry.keys()}"
            )


class MetricRegistry:
    registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_func: Any) -> Any:
            cls.registry[name] = wrapped_func
            return wrapped_func

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Any:
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise ValueError(
                f"There is no element: {name}, available: {cls.registry.keys()}"
            )
