from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(f"{name} already registered in {self.name}")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Unknown {self.name}: {name}. Available: {available}")
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> T:
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._registry)


MODELS = Registry("models")
LOSSES = Registry("losses")
DATASETS = Registry("datasets")
EVALUATORS = Registry("evaluators")
PREDICTORS = Registry("predictors")
