from typing import Any

__all__ = [
    "DummyConfig",
    "DummyModel",
]

class DummyConfig:
    """
    Dummy configuration class
    """
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

class DummyModel:
    """
    Dummy model class
    """
    def __init__(self, **kwargs: Any) -> None:
        self.config = DummyConfig(**kwargs)
