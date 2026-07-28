from typing import Any, Callable, Dict


class _BaseRegistry:
    _registry: Dict[str, Callable[..., Any]] = {}
    _kind: str = "entry"

    @classmethod
    def register(cls, name: str):
        def decorator(factory_fn: Callable[..., Any]):
            cls._registry[name] = factory_fn
            return factory_fn

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(
                f"{cls._kind.title()} '{name}' not found in registry. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry.keys())


class DataModuleRegistry(_BaseRegistry):
    """
    Registry for model-specific data modules (processor, tokenizer, Dataset, Collator).
    See registry_readme.md for detailed usage.
    """

    _registry: Dict[str, Callable[..., Any]] = {}
    _kind = "data module"

    @classmethod
    def register_module(cls, name: str):
        return cls.register(name)

    @classmethod
    def get_module(cls, name: str, **kwargs):
        return cls.get(name, **kwargs)


class DatasetRegistry(_BaseRegistry):
    """
    Registry for dataset factory functions that return DataArguments.
    See registry_readme.md for detailed usage.
    """

    _registry: Dict[str, Callable[..., Any]] = {}
    _kind = "dataset"

    @classmethod
    def register_dataset(cls, name: str):
        return cls.register(name)

    @classmethod
    def get_dataset(cls, name: str, **kwargs):
        return cls.get(name, **kwargs)


class VLMModelRegistry(_BaseRegistry):
    _registry: Dict[str, Callable[..., Any]] = {}
    _kind = "vlm model"

    @classmethod
    def register_model(cls, name: str):
        return cls.register(name)

    @classmethod
    def get_model(cls, name: str, **kwargs):
        return cls.get(name, **kwargs)


class SAMDataModuleRegistry(_BaseRegistry):
    _registry: Dict[str, Callable[..., Any]] = {}
    _kind = "sam data module"

    @classmethod
    def register_sam_data_module(cls, name: str):
        return cls.register(name)

    @classmethod
    def get_sam_data_module(cls, name: str, **kwargs):
        return cls.get(name, **kwargs)


class SAMModelRegistry(_BaseRegistry):
    _registry: Dict[str, Callable[..., Any]] = {}
    _kind = "sam model"

    @classmethod
    def register_sam(cls, name: str):
        return cls.register(name)

    @classmethod
    def get_sam(cls, name: str, **kwargs):
        return cls.get(name, **kwargs)


class VisionEncoderRegistry(_BaseRegistry):
    _registry: Dict[str, Callable[..., Any]] = {}
    _kind = "vision encoder"

    @classmethod
    def register_encoder(cls, name: str):
        return cls.register(name)

    @classmethod
    def get_encoder(cls, name: str, **kwargs) -> "VisionEncoderBase":
        return cls.get(name, **kwargs)
