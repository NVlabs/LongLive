from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import ModelQuantizationConfig


class QuantizedModule:
    """Base class for all quantized modules."""

    _registry: ClassVar[dict[type[nn.Module], type[nn.Module]]] = {}
    _should_replace_existing_modules_in_model: ClassVar[dict[type[nn.Module], bool]] = (
        {}
    )

    @classmethod
    def is_quantized_module_type(cls, module_type: type[nn.Module]) -> bool:
        """Return True if the given module type is a quantized module."""
        return module_type in cls._registry.values()

    @classmethod
    def get_cls(
        cls,
        high_precision_cls: type[nn.Module],
    ) -> type[nn.Module] | None:
        """Get the quantized module for a given high-precision module."""
        return cls._registry.get(high_precision_cls)

    @classmethod
    def should_replace_existing_modules_in_model(
        cls,
        module_type: type[nn.Module],
    ) -> bool:
        """Determine whether module should be replaced."""
        return cls._should_replace_existing_modules_in_model.get(module_type, False)

    @classmethod
    def register(
        cls,
        high_precision_cls: type[nn.Module],
        *,
        replace_existing_modules_in_registry: bool = False,
        replace_existing_modules_in_model: bool = True,
    ) -> Callable[[type[nn.Module]], type[nn.Module]]:
        """
        Register a new type of quantized module.

        Args:
            high_precision_cls: (`type[nn.Module]`): The high precision module to be
            mapped to a fouroversix quantized module.
            replace_existing_modules_in_registry (bool): determines whether we should
            replace the existing module in the registry.
            replace_existing_modules_in_model (bool): determines whether we should
            replace the existing module in the model including the weights.

        """

        if (
            high_precision_cls in cls._registry
            and not replace_existing_modules_in_registry
        ):
            msg = f"High-precision module {high_precision_cls} is already registered."
            raise ValueError(msg)

        modules_to_delete = []

        for module_cls in cls._registry:
            if high_precision_cls is not None and issubclass(
                high_precision_cls,
                module_cls,
            ):
                if replace_existing_modules_in_registry:
                    modules_to_delete.append(module_cls)
                else:
                    msg = (
                        f"High-precision module {high_precision_cls} is a subclass of "
                        f"{module_cls}, which is already registered."
                    )
                    raise TypeError(msg)

        for module_cls in modules_to_delete:
            del cls._registry[module_cls]

        def inner_wrapper(
            wrapped_cls: type[nn.Module],
        ) -> type[nn.Module]:
            cls._registry[high_precision_cls] = wrapped_cls
            cls._should_replace_existing_modules_in_model[high_precision_cls] = (
                replace_existing_modules_in_model
            )
            return wrapped_cls

        return inner_wrapper


def quantize_model(
    model: nn.Module,
    config: ModelQuantizationConfig,
    **kwargs: dict[str, Any],
) -> None:
    for module_name, module in model.named_modules():
        if (
            module_name == ""
            or module_name in config.modules_to_not_convert
            or not isinstance(module, nn.Module)
        ):
            continue

        module_cls = QuantizedModule.get_cls(type(module))
        should_replace = QuantizedModule.should_replace_existing_modules_in_model(
            type(module),
        )

        if module_cls is None or not should_replace:
            continue

        quantized_module = module_cls(
            module,
            config.get_module_config(module_name),
            **kwargs,
        )
        model.set_submodule(module_name, quantized_module)
