# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
# SPDX-License-Identifier: MIT
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/mit-han-lab/fouroversix @ 4c0db4bf3027272c61e32ca7e2cdb9c7fa5bcecb :: src/fouroversix/weight_conversions/conversions.py
# Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: third_party/fouroversix/src/fouroversix/weight_conversions/conversions.py
# End Long-WAM attribution.

from collections.abc import Callable
from typing import ClassVar

from transformers import WeightConverter


class WeightConversions:
    """Base class for weight conversions for quantized models."""

    _registry: ClassVar[dict[str, list[WeightConverter]]] = {}

    @classmethod
    def register(
        cls,
        pre_quantized_model_config_type: str,
    ) -> Callable[[type], list[WeightConverter]]:
        """Register a new type of weight conversion."""
        if pre_quantized_model_config_type in cls._registry:
            msg = f"Model with config {pre_quantized_model_config_type} is \
            already registered."
            raise ValueError(msg)

        def inner_wrapper(
            wrapped_cls: type,
        ) -> list[WeightConverter]:
            weight_conversions = wrapped_cls.get_weight_conversions()
            cls._registry[pre_quantized_model_config_type] = weight_conversions
            return weight_conversions

        return inner_wrapper

    @classmethod
    def get_weight_conversions(
        cls,
        pre_quantized_model_config_type: str,
    ) -> list[WeightConverter]:
        """
        Get the weight conversion for a given model type determined
        by the model config type.
        """
        weight_conversions = cls._registry.get(pre_quantized_model_config_type, None)
        if weight_conversions is None:
            msg = f"Config type {pre_quantized_model_config_type} not supported."
            raise ValueError(msg)
        return weight_conversions
