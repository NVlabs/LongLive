import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from fouroversix.quantize import QuantizationConfig
from fouroversix.utils import (
    DataType,
    MatmulBackend,
    QuantizeBackend,
    RoundStyle,
    ScaleRule,
)


@dataclass
class ModuleQuantizationConfig:
    """
    Configuration for quantizing modules with Four Over Six.

    Args:
        activation_scale_rule (ScaleRule | None): The scaling rule to use for activation
            tensors. If not provided, `scale_rule` will be used.
        dtype (DataType): The quantization data type to use for the module. Defaults to
            `DataType.nvfp4`.
        gradient_round_style (RoundStyle | None): The rounding style to use for gradient
            tensors. Defaults to `RoundStyle.stochastic`.
        gradient_scale_rule (ScaleRule | None): The scaling rule to use for gradient
            tensors. If not provided, `scale_rule` will be used.
        keep_master_weights (bool): Whether to keep the master weights. Defaults to
            `False`.
        matmul_backend (MatmulBackend | None): The backend to use for matrix
            multiplications. If not provided, a backend will be selected automatically
            based on the available GPU and the specified options.
        output_dtype (DataType): The data type to use for the module's output. Defaults
            to `DataType.bfloat16`.
        quantize_backend (QuantizeBackend | None): The backend to use for quantization.
            If not provided, a backend will be selected automatically based on the
            available GPU and the specified options.
        scale_rule (ScaleRule): The fallback scaling rule which will be used if any of
            the other scaling rules are not specified.
        weight_scale_2d (bool): Whether to use 2D block scaling for weights. Should be
            set to `True` if the module is used for training.
        weight_scale_rule (ScaleRule | None): The scaling rule to use for weights. If
            not provided, `scale_rule` will be used.

    """

    activation_scale_rule: ScaleRule | None = None
    dtype: DataType = DataType.nvfp4
    gradient_round_style: RoundStyle = RoundStyle.stochastic
    gradient_scale_rule: ScaleRule | None = None
    keep_master_weights: bool = False
    matmul_backend: MatmulBackend | None = None
    output_dtype: DataType = DataType.bfloat16
    quantize_backend: QuantizeBackend | None = None
    scale_rule: ScaleRule = ScaleRule.mse
    weight_scale_2d: bool = False
    weight_scale_rule: ScaleRule | None = None
    activation_chunk_size: int | None = None

    def __post_init__(self) -> None:
        """Convert string values to enums."""

        if isinstance(self.activation_scale_rule, str):
            self.activation_scale_rule = ScaleRule(self.activation_scale_rule)

        if isinstance(self.dtype, str):
            self.dtype = DataType(self.dtype)

        if isinstance(self.gradient_round_style, str):
            self.gradient_round_style = RoundStyle(self.gradient_round_style)

        if isinstance(self.gradient_scale_rule, str):
            self.gradient_scale_rule = ScaleRule(self.gradient_scale_rule)

        if isinstance(self.matmul_backend, str):
            self.matmul_backend = MatmulBackend(self.matmul_backend)

        if isinstance(self.output_dtype, str):
            self.output_dtype = DataType(self.output_dtype)

        if isinstance(self.quantize_backend, str):
            self.quantize_backend = QuantizeBackend(self.quantize_backend)

        if isinstance(self.scale_rule, str):
            self.scale_rule = ScaleRule(self.scale_rule)

        if isinstance(self.weight_scale_rule, str):
            self.weight_scale_rule = ScaleRule(self.weight_scale_rule)

        self.activation_scale_rule = self.activation_scale_rule or self.scale_rule
        self.gradient_scale_rule = self.gradient_scale_rule or self.scale_rule
        self.weight_scale_rule = self.weight_scale_rule or self.scale_rule

    def get_activation_config(self, **kwargs: dict[str, Any]) -> QuantizationConfig:
        """Return the quantization configuration for the activation tensors."""
        return QuantizationConfig(
            backend=self.quantize_backend,
            dtype=self.dtype,
            scale_rule=self.activation_scale_rule,
            **kwargs,
        )

    def get_gradient_config(self, **kwargs: dict[str, Any]) -> QuantizationConfig:
        """Return the quantization configuration for the gradient tensors."""
        return QuantizationConfig(
            backend=self.quantize_backend,
            dtype=self.dtype,
            round_style=self.gradient_round_style,
            scale_rule=self.gradient_scale_rule,
            **kwargs,
        )

    def get_weight_config(self, **kwargs: dict[str, Any]) -> QuantizationConfig:
        """Return the quantization configuration for the weight tensors."""
        return QuantizationConfig(
            backend=self.quantize_backend,
            block_scale_2d=self.weight_scale_2d,
            dtype=self.dtype,
            scale_rule=self.weight_scale_rule,
            **kwargs,
        )


@dataclass
class ModelQuantizationConfig(ModuleQuantizationConfig):
    """
    Configuration for quantizing a model with Four Over Six.

    Args:
        activation_scale_rule (ScaleRule | None): The scaling rule to use for activation
            tensors. If not provided, `scale_rule` will be used.
        dtype (DataType): The quantization data type to use for the module. Defaults to
            `DataType.nvfp4`.
        gradient_round_style (RoundStyle | None): The rounding style to use for gradient
            tensors. Defaults to `RoundStyle.stochastic`.
        gradient_scale_rule (ScaleRule | None): The scaling rule to use for gradient
            tensors. If not provided, `scale_rule` will be used.
        keep_master_weights (bool): Whether to keep the master weights. Defaults to
            `False`.
        matmul_backend (MatmulBackend | None): The backend to use for matrix
            multiplications. If not provided, a backend will be selected automatically
            based on the available GPU and the specified options.
        output_dtype (DataType): The data type to use for the module's output. Defaults
            to `DataType.bfloat16`.
        quantize_backend (QuantizeBackend | None): The backend to use for quantization.
            If not provided, a backend will be selected automatically based on the
            available GPU and the specified options.
        scale_rule (ScaleRule): The fallback scaling rule which will be used if any of
            the other scaling rules are not specified.
        weight_scale_2d (bool): Whether to use 2D block scaling for weights. Should be
            set to `True` if the module is used for training.
        weight_scale_rule (ScaleRule | None): The scaling rule to use for weights. If
            not provided, `scale_rule` will be used.

        module_config_overrides (dict[str, ModuleQuantizationConfig]): A mapping of
            module names to quantization configurations to use for each module. If a
            module is not specified, the attributes from this class will be used.
        modules_to_not_convert (list[str]): A list of module names that should not be
            quantized.

    """

    module_config_overrides: dict[str, ModuleQuantizationConfig] = field(
        default_factory=dict,
    )
    modules_to_not_convert: list[str] = field(default_factory=lambda: ["lm_head"])

    def __post_init__(self) -> None:
        """Convert module config overrides to ModuleQuantizationConfig instances."""

        super().__post_init__()

        if self.module_config_overrides is not None:
            for module_name, module_config in self.module_config_overrides.items():
                if isinstance(module_config, dict):
                    self.module_config_overrides[module_name] = (
                        ModuleQuantizationConfig(**module_config)
                    )

    def get_module_config(self, module_name: str) -> ModuleQuantizationConfig:
        """Return the quantization configuration for a given module."""
        return (
            self.module_config_overrides.get(module_name, self)
            if self.module_config_overrides is not None
            else self
        )

    def __hash__(self) -> str:
        """Return a hash of the configuration."""
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True).encode(),
        ).hexdigest()
