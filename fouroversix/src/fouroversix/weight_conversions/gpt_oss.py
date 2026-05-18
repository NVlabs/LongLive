import torch
from fouroversix import DataType, ScaleRule
from fouroversix.quantize import QuantizedTensor
from transformers import ConversionOps, GptOssConfig, WeightConverter

from .conversions import WeightConversions


class FourOverSixGptOssDeserialize(ConversionOps):
    """Fouroversix deserializer for gpt oss model."""

    def __init__(
        self,
        dtype: DataType = None,
        scale_rule: ScaleRule = None,
    ) -> None:
        self.dtype = dtype
        self.scale_rule = scale_rule

    def convert(
        self,
        input_dict: torch.Tensor,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> dict[str, list[torch.Tensor]]:
        """
        Convert the quantized parameters in gpt oss model to
        high precision weights.
        """

        prefix = ""
        if ".down_proj_blocks" in input_dict:
            weight = input_dict[".down_proj_blocks"][0]
            scales = input_dict[".down_proj_scales"][0]
            prefix = "down"
        elif ".gate_up_proj_blocks" in input_dict:
            weight = input_dict[".gate_up_proj_blocks"][0]
            scales = input_dict[".gate_up_proj_scales"][0]
            prefix = "gate_up"

        num_experts = weight.shape[0]
        hidden_size = weight.shape[1]
        weight = weight.reshape((num_experts, hidden_size, -1))

        dequantized_proj = []
        for e in range(num_experts):
            weight_uint8 = weight[e].to(torch.uint8)
            quantized_tensor = QuantizedTensor(
                values=weight_uint8,
                scale_factors=scales[e].to(torch.uint8).view(self.dtype.scale_dtype()),
                amax=torch.ones(
                    (1,),
                    device=weight[e].device,
                    dtype=torch.float32,
                ),
                dtype=self.dtype,
                original_shape=(
                    weight[e].shape[0],
                    weight[e].shape[1] * 2,
                ),
                scale_rule=self.scale_rule,
            )

            dequantized = quantized_tensor.dequantize()
            dequantized_proj.append(dequantized)

        dequantized_weight = torch.stack(dequantized_proj, dim=0)

        return {f"{prefix}_proj": [dequantized_weight]}


@WeightConversions.register(str(GptOssConfig))
class GptOssWeightConverter:
    """Stores the weight conversions for the gpt oss model."""

    @classmethod
    def get_weight_conversions(cls) -> list[WeightConverter]:
        """Return weight conversions for the gpt oss model."""
        return [
            WeightConverter(
                source_patterns=[".gate_up_proj_blocks", ".gate_up_proj_scales"],
                target_patterns=".gate_up_proj",
                operations=[
                    FourOverSixGptOssDeserialize(
                        dtype=DataType.mxfp4,
                        scale_rule=ScaleRule.static_6,
                    ),
                ],
            ),
            WeightConverter(
                source_patterns=[".down_proj_blocks", ".down_proj_scales"],
                target_patterns=".down_proj",
                operations=[
                    FourOverSixGptOssDeserialize(
                        dtype=DataType.mxfp4,
                        scale_rule=ScaleRule.static_6,
                    ),
                ],
            ),
        ]
