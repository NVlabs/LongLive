import warnings

try:
    from .conversions import WeightConversions
    from .gpt_oss import FourOverSixGptOssDeserialize, GptOssWeightConverter
except ImportError:
    warnings.warn("Install transformers>=5.0 to use weight conversions", stacklevel=2)

    WeightConversions = None
    FourOverSixGptOssDeserialize = None
    GptOssWeightConverter = None

__all__ = ["FourOverSixGptOssDeserialize", "GptOssWeightConverter", "WeightConversions"]
