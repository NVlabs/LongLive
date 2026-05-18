from typing import Any

import torch
from inspect_ai.model import modelapi
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._providers.hf import HuggingFaceAPI
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_random_seeds(seed: int | None = None) -> None:
    import os

    import numpy as np
    from transformers import set_seed

    if seed is None:
        seed = np.random.default_rng().integers(2**32 - 1)
    # python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # transformers seed
    set_seed(seed)


class LocalHuggingFaceAPI(HuggingFaceAPI):
    """
    Wrapper around HuggingFaceAPI that allows for quantized models to be used during
    evaluation.
    """

    def __init__(  # noqa: C901
        self,
        model_name: str,
        model: AutoModelForCausalLM,
        config: GenerateConfig | None = None,
        **model_args: dict[str, Any],
    ) -> None:
        self.model_name = model_name
        self.base_url = None
        self.api_key = None
        self.api_key_vars = ["HF_TOKEN"]
        self._apply_api_key_overrides()

        if config is None:
            config = GenerateConfig()

        # set random seeds
        if config.seed is not None:
            set_random_seeds(config.seed)

        # collect known model_args (then delete them so we can pass the rest on)
        def collect_model_arg(name: str) -> Any | None:  # noqa: ANN401
            nonlocal model_args
            value = model_args.get(name)
            if value is not None:
                model_args.pop(name)
            return value

        device = collect_model_arg("device")
        tokenizer = collect_model_arg("tokenizer")
        model_path = collect_model_arg("model_path")
        tokenizer_path = collect_model_arg("tokenizer_path")
        self.batch_size = collect_model_arg("batch_size")
        self.chat_template = collect_model_arg("chat_template")
        self.tokenizer_call_args = collect_model_arg("tokenizer_call_args")
        self.enable_thinking = collect_model_arg("enable_thinking")
        if self.tokenizer_call_args is None:
            self.tokenizer_call_args = {}
        self.hidden_states = collect_model_arg("hidden_states")

        # device
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # model
        self.model = model

        # tokenizer
        if tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)  # type: ignore[no-untyped-call]
        elif model_path:
            if tokenizer_path:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # type: ignore[no-untyped-call]
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        # LLMs generally don't have a pad token and we need one for batching
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"


@modelapi(name="local_hf")
def local_hf() -> type[LocalHuggingFaceAPI]:
    return LocalHuggingFaceAPI
