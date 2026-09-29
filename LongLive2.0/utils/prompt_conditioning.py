def encode_prompt_blocks(text_encoder, text_prompts, batch_size):
    """Encode per-sample prompt blocks and return batch-aligned block inputs."""
    if len(text_prompts) != batch_size:
        raise ValueError(
            f"Expected prompt blocks for {batch_size} samples, got {len(text_prompts)}."
        )

    prompt_blocks = [list(sample_prompts) for sample_prompts in text_prompts]
    if not prompt_blocks or not prompt_blocks[0]:
        raise ValueError("Each sample must provide at least one prompt block.")

    num_blocks = len(prompt_blocks[0])
    if any(len(sample_prompts) != num_blocks for sample_prompts in prompt_blocks):
        raise ValueError("All samples must provide the same number of prompt blocks.")

    flat_prompts = [prompt for sample_prompts in prompt_blocks for prompt in sample_prompts]
    conditional_dict = text_encoder(text_prompts=flat_prompts)
    prompt_embeds = conditional_dict["prompt_embeds"]
    if prompt_embeds.shape[0] != len(flat_prompts):
        raise ValueError(
            "Text encoder returned "
            f"{prompt_embeds.shape[0]} embeddings for {len(flat_prompts)} prompts."
        )

    prompt_embeds = prompt_embeds.reshape(
        batch_size, num_blocks, *prompt_embeds.shape[1:]
    )
    conditional_dict_list = [
        {"prompt_embeds": prompt_embeds[:, block_index]}
        for block_index in range(num_blocks)
    ]
    return conditional_dict, conditional_dict_list
