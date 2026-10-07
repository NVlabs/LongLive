# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: scripts/train.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

import hydra
from omegaconf import DictConfig

from longwam.utils.config_resolvers import register_default_resolvers
from longwam.training_contract import check_training_contract

register_default_resolvers()


@hydra.main(
    config_path=str(
        __import__("longwam.paths", fromlist=["repository_root"]).repository_root() / "configs"
    ),
    config_name="train",
    version_base="1.3",
)
def main(cfg: DictConfig):
    import os

    check_training_contract(cfg, world_size=int(os.environ.get("WORLD_SIZE", "1")))
    from longwam.runtime import run_training
    from longwam.paths import setup_model_paths

    setup_model_paths()
    run_training(cfg)


if __name__ == "__main__":
    main()
