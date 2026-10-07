# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

"""Compose and check a training recipe on CPU without importing Torch."""

import argparse
import json
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from longwam.paths import repository_root
from longwam.training_contract import check_training_contract
from longwam.utils.config_resolvers import register_default_resolvers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, required=True)
    args, overrides = parser.parse_known_args()
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(repository_root() / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    print(json.dumps(check_training_contract(cfg, world_size=args.world_size), indent=2))
    print(OmegaConf.to_yaml(cfg, resolve=False))


if __name__ == "__main__":
    main()
