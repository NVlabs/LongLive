# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/utils/misc.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import collections
import collections.abc
import functools
import json
import os
import random
import time
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, TypeVar
from urllib.parse import urlparse

import boto3
import numpy as np
import termcolor
import torch
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor.api import DTensor

_WORK_DIR: str | None = None
_DEFAULT_WORK_DIR = "./runs/"


def register_work_dir(path: str | os.PathLike | None) -> None:
    global _WORK_DIR
    _WORK_DIR = str(path) if path is not None else None
    os.makedirs(path, exist_ok=True)


def get_work_dir() -> str | None:
    return _WORK_DIR if _WORK_DIR is not None else _DEFAULT_WORK_DIR
