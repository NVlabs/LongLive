# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: utils/__init__.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/utils/__init__.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

# Marker file: turn `utils/` from a namespace package into a regular package.
# torch 2.12's torchrun + multiprocessing has trouble resolving namespace
# packages from cwd in subprocesses; making this an explicit regular package
# makes `from utils.position_embedding_utils import ...` (used inside
# wan_5b/modules/model.py) reliable across torch versions.
