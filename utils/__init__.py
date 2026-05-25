# Marker file: turn `utils/` from a namespace package into a regular package.
# torch 2.12's torchrun + multiprocessing has trouble resolving namespace
# packages from cwd in subprocesses; making this an explicit regular package
# makes `from utils.position_embedding_utils import ...` (used inside
# wan_5b/modules/model.py) reliable across torch versions.
