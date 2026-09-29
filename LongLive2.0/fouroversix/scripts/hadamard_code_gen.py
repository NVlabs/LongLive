from pathlib import Path

import numpy as np

# had_16 = """
# ++++++++++++++++
# +-+-+-+-+-+-+-+-
# ++--++--++--++--
# +--++--++--++--+
# ++++----++++----
# +-+--+-++-+--+-+
# ++----++++----++
# +--+-++-+--+-++-
# ++++++++--------
# +-+-+-+--+-+-+-+
# ++--++----++--++
# +--++--+-++--++-
# ++++--------++++
# +-+--+-+-+-++-+-
# ++----++--++++--
# +--+-++--++-+--+
# """

# transformerEngine style rht matrix
had_16 = """
+++-+------+-+--
+-++++-+-+-----+
++-++-++--+--+++
+---+++--+++--+-
+++--+++---++-++
+-++--+--+--+++-
++-+-+----+-+---
+------+-+++++-+
+++-+---+++-+-++
+-++++-++-+++++-
++-++-++++-++---
+---+++-+---++-+
+++--++++++--+--
+-++--+-+-++---+
++-+-+--++-+-+++
+------++-----+-
"""


header = """
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/csrc/code_gen.py
 * Copyright (c) 2025, FourOverSix Team.
 ******************************************************************************/

// This file is auto-generated. See "hadamard_code_gen.py"\n

#pragma once

"""

template = """
namespace fouroversix {{

__device__ __forceinline__ void hadamard_mult_thread_{N}(float x[{N}]) {
    float out[{N}];
    {code}
    #pragma unroll
    for (int i = 0; i < {N}; i++) { x[i] = out[i]; }
}

}} // namespace fouroversix

"""


def string_to_array(string: str) -> np.ndarray:
    # Convert strings of + and - to bool arrays
    string = string.strip().replace("+", "1").replace("-", "-1").split()
    return np.stack(
        [
            np.fromstring(" ".join(string[i]), dtype=np.int32, sep=" ")
            for i in range(len(string))
        ],
    )


def array_code_gen(arr: np.ndarray) -> str:
    n = arr.shape[0]
    if arr.shape[0] != arr.shape[1]:
        msg = f"Hadamard matrix is not square: {arr.shape}"
        raise ValueError(msg)
    out = [
        f"out[{i}] = "
        + " ".join([f"{'+' if arr[i, j] == 1 else '-'} x[{j}]" for j in range(n)])
        + ";"
        for i in range(n)
    ]
    return template.replace("{N}", str(n)).replace("{code}", "\n    ".join(out))


def main() -> None:
    output_dir = (
        Path(__file__).parent.parent
        / "src"
        / "fouroversix"
        / "csrc"
        / "include"
        / "hadamard_transform_te.h"
    )
    output_dir.write_text(header + array_code_gen(string_to_array(had_16)))


if __name__ == "__main__":
    main()
