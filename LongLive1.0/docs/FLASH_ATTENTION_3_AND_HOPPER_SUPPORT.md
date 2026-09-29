# Flash Attention 3 and Hopper GPU Support

This document describes the Flash Attention 3 (FA3) integration and extended Hopper GPU support in LongLive.

## Overview

LongLive supports both Flash Attention 2 (FA2) and Flash Attention 3 (FA3) for efficient attention computation. FA3 is automatically enabled on Hopper architecture GPUs (Compute Capability 9.0+), providing improved performance.

## Supported Hardware

### Hopper Architecture GPUs (FA3 Enabled)
- **NVIDIA H100** - Data center GPU
- **NVIDIA H800** - China-specific variant
- **NVIDIA H20** - China-specific variant

All Hopper GPUs share Compute Capability 9.0, which is the requirement for FA3.

### Other GPUs (FA2 Fallback)
- **NVIDIA A100** - Ampere architecture (Compute Capability 8.0)
- **NVIDIA A800** - Ampere architecture (Compute Capability 8.0)
- Other CUDA-capable GPUs with FA2 support

## Design Choices

### 1. GPU Detection via Compute Capability

Instead of relying on device name string matching (which would miss H800/H20), we detect Hopper GPUs using CUDA Compute Capability:

```python
def is_hopper_gpu():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return major >= 9  # Hopper Compute Capability == 9.0
    return False
```

**Rationale:**
- Device names vary across vendors and regions (H100, H800, H20, etc.)
- Compute Capability is a reliable, standardized way to identify GPU architecture
- All Hopper GPUs report `major=9` regardless of their marketing name

### 2. FA3 Return Value Handling

Flash Attention 3's `flash_attn_varlen_func` has a different return signature than FA2:

| Version | Return Value |
|---------|--------------|
| FA2 | `(output, softmax_lse, ...)` - tuple, use `[0]` to get output |
| FA3 | `output` - tensor directly |

The code correctly handles this difference:

```python
# FA3 path - direct tensor return
x = flash_attn_interface.flash_attn_varlen_func(...).unflatten(0, (b, lq))

# FA2 path - tuple return (handled in else branch)
x = flash_attn.flash_attn_varlen_func(...).unflatten(0, (b, lq))
```

### 3. Automatic Fallback

The system gracefully falls back to FA2 when FA3 is unavailable:
- If `flash_attn_interface` module is not installed
- If running on non-Hopper GPU
- If user explicitly requests FA2 via `version=2` parameter

A warning is issued when FA3 is explicitly requested but unavailable.

## Usage

### Automatic Selection (Recommended)

By default, LongLive automatically selects the optimal attention implementation:

```python
from wan.modules.attention import attention

# FA3 will be used on Hopper GPUs, FA2 otherwise
output = attention(q, k, v)
```

### Explicit Version Selection

You can force a specific Flash Attention version:

```python
# Force FA2 (useful for debugging or compatibility)
output = attention(q, k, v, fa_version=2)

# Request FA3 (falls back to FA2 with warning if unavailable)
output = attention(q, k, v, fa_version=3)
```

## Installation

### Flash Attention 3 (Hopper GPUs)

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```
