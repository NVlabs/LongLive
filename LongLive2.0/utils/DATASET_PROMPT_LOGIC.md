# Dataset Prompt 分配逻辑说明

本文档说明 `MultiTextConcatDataset`（纯文本）和 `MultiVideoConcatDataset`（视频训练）在不同配置下产生 `prompts` 的行为。

---

## 架构总览

| 使用场景 | Dataset 类 | 输出 |
|---|---|---|
| diffusion.py 训练 | `MultiVideoConcatDataset` | `frames` + `prompts` |
| diffusion.py 推理 | `MultiTextConcatDataset` | `prompts` |
| distillation.py 训练（backward_sim） | `MultiTextConcatDataset` | `prompts` |
| distillation.py 可视化 | `MultiTextConcatDataset` | `prompts` |
| inference.py | `MultiTextConcatDataset` | `prompts` |

---

## 一、MultiTextConcatDataset（纯文本）

### 关键参数

| 参数 | 含义 |
|---|---|
| `num_blocks` | 输出 prompt 列表的固定长度 |
| `chunks_per_shot` | 每个 shot 重复的 block 数（0=使用 even_durations 均分） |
| `scene_cut_prefix` | 切镜标记前缀，默认 `"The scene transitions. "` |

简写约定：`0`, `1`, `2` 表示不同 caption；`p+X` 表示 `scene_cut_prefix + X`

### 1. txt 模式（data_path 指向 .txt 文件）

每行一个 caption，每个 sample 取 `idx` 行的 caption，重复 `num_blocks` 次。
**不加** scene_cut_prefix（单 shot 语义）。

```
data_path="prompts.txt", line[3]="A dog runs", num_blocks=12
→ [A, A, A, A, A, A, A, A, A, A, A, A]
```

无论 `chunks_per_shot` 设为多少，txt 模式始终单 shot 重复。

### 2. 目录模式（data_path 指向目录）

读取 `caption/<subfolder>/*.json`（不需要 `video/` 目录）。
每个 JSON 的 `caption` 字段作为一个 shot 的文本。

#### Shot duration 三级 fallback

按优先级决定每个 shot 占多少个 block：

1. **`shot_durations.txt`**（per-folder 文件）— 每行或逗号分隔的整数
2. **`chunks_per_shot`**（全局 config）— 所有 shot 统一重复固定次数
3. **`_even_durations`**（均分）— 将 `num_blocks` 均匀分给所有 caption

#### 长度处理

输出始终恰好 `num_blocks` 个 prompt：
- **超过** → 截断尾部
- **不足** → 用最后一个 caption 直接 padding（**不加** scene_cut_prefix）

#### 示例

**3 caption, num_blocks=12, chunks_per_shot=4**

```
captions: [0, 1, 2]
shot_durations: [4, 4, 4]
→ [0, 0, 0, 0, p+1, 1, 1, 1, p+2, 2, 2, 2]
```

**3 caption, num_blocks=12, chunks_per_shot=0（even_durations）**

```
captions: [0, 1, 2]
even_durations: base=4, extra=0 → [4, 4, 4]
→ [0, 0, 0, 0, p+1, 1, 1, 1, p+2, 2, 2, 2]
```

**2 caption, num_blocks=12, chunks_per_shot=4**

```
captions: [0, 1]
shot_durations: [4, 4], sum=8 < 12 → 最后一个 shot 扩展到 8
→ [0, 0, 0, 0, p+1, 1, 1, 1, 1, 1, 1, 1]
                                  ↑ padding 不加 prefix
```

**5 caption, num_blocks=12, chunks_per_shot=4**

```
captions: [0, 1, 2, 3, 4]
shot_durations: [4, 4, 4, 4, 4], 但 num_blocks=12 → clamped 到 [4, 4, 4]
→ [0, 0, 0, 0, p+1, 1, 1, 1, p+2, 2, 2, 2]
   caption 3 和 4 被截断
```

**2 caption, num_blocks=12, chunks_per_shot=0（even_durations）**

```
captions: [0, 1]
even_durations: base=6, extra=0 → [6, 6]
→ [0, 0, 0, 0, 0, 0, p+1, 1, 1, 1, 1, 1]
```

**1 caption, num_blocks=12**

```
captions: [0]
→ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   单 shot，不加 prefix
```

**shot_durations.txt 覆盖**

```
captions: [0, 1, 2], shot_durations.txt 内容: "2, 6, 4", num_blocks=12
→ [0, 0, p+1, 1, 1, 1, 1, 1, p+2, 2, 2, 2]
```

---

## 二、MultiVideoConcatDataset（视频训练）

### 关键参数

| 参数 | 含义 |
|---|---|
| `total_segments` | 总 segment 数量（= 1 + num_subsequent_segments） |
| `single_video_only` | config 中的 `uniform_prompt`，True 时只从一个视频采样 |
| `max_chunks_per_shot` | 单镜头最大连续 chunk 数，超过则跳 1 秒做虚拟切镜（0=不限制） |
| `scene_cut_prefix` | 切镜标记前缀 |
| `allow_padding` | 视频不够时是否允许 padding（否则跳过该 folder） |

Prompt 由实际视频采样决定，逐 segment 从视频文件加载对应的 per-video caption。

### 1. 多视频自然拼接（`max_chunks_per_shot=0`，默认）

按视频文件顺序采样，切换视频文件时加 `scene_cut_prefix`：

```
video A 够采 3 chunks, video B 够采 4 chunks, total_segments=7
→ [A, A, A, p+B, B, B, B]
```

### 2. `single_video_only=True`

强制只从一个视频文件采样。视频不够长则整个 folder 被跳过：

```
video A 够采 7 chunks, total_segments=7
→ [A, A, A, A, A, A, A]

video A 只够采 5 chunks, total_segments=7
→ (失败，跳到下一个 folder)
```

### 3. `max_chunks_per_shot=3`（限制单镜头时长）

从同一视频连续采超过 3 chunks 后，跳 1 秒做虚拟切镜，加 `scene_cut_prefix`：

```
video A 很长, video B, total_segments=7, max_chunks_per_shot=3
→ [A, A, A, p+A, A, A, p+B]
             ↑跳1秒虚拟切镜   ↑换视频
```

如果跳 1 秒后 A 不够了，直接跳到 B：

```
video A 够 4 chunks（跳1秒后不够）, video B, total_segments=7, max_chunks_per_shot=3
→ [A, A, A, p+B, B, B, B]
             ↑A跳1秒后不够，换B
```

### 4. `single_video_only=True` + `max_chunks_per_shot=3`

单视频内也可以做虚拟切镜：

```
video A 很长, total_segments=7, single_video_only=True, max_chunks_per_shot=3
→ [A, A, A, p+A, A, A, p+A]
```

### 5. 训练 padding（`allow_padding=True`）

视频不够时用最后一个 caption 直接 padding（不加 prefix）：

```
video A 够采 3 chunks, video B 够采 2 chunks, total_segments=7, allow_padding=True
→ [A, A, A, p+B, B, B, B]
                      ↑后 2 个用最后一个 caption padding
```

---

## 三、Multi-Shot Sink

通过 config 的 `multi_shot_sink: true` 开启。
开启后，当检测到某个 block 处于新场景的起始位置时，会在该 block 去噪完成并更新 cache 后，
将 KV cache 的 attention sink 从旧场景的第一帧迁移到新场景的第一帧。
同时会自动把全局 sink 长度设为 `sink_size`，不需要单独配置 global sink。

**配置方式**（yaml config 中添加）：

```yaml
sink_size: 8
multi_shot_sink: true    # 默认 false，不迁移 sink
```

此选项在以下所有场景均生效：

| 场景 | Pipeline | 检测方式 |
|---|---|---|
| **Diffusion trainer evaluation** | `CausalDiffusionInferencePipeline` | 检查 prompt 是否以 `scene_cut_prefix` 开头 |
| **Distillation trainer evaluation** | `CausalDiffusionInferencePipeline` | 同上 |
| **离线推理 `inference.py`** | `CausalDiffusionInferencePipeline` | 同上 |
| **Distillation backward simulation** | `SelfForcingTrainingPipeline` | trainer 预计算 `scene_cut_mask` 传入 `conditional_dict` |
| **Streaming long tuning** | `SelfForcingTrainingPipeline` | 同上（mask 随 chunk 自动 slice） |

### 实现机制

**Inference pipeline**（`CausalDiffusionInferencePipeline`）：
直接检查每个 chunk 的 raw prompt 是否以 `scene_cut_prefix` 开头。

**Training pipeline**（`SelfForcingTrainingPipeline`）：
由于 prompts 在进入 pipeline 前已编码为 embedding，无法从 embedding 反推文本。
因此 trainer 在编码前从原始 prompts 计算布尔列表 `scene_cut_mask`，
放入 `conditional_dict["scene_cut_mask"]` 一路透传到 pipeline。
对于 streaming training，`scene_cut_mask` 在 `_slice_cond_dict_for_chunk` 中随 prompt 一起 slice。

```
block:  [0]  [1]  [2]  [p+3]  [4]  [5]  [p+6]  [7]
mask:    F    F    F    T      F    F    T       F
sink:    0    0    0   →3      3    3   →6       6
                       ↑更新sink          ↑更新sink
```

当 `multi_shot_sink: false`（默认）时，sink 始终锚定在视频的第一帧，不做任何迁移。

`scene_cut_prefix` 由 `DEFAULT_SCENE_CUT_PREFIX` 常量定义（`dataset.py`），
inference pipeline 引用同一常量，确保训练和推理的切镜检测一致。
可通过 config 中的 `scene_cut_prefix` 字段统一覆盖。
