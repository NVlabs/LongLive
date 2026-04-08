# 核心目标：为 LongLive 添加参考图控制的视频生成（Reference-to-Video）支持

## 任务背景 (Context)
当前 LongLive 模型的推理流水线（pipeline）仅支持标准的文本到视频（Text-to-Video）或无条件视频生成。我们需要修改现有的推理代码，使其能够支持引入最多3张参考图像（Reference Images）来控制视频的生成。

控制的核心逻辑是：将输入的参考图像通过模型的 VAE 编码器提取为 latent code，并在降噪（Denoising）过程开始前，将其与视频的 noise latent 进行拼接（Concatenation）。如果未提供参考图像，模型必须保持原有的标准生成行为不变。

## 具体执行任务 (Actionable Tasks)

### 1. 配置与入口修改 (Config & Entrypoint)
- [ ] **修改配置文件及解析**：在对应的 config 文件（例如 YAML 或 JSON 格式）中，新增 `reference_images` 字段，类型为列表（List of Strings），用于指定参考图像的本地路径。
- [ ] **修改 `inference.py` 和 `inference.sh`**：为推理脚本添加支持读取和传递 `reference_images` 参数的逻辑。确保它可以接受 0 到 3 个图像路径。

### 2. Pipeline 内部逻辑修改 (Pipeline Modification)
- [ ] **可选读取逻辑**：在 Pipeline 的推理函数（如 `__call__`）中，增加对 `reference_images` 的判断。如果该字段为空或未传入，继续执行原有的生成逻辑（Fallback）。
- [ ] **图像预处理与 VAE 编码**：
  - 如果提供了 `reference_images`，首先加载这些图像，并对其进行必要的预处理（如 Resize、Normalize 等），确保其分辨率与生成的视频帧分辨率对齐。
  - 将处理后的图像输入到视频模型的 VAE 中，获取对应的 reference latent codes。
- [ ] **Latent 拼接 (Concatenation)**：
  - 在初始化视频的 noise latent 之后、正式进入 UNet/DiT 降噪循环之前，将 reference latent codes 与初始的 video noise latents 进行拼接。
  - *注意分析 Tensor 维度*：请根据 LongLive 模型的网络结构要求，处理好拼接的维度。通常是在时间维度（Temporal dimension，作为前置帧）或通道维度（Channel dimension，需要网络输入层支持）进行拼接。请在代码中添加清晰的注释说明拼接策略。

## 技术约束与注意事项 (Constraints & Notes)
1. **最大数量限制**：系统最多接受 3 张参考图像。如果传入超过 3 张，需要抛出明确的 `ValueError`。
2. **显存优化**：VAE 编码参考图像时，请确保使用 `torch.no_grad()`，并注意及时释放不需要的图像 Tensor，避免 OOM 错误。
3. **向后兼容**：必须保证不传入参考图时的 Text-to-Video 任务完全不受影响，不产生额外的计算开销。
4. **设备对齐**：确保参考图像的 latent tensors 和 video noise latents 在相同的设备（device）和具有相同的数据类型（dtype，如 fp16/bf16）上。

## 验收标准 (Acceptance Criteria)
1. `inference.sh` 可以通过传参成功运行包含 1-3 张参考图的生成任务。
2. `inference.sh` 在不传入参考图时，能按原有逻辑成功生成视频，且结果与修改前一致。
3. Pipeline 内部正确调用了 VAE，完成了参考图像到 latent 的转换及拼接操作。