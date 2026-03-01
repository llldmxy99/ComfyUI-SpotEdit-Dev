# ComfyUI-SpotEdit 使用说明

**ComfyUI-SpotEdit** 是 **SpotEdit** 的 ComfyUI 自定义节点实现。SpotEdit 是一种**无需训练**的、具有**区域感知能力**的图像编辑框架，基于扩散 Transformer (DiTs) 模型。

本插件针对 **Qwen Image Edit** 模型进行了深度适配，通过 **Model Patcher** 技术实现了原论文中的核心加速与一致性保持机制。

---

## 🌟 核心原理

SpotEdit 不仅仅是一个简单的 Prompt 修改工具，它通过以下两个核心机制实现了“只编辑该编辑的地方，且速度更快”：

1.  **动态计算 (Dynamic Computation / Token Pruning)**
    *   传统的扩散模型即使只修图的一小部分，也会计算整张图的所有 Token。
    *   本插件劫持了 Qwen 模型的 `forward` 过程，利用 **Spotselect** 算法动态识别出哪些 Token 是“背景”，哪些是“编辑区”。
    *   在 Transformer 层计算前，**直接剔除（Prune）背景 Token**，只计算编辑区的 Token。这不仅保证了背景绝对不发生变化，还随着编辑区域的减小显著**提升了推理速度并降低了显存占用**。

2.  **KV Cache 注入 (KV Cache Injection)**
    *   对于被剔除的背景区域，模型在 Self-Attention 层会**直接注入源图像的 Key/Value 特征**。
    *   这意味着背景区域的像素特征直接取自原图，而非重新生成，从而完美解决了重绘时的像素漂移和一致性问题。

---

## 🛠️ 安装指南

1.  **进入插件目录**
    打开终端，进入你的 ComfyUI `custom_nodes` 目录：
    ```bash
    cd ComfyUI/custom_nodes
    ```

2.  **克隆仓库**
    ```bash
    git clone https://github.com/your-username/ComfyUI-SpotEdit.git
    ```

3.  **安装依赖**
    进入插件目录并安装所需的 Python 依赖库：
    ```bash
    cd ComfyUI-SpotEdit
    pip install -r requirements.txt
    ```
    *主要依赖为 `einops` 用于张量重排。*

4.  **重启 ComfyUI**

---

## 🧩 节点介绍与新版工作流

为了提供更强的兼容性和可视化能力，本插件采用了**两阶段采样**的工作流设计。

### 1. SpotEdit Mask Generator (Mask 生成器)
**节点路径**: `SpotEdit` -> `SpotEdit Mask Generator`

该节点负责对比“原图 Latent”和“预采样 Latent”，计算出需要编辑的区域 Mask。

*   **输入**:
    *   `original_latents`: 原始图像的 Latent (VAE Encode)。
    *   `modified_latents`: **预采样** 4步后的 Latent (来自第一个 KSampler)。
    *   `vae` (可选): 仅当使用 LPIPS 方法时需要连接。
*   **参数**:
    *   `threshold`: 编辑阈值。越大编辑区域越广。
    *   `judge_method`: 推荐 LPIPS (需连接VAE) 或 L4/Cosine。
*   **输出**:
    *   `SPOTEDIT_MASK`: 计算好的 Mask 数据。
    *   `IMAGE`: **可视化 Mask 预览图**。白色代表编辑区域，黑色代表背景（保持不变）。你可以通过 Preview Image 节点实时查看并调整阈值！

### 2. SpotEdit Apply (Static) (应用器)
**节点路径**: `SpotEdit` -> `SpotEdit Apply (Static)`

该节点将计算好的 Mask 注入模型，开启加速采样。

*   **输入**:
    *   `model`: Qwen 模型。
    *   `reference_latents`: 原始图像 Latent (用于背景 KV 注入)。
    *   `spotedit_mask`: 上一步生成的 Mask。
*   **输出**:
    *   `MODEL`: 注入了 SpotEdit 逻辑的模型。连接到第二个 KSampler。

---

## 🚀 推荐工作流 (Two-Pass)

1.  **加载模型**: 加载 Checkpoint, CLIP, VAE。
2.  **准备 Latent**: 加载原图 -> VAE Encode -> 得到 `Original Latent`。
3.  **第一阶段采样 (Pre-Run)**:
    *   使用标准 `KSampler`。
    *   **steps**: 20 (总步数)。
    *   **start_at_step**: 0。
    *   **end_at_step**: 4 (只跑前4步)。
    *   **return_with_leftover_noise**: **Enable** (关键！保留噪声以便后续采样)。
    *   输入: `Original Latent`。
    *   输出: `Coarse Latent`。
4.  **生成 Mask**:
    *   连接 `SpotEdit Mask Generator`。
    *   输入: `Original Latent` 和 `Coarse Latent`。
    *   **调节**: 连接一个 `Preview Image` 到 `mask_image` 输出端口，调整 `threshold` 直到白色区域准确覆盖你想编辑的物体。
5.  **应用 SpotEdit**:
    *   连接 `SpotEdit Apply (Static)`。
    *   输入: 原始 Model, `Original Latent`, 和生成的 `mask`。
6.  **第二阶段采样 (Final Run)**:
    *   使用另一个 `KSampler`。
    *   **model**: 连接来自 `SpotEdit Apply` 的模型。
    *   **latent_image**: 连接来自第一阶段的 `Coarse Latent`。
    *   **steps**: 20。
    *   **start_at_step**: 4 (接续采样)。
    *   **end_at_step**: 20。
    *   **return_with_leftover_noise**: Disable。
    *   **denoise**: 1.0 (虽然是接续采样，但在 ComfyUI 逻辑中通常设为 1.0，因为 start_step 已经控制了进度)。

---

## 💻 技术实现细节 (对于开发者)

*   **静态 Mask 注入**: 相比于旧版的动态计算，新版将 Mask 计算移到了采样循环外部。这解决了 Tensor 形状不匹配的问题，并允许用户干预 Mask。
*   **Token Pruning**: 在第二阶段采样的第1步（即总第5步），插件会强制进行一次全量计算以构建 KV Cache，随后的步骤将根据 Mask 进行 Token Pruning（剪枝），大幅减少计算量。
*   **KV Cache Injection**: 背景区域的 Key/Value 直接复用自 Reference Latent（实际上是利用第一步全量计算构建的 Cache），确保背景像素级一致。

## ⚠️ 注意事项与限制

1.  **模型兼容性**:
    *   ✅ **FP8**: 兼容。
    *   ❌ **Nunchaku**: 不兼容。
2.  **VAE**:
    *   LPIPS 模式依然依赖 VAE 结构。如果报错，请使用 Cosine 模式，或调整 `threshold`。
3.  **Mask 形状**: 插件内部会自动处理 Latent 到 Token (Patch Size=2) 的映射，用户无需关心。

## 引用

> SpotEdit: Selective Region Editing in Diffusion Transformers (arXiv:2512.22323)
