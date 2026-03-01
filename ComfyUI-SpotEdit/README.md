# ComfyUI-SpotEdit

<div align="center">
  <br>
  <em>
    SpotEdit in ComfyUI - Selective region editing by reusing source image KV cache
  </em>
</div>

## 📚 Overview

**ComfyUI-SpotEdit** is a ComfyUI custom node implementation of **SpotEdit**, a **training-free, region-aware framework** for instruction-based image editing with **Diffusion Transformers (DiTs)**.

While most image editing tasks only modify small local regions, existing diffusion-based editors regenerate the entire image at every denoising step, leading to redundant computation and potential degradation in preserved areas. SpotEdit follows a simple principle: **edit only what needs to be edited**.

SpotEdit dynamically identifies **non-edited regions** during the diffusion process and skips unnecessary computation for these regions, while maintaining contextual coherence for edited regions through adaptive feature fusion.

## 🛠️ Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ComfyUI-SpotEdit.git
   ```

3. Install the required dependencies:
   ```bash
   cd ComfyUI-SpotEdit
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## 📦 Requirements

- torch>=1.13.0
- torchvision
- numpy
- einops

## 🧩 Node: SpotEdit Apply (Qwen)

Currently supports **Qwen Image Edit** models.

### Inputs
- **model** (MODEL): The Qwen diffusion model to patch.
- **vae** (VAE): The VAE model used for LPIPS calculation (if selected) and feature extraction.
- **reference_latents** (LATENT): The original image encoded as latents. This is used as the reference for determining which regions to edit.
- **threshold** (FLOAT): Similarity threshold for region reuse (default: 0.15). Lower values mean stricter reuse (more editing), higher values mean more reuse (less editing).
- **initial_steps** (INT): Number of initial steps to run full inference before enabling caching (default: 4).
- **judge_method** (STRING): Method for computing similarity - "LPIPS" (recommended), "L4", or "cosine".
- **dilation_radius** (INT): Radius for mask dilation to smooth boundaries (default: 1).

### Outputs
- **MODEL**: The patched model. Connect this to your KSampler.

### Usage in Workflow

1. **Load Qwen Model**: Use standard loaders.
2. **Load Original Image**: Load the image you want to edit.
3. **Encode Original Image**: Use VAE Encode to convert original image to `LATENT`.
4. **Connect SpotEditApply**:
   - Connect Qwen Model to `model`.
   - Connect VAE to `vae`.
   - Connect Encoded Original Latents to `reference_latents`.
5. **KSampler**:
   - Connect the output `MODEL` from `SpotEditApply` to KSampler.
   - Set KSampler to start from noise (std Qwen workflow) or as appropriate for your editing task.
   - SpotEdit will automatically handle the selective editing during sampling.

## 🧠 How It Works

SpotEdit implements the following key mechanisms:

1. **Cache Decision**: During denoising, computes similarity between current prediction and source image latents.
2. **Attention Injection**: Injects cached key-value pairs into attention computation.
3. **Selective Processing**: Only denoises uncached tokens while reusing source image latents for cached tokens.

The implementation follows the original SpotEdit paper's approach but is adapted for ComfyUI's architecture.

## ⚠️ Limitations

1. **Model Support**: Currently implemented for **Qwen Image Edit**. Flux support is planned.
2. **Global Edits**: SpotEdit is not intended for global edits that affect most or all regions of the image.
3. **Resolution**: Best results are typically achieved at resolutions of 1024×1024.

## 📄 Citation

If you use this implementation in your research, please cite the original SpotEdit paper:

```bibtex
@article{qin2025spotedit,
  title={SpotEdit: Selective Region Editing in Diffusion Transformers},
  author={Qin, Zhibin and Tan, Zhenxiong and Wang, Zeqing and Liu, Songhua and Wang, Xinchao},
  journal={arXiv preprint arXiv:2512.22323},
  year={2025}
}
```
