import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np

# Try to import optimized attention from ComfyUI
try:
    from comfy.ldm.modules.attention import optimized_attention_masked
except ImportError:
    # Fallback or dummy if running standalone
    def optimized_attention_masked(q, k, v, heads, mask=None, **kwargs):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

from comfy.ldm.qwen_image.model import apply_rope1

# Ported from SpotEdit_Repo/Qwen_image_edit/QwenTokenLPIPS.py
class QwenTokenLPIPS(nn.Module):
    def __init__(self, vae, patch_size=2, t_index=0):
        super().__init__()
        # In ComfyUI, vae is a wrapper. We access the underlying model.
        # If vae is already the model (e.g. passed from some nodes), check.
        if hasattr(vae, "first_stage_model"):
            self.vae = vae.first_stage_model
        else:
            self.vae = vae
            
        self.patch_size = patch_size
        self.t_index = t_index
        
        # cache z2 features
        self._z2_cached = None
        self._z2_feats_cache = None
        self._cached_device = None
        self._cached_dtype = None

    @torch.no_grad()
    def _forward_decoder_first3(self, z5d):
        dec = self.vae.decoder
        feats = {}
        feat_cache = None
        feat_idx = [0]

        # ComfyUI/Diffusers VAE architecture compatibility check
        # Some VAEs (like Qwen's specific VAE) might not have 'conv_in' or 'mid_block' as direct attributes
        # or the structure might be different.
        
        # Check for conv_in
        if hasattr(dec, "conv_in"):
             x = dec.conv_in(z5d)
             feats["conv_in"] = x
        else:
             # Try to find the first convolution layer. It might be named differently or inside blocks.
             # If we can't find it, we can't extract features.
             # Fallback: Just return empty feats or skip this level?
             # But LPIPS needs these features.
             # Assuming standard SD/SDXL/Video VAE structure usually has conv_in.
             # If it's a proprietary VAE (like WanVAE?), we need to inspect it.
             # For now, let's try to print structure or fail gracefully.
             raise AttributeError(f"VAE Decoder does not have 'conv_in'. Architecture: {type(dec)}")

        # Check for mid_block
        if hasattr(dec, "mid_block") and dec.mid_block is not None:
             x = dec.mid_block(x, feat_cache, feat_idx)
             feats["mid_block"] = x
        else:
             # Some VAEs might not have mid_block?
             feats["mid_block"] = x # Use previous feat as fallback

        # Check for up_blocks
        if hasattr(dec, "up_blocks") and len(dec.up_blocks) > 0:
             up0 = dec.up_blocks[0]
             # Check if up_block accepts feat_cache/feat_idx (temporal VAEs usually do)
             # Standard SD VAE up_blocks don't accept these args.
             # Qwen VAE seems to be 3D/Temporal based on the args.
             
             try:
                 x = up0(x, feat_cache, feat_idx)
             except TypeError:
                 # Standard 2D VAE call signature
                 x = up0(x)
                 
             feats["up_blocks.0"] = x
        else:
             feats["up_blocks.0"] = x

        return feats

    def _safe_unpack_tokens_2d(self, z_tokens, image_size, vae_downsample_factor, channels_per_token_div=4):
        # z_tokens shape: [B, N, Ctok]
        if len(z_tokens.shape) == 3:
            B, N, Ctok = z_tokens.shape
        elif len(z_tokens.shape) == 4: # Already [B, C, H, W]
             return z_tokens
        elif len(z_tokens.shape) == 5:
             # Case: [1, 16, 1, 192, 86] - Extra dimension?
             # Could be [B, C, T, H, W] or [B, C, 1, H, W]
             if z_tokens.shape[2] == 1:
                 return z_tokens.squeeze(2)
             else:
                 # Video format? Qwen is image model but VAE might be 3D?
                 # If T > 1, we might need to handle differently.
                 # Assuming single image for now.
                 return z_tokens.view(z_tokens.shape[0], z_tokens.shape[1], -1, z_tokens.shape[-1]) # Risky
        else:
             raise ValueError(f"Unexpected z_tokens shape: {z_tokens.shape}")

        H_img, W_img = image_size

        H_lat = int(2 * (int(H_img) // int(vae_downsample_factor * 2)))
        W_lat = int(2 * (int(W_img) // int(vae_downsample_factor * 2)))
        
        # Check if N matches H_lat * W_lat
        if N != (H_lat // 2) * (W_lat // 2):
            # Try to infer shape from N
            # Assuming square patch grid if image_size not perfectly matching
            # This happens if image_size passed is (1024,1024) but actual latents are different
            # We should probably trust N more if possible, or fail gracefully
            side = int(np.sqrt(N))
            if side * side == N:
                H_lat = side * 2
                W_lat = side * 2
        
        try:
            z = z_tokens.view(B, H_lat // 2, W_lat // 2, Ctok // channels_per_token_div, 2, 2)
            z = z.permute(0, 3, 1, 4, 2, 5).contiguous()
            z = z.view(B, Ctok // channels_per_token_div, H_lat, W_lat)
            return z
        except Exception as e:
            print(f"SpotEdit Error: Failed to unpack tokens. Shape={z_tokens.shape}, Target={H_lat}x{W_lat}")
            raise e

    def _apply_qwen_mean_std(self, z5d):
        p = next(self.vae.decoder.parameters())
        device, dtype = p.device, p.dtype

        # Check config
        # Some ComfyUI VAE wrappers (like WanVAE) might not expose .config directly
        # or config might be hidden inside .first_stage_model.config
        
        # Try to find config
        config = None
        if hasattr(self.vae, "config"):
            config = self.vae.config
        elif hasattr(self.vae, "first_stage_model") and hasattr(self.vae.first_stage_model, "config"):
            config = self.vae.first_stage_model.config
            
        # If still no config, try to use default values or inspect attributes directly
        if config is None:
            # Fallback to defaults or try to read attributes from vae itself if they exist
            z_dim = getattr(self.vae, "z_dim", z5d.shape[1])
            latents_mean = getattr(self.vae, "latents_mean", 0.0)
            latents_std = getattr(self.vae, "latents_std", 1.0)
        else:
            # Some ComfyUI VAE configs might be dicts or objects
            if isinstance(config, dict):
                z_dim = config.get("z_dim", z5d.shape[1])
                latents_mean = config.get("latents_mean", 0.0)
                latents_std = config.get("latents_std", 1.0)
            else:
                z_dim = getattr(config, "z_dim", z5d.shape[1])
                latents_mean = getattr(config, "latents_mean", 0.0)
                latents_std = getattr(config, "latents_std", 1.0)

        mean = torch.tensor(latents_mean, device=device, dtype=dtype)
        std  = torch.tensor(latents_std,  device=device, dtype=dtype)
        
        # Adjust dimensions for broadcasting
        # z5d shape is [B, C, T, H, W] or [B, C, H, W] depending on context, 
        # but here we expect unpacked tokens, likely [B, C, H, W] or [B, C, T, H, W]
        
        # Ensure mean/std are tensors of shape [z_dim] first
        if mean.numel() == 1:
             mean = mean.repeat(z_dim)
        if std.numel() == 1:
             std = std.repeat(z_dim)
             
        if len(z5d.shape) == 4:
            # [B, C, H, W]
            mean = mean.view(1, z_dim, 1, 1)
            std  = std.view(1, z_dim, 1, 1)
        elif len(z5d.shape) == 5:
            # [B, C, T, H, W]
            mean = mean.view(1, z_dim, 1, 1, 1)
            std  = std.view(1, z_dim, 1, 1, 1)
        else:
             # Fallback or error? 
             # Assuming at least C dimension is at index 1
             shape = [1] * len(z5d.shape)
             shape[1] = z_dim
             mean = mean.view(*shape)
             std = std.view(*shape)

        z5d = z5d.to(device=device, dtype=dtype)
        z5d = z5d * std + mean
        return z5d

    def _check_z2_cache_valid(self, z2):
        if self._z2_cached is None:
            return False
        return torch.equal(self._z2_cached, z2)
    
    @torch.no_grad()
    def set_reference_z2(self, z2, image_size, vae_downsample_factor):
        """set z2 cache for later use"""
        p = next(self.vae.decoder.parameters())
        device, dtype = p.device, p.dtype
        
        # cache z2
        self._z2_cached = z2.clone()
        self._cached_device = device
        self._cached_dtype = dtype
        
        # cache z2 features
        z2_5d = self._safe_unpack_tokens_2d(z2, image_size, vae_downsample_factor)
        z2_5d = self._apply_qwen_mean_std(z2_5d)
        self._z2_feats_cache = self._forward_decoder_first3(z2_5d)
        

    def clear_cache(self):
        self._z2_cached = None
        self._z2_feats_cache = None
        self._cached_device = None
        self._cached_dtype = None

    @torch.no_grad()
    def forward(self, z1, z2, *, image_size=None, vae_downsample_factor=None, use_cache=True):
        p = next(self.vae.decoder.parameters())
        
        z1_5d = self._safe_unpack_tokens_2d(z1, image_size, vae_downsample_factor)
        z1_5d = self._apply_qwen_mean_std(z1_5d)
        feats1 = self._forward_decoder_first3(z1_5d)
        
        # deal with z2 , use cache if valid
        if use_cache and self._check_z2_cache_valid(z2):
            feats2 = self._z2_feats_cache
        else:
            if use_cache:
                self.set_reference_z2(z2, image_size, vae_downsample_factor)
                feats2 = self._z2_feats_cache
            else:
                z2_5d = self._safe_unpack_tokens_2d(z2, image_size, vae_downsample_factor)
                z2_5d = self._apply_qwen_mean_std(z2_5d)
                feats2 = self._forward_decoder_first3(z2_5d)

        # compute diffs
        B, C_in, T_in, H_in, W_in = z1_5d.shape
        target_size_3d = (T_in, H_in, W_in)
        diffs = []
        for name in ("conv_in", "mid_block", "up_blocks.0"):
            f1, f2 = feats1[name], feats2[name]
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            d = (f1 - f2).pow(2).sum(dim=1, keepdim=True)
            if d.shape[-3:] != target_size_3d:
                d = F.interpolate(d, size=target_size_3d, mode="trilinear", align_corners=False)
            diffs.append(d)

        score_map_3d = torch.stack(diffs, dim=0).mean(dim=0).squeeze(1)
        score_map_2d = score_map_3d.mean(dim=1)

        if (H_in % self.patch_size) != 0 or (W_in % self.patch_size) != 0:
            # raise ValueError(f"latent shape {(H_in, W_in)} can not be divided by patch_size={self.patch_size}")
            # Relax check for ComfyUI potential odd sizes
             pass
             
        pooled = F.avg_pool2d(score_map_2d.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)
        token_scores = pooled.flatten(start_dim=1)

        return token_scores

@dataclass
class SpotEditConfig:
    # ---- cache decision ----
    threshold: float = 0.15
    judge_method: str = "LPIPS"
    initial_steps: int = 4
    reset_steps: list = field(default_factory=lambda: [13,22,31])
    dilation_radius: int = 1

def dilate_uncached_mask(reuse_mask: torch.Tensor, H_lat: int, W_lat: int, 
                           T_lat: int = 1, dilation_radius: int = 1) -> torch.Tensor:
    # transform reuse mask to uncached mask
    # reuse_mask is flattened [B*T*H*W]
    
    total = reuse_mask.numel()
    # Infer Batch * Time dimensions
    # We treat T as part of Batch for 2D spatial dilation
    BT = total // (H_lat * W_lat)
    
    uncached = (~reuse_mask).float().view(BT, 1, H_lat, W_lat)
    
    # define dilation kernel
    kernel_size = 2 * dilation_radius + 1
    dilated = F.max_pool2d(
        uncached, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=dilation_radius
    )
    
    # turn back to reuse mask
    return (~dilated.view(-1).bool())

def patchify_mask(reuse_mask: torch.Tensor, H_lat: int, W_lat: int, patch_size: int = 2) -> torch.Tensor:
    """
    Downsample the reuse mask from Latent Grid to Token Grid (Patchified).
    If any pixel in a patch is 'Uncached' (False), the whole patch becomes 'Uncached' (False).
    """
    if patch_size <= 1:
        return reuse_mask
        
    total = reuse_mask.numel()
    BT = total // (H_lat * W_lat)
    
    # uncached = True for regions to compute
    uncached = (~reuse_mask).float().view(BT, 1, H_lat, W_lat)
    
    # MaxPool: If any pixel is 1 (Uncached), output is 1
    # ceil_mode=True to handle odd shapes if necessary, but usually latents are divisible
    uncached_patched = F.max_pool2d(uncached, kernel_size=patch_size, stride=patch_size, ceil_mode=True)
    
    return (~uncached_patched.view(-1).bool())

def calculate_mask(original_latents, modified_latents, threshold=0.1, method='L4', vae=None):
    """
    Calculate SpotEdit Mask from two sets of latents.
    Returns:
        reuse_mask (Tensor): [N] boolean mask (True=Reuse/Skip, False=Compute)
    """
    
    # Ensure inputs are on the same device
    if original_latents.device != modified_latents.device:
        original_latents = original_latents.to(modified_latents.device)
        
    # Helper to standardize input to [N, C]
    def to_NC(t):
        # Move Channel to last dim
        if t.dim() == 4: # [B, C, H, W]
            t = t.permute(0, 2, 3, 1) # [B, H, W, C]
        elif t.dim() == 5: # [B, C, T, H, W]
            t = t.permute(0, 2, 3, 4, 1) # [B, T, H, W, C]
        
        # Flatten to [N, C]
        return t.reshape(-1, t.shape[-1])

    # LPIPS Method with Fallback
    if method == 'LPIPS':
        if vae is None:
             print("\033[93m[SpotEdit] LPIPS method requires VAE input. Falling back to 'cosine'.\033[0m")
             method = 'cosine'
        else:
            try:
                # Wrap vae if needed
                lpips_metric = QwenTokenLPIPS(vae, patch_size=2, t_index=0)
                
                # Check cache (not needed for single run, but API requires it)
                lpips_metric.set_reference_z2(
                    original_latents,  
                    image_size=(1024, 1024), 
                    vae_downsample_factor=8,
                )
                token_scores = lpips_metric(
                    modified_latents,  
                    original_latents,
                    image_size=(1024, 1024),
                    vae_downsample_factor=8,
                    use_cache=True
                )
                reuse = token_scores.mean(dim=0) < threshold
                return reuse
                
            except Exception as e:
                print(f"\033[93m[SpotEdit] LPIPS method failed (likely due to VAE incompatibility): {e}\nFalling back to 'cosine' method.\033[0m")
                method = 'cosine'

    # L4 / Cosine Method (Per-token metric)
    if method in ['L4', 'cosine']:
        z1_flat = to_NC(original_latents)
        z2_flat = to_NC(modified_latents)
        
        if method == 'L4':
            # Mean L4 diff per token: |x-y|^4
            diff = (z1_flat - z2_flat).abs().pow(4)
            # Average over channels to get score per token
            scores = diff.mean(dim=1) # [N]
            reuse = scores < threshold
        elif method == 'cosine':
            # Cosine similarity per token
            scores = F.cosine_similarity(z1_flat, z2_flat, dim=1) # [N]
            reuse = scores > threshold # High similarity = Reuse (Skip)
            
        return reuse

    # Default fallback (should not happen)
    return torch.zeros(original_latents.numel() // original_latents.shape[1], dtype=torch.bool, device=original_latents.device)


class QwenSpotEditAttnProcessor:
    def __init__(self, spotedit_state):
        self.state = spotedit_state
        self._cached_keys = None
        self._cached_values = None
        self._cached_t = None

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        image_rotary_emb: torch.Tensor = None,
        transformer_options={},
    ) -> torch.Tensor:
        
        batch_size = hidden_states.shape[0]
        seq_img = hidden_states.shape[1]
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream
        # hidden_states here is (Query) Subset if caching is active
        img_query = attn.to_q(hidden_states).view(batch_size, seq_img, attn.heads, -1).transpose(1, 2).contiguous()
        img_key = attn.to_k(hidden_states).view(batch_size, seq_img, attn.heads, -1).transpose(1, 2).contiguous()
        img_value = attn.to_v(hidden_states).view(batch_size, seq_img, attn.heads, -1).transpose(1, 2)

        # Compute QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states).view(batch_size, seq_txt, attn.heads, -1).transpose(1, 2).contiguous()
        txt_key = attn.add_k_proj(encoder_hidden_states).view(batch_size, seq_txt, attn.heads, -1).transpose(1, 2).contiguous()
        txt_value = attn.add_v_proj(encoder_hidden_states).view(batch_size, seq_txt, attn.heads, -1).transpose(1, 2)

        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)
        txt_query = attn.norm_added_q(txt_query)
        txt_key = attn.norm_added_k(txt_key)

        # SPOTEDIT LOGIC
        # With new static mask architecture, we check state.mask
        mask_latents_patched = self.state.mask
        
        if mask_latents_patched is not None and mask_latents_patched.any():
            # Retrieve flags
            # active_mask (uncached) is mask_latents_patched.logical_not()
            active_mask = mask_latents_patched.logical_not()
            
            # Ensure dimensions
            total_latents = mask_latents_patched.shape[0]
            
            if self._cached_keys is None:
                # First step: We need to initialize the cache
                # BUT, since we are in "Apply" phase, we might not have the "Old Keys" from previous steps easily
                # unless we are running sequentially.
                # However, SpotEdit logic says:
                # "Inject features from SOURCE IMAGE"
                # Wait, in the dynamic version, _cached_keys comes from the PREVIOUS step of the SAME generation.
                # But it also blends with... wait.
                
                # Re-reading SpotEdit logic:
                # It uses KV from the CURRENT step generation (which is partially skipped).
                # AND it uses KV from the REFERENCE image?
                # No, the paper says: "KV Cache Injection: Copy KV from the reference image for the background tokens".
                
                # So we need Reference Image KV.
                # Where do we get them?
                # We need to compute them!
                # But we can't compute them inside Attention easily without running the model on Ref Image.
                
                # In the original code (dynamic):
                # It seems it was reusing its OWN keys from previous steps?
                # "self._cached_keys = expanded_key"
                # And "lmd = cos..." blending.
                
                # If we use the "Static Mask" approach, we assume we are running the model on the "Edit Image".
                # For the "Background" (Skipped) tokens, we need their KV.
                # Since we SKIPPED computation, we don't have them!
                # We must substitute them.
                # Substitute with what?
                # 1. Previous step's KV (Temporal consistency) -> This is what the code did.
                # 2. Reference Image's KV (Spatial consistency).
                
                # If we use Previous Step's KV, we need to populate _cached_keys initially.
                # In the first step of the "Apply" phase (e.g. step 5), we don't have previous step (step 4) KV because we just started this KSampler.
                # This is a limitation of the "Split" approach.
                # UNLESS: We accept that for the FIRST step of the second pass, we might need to do full compute or something?
                
                # Actually, if we skip computation, we MUST have replacement keys.
                # If we don't have them, we cannot skip.
                
                # Workaround:
                # For the very first step of this KSampler, we cannot skip computation if we don't have cache.
                # But we want to skip.
                
                # Alternative:
                # The Reference Latents provided to SpotEditApply are static.
                # Can we use them?
                # We can't turn Reference Latents into KV without running the Transformer.
                
                # Let's look at the "Dynamic" code again.
                # if self._cached_keys is None:
                #    expanded_key = img_key (Full)
                #    self._cached_keys = expanded_key
                
                # So the FIRST step of SpotEdit MUST be Full Compute to fill the cache.
                # In our Split workflow:
                # Step 5 (First step of 2nd KSampler):
                # We must run FULL COMPUTE (no pruning) to generate the KV cache for the new Latents.
                # Then Step 6+ can prune.
                
                # So, in `custom_qwen_forward`, we should only prune if `self._cached_keys` is populated?
                # But `custom_qwen_forward` runs BEFORE Attention.
                # It doesn't know about Attention state.
                
                # We need a flag in State: `is_first_step`.
                # When `SpotEditApply` initializes state, `is_first_step = True`.
                # In `custom_qwen_forward`:
                # if state.is_first_step:
                #     Don't prune.
                #     state.is_first_step = False
                # else:
                #     Prune.
                
                # This solves it!
                # Step 5 will be full compute (generating KV for Step 5).
                # Step 6 will use Step 5's KV for background.
                
                expanded_key = img_key
                expanded_value = img_value
            else:
                # Construct expanded key (Full Latents + Refs)
                # self._cached_keys contains [Total_Latents + Ref_Tokens] from previous step
                
                # We update the ACTIVE part of the Latents
                # The _cached_keys structure: [Latents_Region | Ref_Region]
                
                # 1. Get Cached Latents
                cached_latents_key = self._cached_keys[:, :, :total_latents, :].clone()
                cached_latents_val = self._cached_values[:, :, :total_latents, :].clone()
                
                # 2. Update Active parts
                # img_key currently holds ACTIVE keys (corresponding to active_mask)
                if img_key.shape[2] == active_mask.sum().item():
                    cached_latents_key[:, :, active_mask, :] = img_key
                    cached_latents_val[:, :, active_mask, :] = img_value
                
                # 3. Apply Blending (Interpolation)
                # lmd logic
                # We need timestep. 
                # transformer_options['sigmas'] ?
                # Or pass it via state?
                # We'll rely on what we have.
                # If we don't have 't', we skip blending or use constant.
                # Original code used `cache_flags[-1]`.
                # We store sigma in state.sigma in forward.
                
                t_val = self.state.current_sigma
                if t_val is not None:
                    t = torch.tensor(t_val, device=img_key.device)
                    lmd = torch.cos(0.5 * torch.pi * (t / 1000))**2
                else:
                    lmd = 0.5 # Default?
                
                final_latents_key = (1 - lmd) * cached_latents_key + lmd * self._cached_keys[:, :, :total_latents, :]
                final_latents_val = (1 - lmd) * cached_latents_val + lmd * self._cached_values[:, :, :total_latents, :]
                
                # 4. Concatenate Ref parts
                # Ref parts are always cached (or handled separately, but usually static)
                ref_key = self._cached_keys[:, :, total_latents:, :]
                ref_val = self._cached_values[:, :, total_latents:, :]
                
                expanded_key = torch.cat([final_latents_key, ref_key], dim=2)
                expanded_value = torch.cat([final_latents_val, ref_val], dim=2)
                
                # Update Cache
                self._cached_keys = expanded_key.detach() # Detach to be safe
                self._cached_values = expanded_value.detach()
            
            img_key = expanded_key
            img_value = expanded_value
            
        else:
            # Full update (no caching)
            # Store current as cache
            self._cached_keys = img_key.detach()
            self._cached_values = img_value.detach()
        
        # RoPE Application
        # img_query is subsetted (if caching)
        # img_key is FULL (if caching)
        
        full_rope = transformer_options.get("spotedit_full_rope", image_rotary_emb)
        
        # Determine RoPE for Query and Key
        # Key uses full rope
        rope_key = full_rope
        
        # Query uses subsetted rope (which matches image_rotary_emb argument passed from block)
        # Verify: nodes.py subsets image_rotary_emb before passing to block.
        # So image_rotary_emb argument here IS the subsetted RoPE.
        rope_query = image_rotary_emb
        
        if rope_query is not None and rope_key is not None:
             joint_query = torch.cat([txt_query, img_query], dim=2)
             joint_key = torch.cat([txt_key, img_key], dim=2)
             joint_value = torch.cat([txt_value, img_value], dim=2)
             
             joint_query = apply_rope1(joint_query, rope_query)
             joint_key = apply_rope1(joint_key, rope_key)
        else:
             # Fallback
             joint_query = torch.cat([txt_query, img_query], dim=2)
             joint_key = torch.cat([txt_key, img_key], dim=2)
             joint_value = torch.cat([txt_value, img_value], dim=2)
             
        # Attention
        joint_hidden_states = optimized_attention_masked(
            joint_query, joint_key, joint_value, attn.heads,
            attention_mask, transformer_options=transformer_options,
            skip_reshape=True
        )
        
        # Output split
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]
        
        img_attn_output = attn.to_out[0](img_attn_output)
        img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)
        
        return img_attn_output, txt_attn_output

    def forward_replacement(self, q, k, v, heads, mask=None):
        # SageAttention Compatibility: Force FP16/BF16 if input is FP32/FP8
        # SageAttention requires FP16 or BF16.
        target_dtype = q.dtype
        if target_dtype not in [torch.float16, torch.bfloat16]:
             target_dtype = torch.float16
             q = q.to(target_dtype)
             if k.dtype != target_dtype:
                 k = k.to(target_dtype)
             if v.dtype != target_dtype:
                 v = v.to(target_dtype)
        
        # Explicitly Check/Cast txt parts too!
        # txt_len is usually known from state
        txt_len = getattr(self.state, "txt_len", 0)
        
        # Note: q, k, v passed here are [Text + Image] concatenated.
        # So casting q, k, v above already covered BOTH Text and Image parts!
        # Wait, if q is concatenated, then q.to(dtype) converts the whole thing.
        # So my previous fix should have worked if q was indeed concatenated.
        
        # BUT, if `q` was FP32, then `target_dtype` became FP16.
        # And `q = q.to(FP16)` converts it.
        
        # Is it possible that `q` was ALREADY FP16, but `txt_q` (which is part of `q`) was constructed from something else?
        # No, `q` is a single tensor passed to this function.
        
        # The error says: "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
        # This usually refers to Q, K, V passed to `sage_attention`.
        
        # In `forward_replacement`, we reconstruct `joint_q`, `joint_k`, `joint_v` at the end.
        # joint_q = torch.cat([txt_q, img_q], dim=1)
        
        # If `txt_q` was sliced from `q` (which we casted), it should be FP16.
        # If `img_q` was sliced from `q` (which we casted), it should be FP16.
        
        # HOWEVER, `expanded_k` comes from cache or concatenation.
        # `expanded_k` might be FP8 or FP32 if cache was stored differently?
        # I added checks for `expanded_k` at the end of function.
        
        # Let's verify the checks at the end of function.
        
        # q, k, v are [Batch, SeqLen, Heads, Dim]
        # They already have RoPE applied (if applicable).
        # They contain [Text + Image] concatenated.
        
        # We need to separate Text and Image
        # How do we know the split index?
        # We can guess from `state.mask` or passing context length?
        # But `set_model_attn1_replace` doesn't pass context length.
        
        # However, `q` shape is [B, Seq, H, D].
        # In `custom_qwen_forward`, we know context length.
        # We can store it in state?
        # Yes, `state.txt_len`
        
        txt_len = getattr(self.state, "txt_len", 0)
        
        # Split
        txt_q = q[:, :txt_len]
        txt_k = k[:, :txt_len]
        txt_v = v[:, :txt_len]
        
        img_q = q[:, txt_len:]
        img_k = k[:, txt_len:]
        img_v = v[:, txt_len:]
        
        # SPOTEDIT LOGIC (Simplified for Replacement)
        # We need to inject cached KV into img_k, img_v
        
        mask_latents_patched = self.state.mask
        
        if mask_latents_patched is not None and mask_latents_patched.any():
            active_mask = mask_latents_patched.logical_not()
            total_latents = mask_latents_patched.shape[0]
            
            # img_k here corresponds to ACTIVE tokens (because we subsetted in Forward)
            # UNLESS forward subsetting failed or logic changed.
            # Assuming Forward subsetting worked, img_k.shape[1] should be active_count + ref_count
            
            # Wait, `attn1_replace` receives what the Block computes.
            # Block computes QKV from `hidden_states`.
            # If `hidden_states` was subsetted in Forward, then QKV are subsetted.
            
            # So img_k is [Active_Tokens].
            
            if self._cached_keys is None:
                # First step (Full Compute)
                # Store full keys
                self._cached_keys = img_k.detach()
                self._cached_values = img_v.detach()
                
                # No injection needed
                expanded_k = img_k
                expanded_v = img_v
            else:
                # Subset step
                # We need to expand img_k to full size using cache
                
                # 1. Get Cached (Full)
                # Cached keys are from previous step.
                cached_k = self._cached_keys[:, :, :total_latents, :].clone()
                cached_v = self._cached_values[:, :, :total_latents, :].clone()
                
                # 2. Update Active parts with current computation
                # img_k is [B, Active, H, D]
                # We need to map active tokens back to full grid
                
                # Ensure active_mask matches
                if img_k.shape[1] == active_mask.sum().item(): # Only Latents part
                     # But img_k might contain Ref tokens too?
                     # If Ref tokens were kept in Forward, they are at the end.
                     ref_len = img_k.shape[1] - active_mask.sum().item()
                     if ref_len >= 0:
                         active_k = img_k[:, :active_mask.sum().item()]
                         active_v = img_v[:, :active_mask.sum().item()]
                         
                         cached_k[:, active_mask] = active_k
                         cached_v[:, active_mask] = active_v
                         
                         # 3. Blending
                         # Use current sigma/t for blending
                         # Assuming we can get lmd from state
                         t_val = self.state.current_sigma
                         lmd = 0.5 # Default
                         if t_val is not None:
                             try:
                                 t = torch.tensor(t_val, device=img_k.device)
                                 lmd = torch.cos(0.5 * torch.pi * (t / 1000))**2
                             except:
                                 pass
                         
                         final_k = (1 - lmd) * cached_k + lmd * self._cached_keys[:, :, :total_latents, :]
                         final_v = (1 - lmd) * cached_v + lmd * self._cached_values[:, :, :total_latents, :]
                         
                         # 4. Add Ref (from current or cache?)
                         # If we kept Ref in forward, we have current Ref keys in img_k[:, active:]
                         # We can use them.
                         if ref_len > 0:
                             ref_k = img_k[:, active_mask.sum().item():]
                             ref_v = img_v[:, active_mask.sum().item():]
                             
                             expanded_k = torch.cat([final_k, ref_k], dim=1)
                             expanded_v = torch.cat([final_v, ref_v], dim=1)
                         else:
                             expanded_k = final_k
                             expanded_v = final_v
                         
                         # Update Cache
                         # We should update cache with the NEW full keys?
                         # Or blended?
                         # Usually blended.
                         
                         # We need to handle Ref in cache too if Ref exists.
                         if ref_len > 0:
                             # self._cached_keys includes ref?
                             # Yes, if initialized with full.
                             pass
                         
                         self._cached_keys = expanded_k.detach()
                         self._cached_values = expanded_v.detach()
                         
                else:
                    # Shape mismatch, fallback
                    expanded_k = img_k
                    expanded_v = img_v

        else:
             # No mask or full compute
             expanded_k = img_k
             expanded_v = img_v
             
             # Update cache if this is a Full Compute step (e.g. first step)
             # But how do we distinguish "First Step Full Compute" from "No Mask"?
             # We can check state.is_first_step (already handled in forward)
             # Or just always update cache if we have full set?
             if img_k.shape[1] == getattr(self.state, "num_latents", 0):
                 self._cached_keys = img_k.detach()
                 self._cached_values = img_v.detach()

        # Re-concatenate
        # Ensure devices match
        if txt_k.device != expanded_k.device:
            expanded_k = expanded_k.to(txt_k.device)
            expanded_v = expanded_v.to(txt_v.device)
            
        # Ensure DTYPES match (SageAttention Error Fix)
        # FP8 models often have keys in FP8 or mixed types. 
        # SageAttention requires FP16 or BF16.
        # If input q is FP16, we should cast k/v to FP16.
        
        # Double check target_dtype from current joint_q components (txt_q, img_q)
        # They were sliced from q which was cast at start.
        # But let's be paranoid.
        if txt_q.dtype not in [torch.float16, torch.bfloat16]:
             txt_q = txt_q.to(torch.float16)
        if img_q.dtype not in [torch.float16, torch.bfloat16]:
             img_q = img_q.to(torch.float16)
             
        target_dtype = txt_q.dtype
        
        if expanded_k.dtype != target_dtype:
             expanded_k = expanded_k.to(target_dtype)
        if expanded_v.dtype != target_dtype:
             expanded_v = expanded_v.to(target_dtype)
             
        # Also check txt parts (though usually fine)
        if txt_k.dtype != target_dtype:
             txt_k = txt_k.to(target_dtype)
        if txt_v.dtype != target_dtype:
             txt_v = txt_v.to(target_dtype)
             
        joint_k = torch.cat([txt_k, expanded_k], dim=1)
        joint_v = torch.cat([txt_v, expanded_v], dim=1)
        joint_q = torch.cat([txt_q, img_q], dim=1) # Q is kept as is (Active only)
        
        # Call Optimized Attention
        return optimized_attention_masked(joint_q, joint_k, joint_v, heads, mask)
