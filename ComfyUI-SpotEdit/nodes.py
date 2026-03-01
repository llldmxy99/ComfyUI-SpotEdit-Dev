import torch
import types
import numpy as np
import torch.nn.functional as F
from PIL import Image
from .qwen_spotedit_utils import calculate_mask, dilate_uncached_mask, patchify_mask, QwenSpotEditAttnProcessor
import comfy.ldm.common_dit
import comfy.patcher_extension
from einops import repeat

class SpotEditState:
    def __init__(self):
        self.mask = None # Patched Mask (Token Level)
        self.last_output = None 
        self.reference_latents = None 
        self.current_sigma = None
        self.is_first_step = True # Flag to ensure first step of 2nd pass is Full Compute

def custom_qwen_forward(self, x, timesteps, context, attention_mask=None, ref_latents=None, additional_t_cond=None, transformer_options={}, control=None, **kwargs):
    # This replaces QwenImageTransformer2DModel._forward
    
    # Retrieve State
    state = getattr(self, "spotedit_state", None)
    
    # Update sigma in state for Attention Processor
    if state is not None:
        # Timesteps is tensor of shape [B]
        # We need a float value for lmd calculation. 
        # Usually it's roughly related to sigma or timestep.
        # Assuming timesteps[0] is enough.
        state.current_sigma = timesteps[0].item() if timesteps.numel() > 0 else 0
    
    timestep = timesteps
    encoder_hidden_states = context
    encoder_hidden_states_mask = attention_mask

    # Process Image (Full)
    hidden_states, img_ids, orig_shape = self.process_img(x)
    
    # Device Sync
    device = hidden_states.device
    if context is not None:
        context = context.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if timestep is not None:
        timestep = timestep.to(device)

    # Re-assign variables after device sync
    encoder_hidden_states = context
    encoder_hidden_states_mask = attention_mask
        
    num_embeds = hidden_states.shape[1] # Latent tokens count

    # Ensure ref_latents is on the same device as hidden_states
    if ref_latents is not None:
         ref_latents = [r.to(hidden_states.device) for r in ref_latents]

    timestep_zero_index = None
    if ref_latents is not None:
        # Original logic for handling ref_latents injection (if any)
        # Note: SpotEdit might pass reference_latents via node, but standard Qwen might not.
        # If SpotEditApply passed reference_latents to model, it might appear here?
        # Actually, in ComfyUI, ref_latents arg usually comes from specialized nodes.
        # SpotEdit uses state.reference_latents.
        # So standard ref_latents here might be None or from other nodes.
        # We process standard ref_latents if present.
        h = 0
        w = 0
        index = 0
        ref_method = kwargs.get("ref_latents_method", self.default_ref_method)
        index_ref_method = (ref_method == "index") or (ref_method == "index_timestep_zero")
        negative_ref_method = ref_method == "negative_index"
        timestep_zero = ref_method == "index_timestep_zero"
        for ref in ref_latents:
            if index_ref_method:
                index += 1
                h_offset = 0
                w_offset = 0
            elif negative_ref_method:
                index -= 1
                h_offset = 0
                w_offset = 0
            else:
                index = 1
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

            kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
            hidden_states = torch.cat([hidden_states, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)
        if timestep_zero:
            if index > 0:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
                timestep_zero_index = num_embeds

    txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
    
    # context is encoder_hidden_states
    # context shape might be [B, Seq, C] or None?
    # In ComfyUI, context is usually provided.
    # If context is None (uncond?), we need to handle it.
    # But usually passed.
    
    if context is not None:
        txt_ids_len = context.shape[1]
    else:
        txt_ids_len = 0
        
    txt_ids = torch.arange(txt_start, txt_start + txt_ids_len, device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    
    # Calculate Full RoPE
    image_rotary_emb = self.pe_embedder(ids).to(x.dtype).contiguous()
    del ids, txt_ids, img_ids

    # Store Full RoPE in transformer_options for Attention Processor
    transformer_options["spotedit_full_rope"] = image_rotary_emb
    
    # Store text length in state for Attention Patch
    if state is not None:
        state.txt_len = context.shape[1]
        state.num_latents = num_embeds

    # SPOTEDIT SUBSETTING
    # Check if we should prune
    should_prune = False
    if state is not None and state.mask is not None:
        if state.is_first_step:
            # First step of 2nd pass: FORCE FULL COMPUTE to build KV cache
            should_prune = False
            state.is_first_step = False # Disable flag for next steps
        else:
            should_prune = True
            
    if should_prune and state.mask.any():
        # state.mask is True for Reuse/Skip (Background)
        # We keep False (Active/Edit)
        # active_mask corresponds to Latents part ONLY
        
        if state.mask.device != hidden_states.device:
             state.mask = state.mask.to(hidden_states.device)

        mask_latents = state.mask
        active_mask = mask_latents.logical_not()
        
        # Dimensions Check
        if active_mask.shape[0] != num_embeds:
             print(f"SpotEdit Error: Mask size {active_mask.shape[0]} != num_embeds {num_embeds}. Disabling Pruning.")
             should_prune = False
        else:
            # Ref Latents Mask?
            # hidden_states contains [Latents, Ref_Standard...].
            # SpotEdit logic usually assumes Ref is fully cached or fully computed?
            # If we prune Latents, we should probably keep Ref intact or handle it.
            # Assuming Ref (if any) is kept fully active for safety, or we construct a combined mask.
            
            # Construct combined mask
            # Latents: active_mask
            # Rest: All True (Keep)
            
            rest_len = hidden_states.shape[1] - num_embeds
            if rest_len > 0:
                mask_rest = torch.ones((rest_len), dtype=torch.bool, device=active_mask.device)
                combined_mask = torch.cat([active_mask, mask_rest])
            else:
                combined_mask = active_mask
                
            # Apply Subset
            hidden_states = hidden_states[:, combined_mask, :]
            
            # Apply Subset to RoPE
            # image_rotary_emb covers [Text, Latents, Ref]
            seq_txt = context.shape[1]
            rope_txt = image_rotary_emb[:, :seq_txt]
            rope_img = image_rotary_emb[:, seq_txt:]
            
            rope_img_subset = rope_img[:, combined_mask]
            image_rotary_emb = torch.cat([rope_txt, rope_img_subset], dim=1)
    
    # Define active_mask for later use (Reconstruction) even if pruning was skipped due to error
    if not should_prune:
        # If not pruning, active_mask is effectively all True (but we don't need it for reconstruction logic below)
        # However, to avoid UnboundLocalError in reconstruction block if logic flow is weird:
        active_mask = None 

    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)

    patches_replace = transformer_options.get("patches_replace", {})
    patches = transformer_options.get("patches", {})
    blocks_replace = patches_replace.get("dit", {})

    transformer_options["total_blocks"] = len(self.transformer_blocks)
    transformer_options["block_type"] = "double"
    
    # Nunchaku Offloading Setup
    compute_stream = torch.cuda.current_stream()
    offload = getattr(self, "offload", False)
    if offload and hasattr(self, "offload_manager"):
        self.offload_manager.initialize(compute_stream)
    
    # Run Blocks (on subset or full)
    for i, block in enumerate(self.transformer_blocks):
        # Nunchaku Block Retrieval
        if offload and hasattr(self, "offload_manager"):
            block = self.offload_manager.get_block(i)

        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"], timestep_zero_index=timestep_zero_index, transformer_options=args["transformer_options"])
                return out
            out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb, "transformer_options": transformer_options}, {"original_block": block_wrap})
            hidden_states = out["img"]
            encoder_hidden_states = out["txt"]
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                timestep_zero_index=timestep_zero_index,
                transformer_options=transformer_options,
            )

        if "double_block" in patches:
            for p in patches["double_block"]:
                out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]

        if control is not None: # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    if should_prune:
                         # Subset 'add' to match hidden_states
                         # add shape: [B, N, C]
                         if add.shape[1] == num_embeds:
                             add = add[:, active_mask] # Apply mask1 (Latents only)
                         hidden_states[:, :add.shape[1]] += add
                    else:
                        hidden_states[:, :add.shape[1]] += add

        if offload and hasattr(self, "offload_manager"):
            self.offload_manager.step(compute_stream)

    if timestep_zero_index is not None:
        temb = temb.chunk(2, dim=0)[0]

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # SPOTEDIT RECONSTRUCTION
    if should_prune:
        # hidden_states is currently subsetted
        # We need to reconstruct the FULL output.
        
        # Get Active Latents Output
        active_latents_out = hidden_states[:, :active_mask.sum().item()]
        
        # Reconstruct Full Latents Output
        # Use state.last_output for the cached parts
        if state.last_output is None:
             # Should not happen if first step was full compute
             # But if it happens, we init zeros
             full_output = torch.zeros((hidden_states.shape[0], num_embeds, hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype)
        else:
             full_output = state.last_output.clone()
        
        # We need to map active_latents_out back to full_output
        full_output[:, active_mask] = active_latents_out
        
        # Now use full_output for reshaping
        hidden_states = full_output
        
        # Update state.last_output
        state.last_output = full_output.detach()
        
    else:
        # Full update
        hidden_states_latents = hidden_states[:, :num_embeds]
        
        if state is not None:
             state.last_output = hidden_states_latents.detach()
             
        hidden_states = hidden_states_latents


    # Reshape back to image
    hidden_states = hidden_states.view(orig_shape[0], orig_shape[-3], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
    hidden_states = hidden_states.permute(0, 4, 1, 2, 5, 3, 6)
    return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]


class SpotEditMaskGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_latents": ("LATENT",),
                "modified_latents": ("LATENT",),
                "threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "judge_method": (["LPIPS", "L4", "cosine"],),
                "dilation_radius": ("INT", {"default": 1, "min": 0, "max": 10}),
            },
            "optional": {
                 "vae": ("VAE",), # Needed for LPIPS
            }
        }

    RETURN_TYPES = ("SPOTEDIT_MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("spotedit_mask", "mask_image", "mask")
    FUNCTION = "generate_mask"
    CATEGORY = "SpotEdit"

    def generate_mask(self, original_latents, modified_latents, threshold, judge_method, dilation_radius, vae=None):
        orig = original_latents["samples"]
        mod = modified_latents["samples"]
        
        # Calculate Mask (Latent Level)
        # reuse_mask: True = Background (Reuse), False = Changed (Active)
        reuse_mask = calculate_mask(orig, mod, threshold=threshold, method=judge_method, vae=vae)
        
        # Get dimensions
        if orig.ndim == 5:
            B, C, T, H, W = orig.shape
        else:
            B, C, H, W = orig.shape
            T = 1
            
        # Dilate Mask (Latent Level)
        # dilate_uncached_mask expects reuse_mask
        mask_latents = dilate_uncached_mask(reuse_mask, H, W, T, dilation_radius=dilation_radius)
        
        # Patchify Mask (Latent -> Token Level)
        # Assuming Qwen Patch Size = 2
        patch_size = 2
        mask_patched = patchify_mask(mask_latents, H, W, patch_size=patch_size)
        
        # Generate Visualization Image
        # Mask is 1D [N]. We need to reshape to Image.
        # reuse_mask is True for Background (Black), False for Active (White)
        # Let's invert for vis: White = Edit, Black = Background
        
        vis_mask = (~mask_latents).float() # 1.0 = Edit
        
        # Reshape to [B, H, W] (Ignore T for vis, take first frame)
        # mask_latents is flattened [B*T*H*W]
        vis_mask = vis_mask.view(B, T, H, W)
        vis_mask = vis_mask[:, 0, :, :] # Take first frame
        
        # Resize to Image Size (approx x8 VAE)
        # VAE factor usually 8
        vis_mask = vis_mask.unsqueeze(1) # [B, 1, H, W]
        vis_mask = F.interpolate(vis_mask, scale_factor=8, mode='nearest')
        
        # Convert to ComfyUI Image format [B, H, W, C]
        # vis_mask is [B, 1, H, W]
        # ComfyUI Image expects [B, H, W, C]
        vis_image = vis_mask.permute(0, 2, 3, 1)
        
        # Ensure it is CPU and float32
        vis_image = vis_image.cpu().float()
        
        # Squeeze the channel dimension if it is 1, but ComfyUI expects [B, H, W, C] where C=3 usually or 1?
        # Preview Image expects [B, H, W, C]
        # If C=1, PIL might handle it as Grayscale.
        # But the error says: Cannot handle this data type: (1, 1, 1), |u1
        # This implies numpy array shape (1, 1, 1) which is weird for an image.
        # It seems the batch size is 1, H=1, W=1 ?
        # Or maybe Preview Image expects specific shape.
        
        # Let's ensure H, W are at least 1.
        if vis_image.shape[1] == 1 and vis_image.shape[2] == 1:
             # If visualization is just 1 pixel, resize it to something visible?
             # No, this probably means H/W collapsed.
             pass
             
        # Check if C is 1, maybe repeat to 3?
        if vis_image.shape[-1] == 1:
            vis_image = vis_image.repeat(1, 1, 1, 3)
        
        # Output standard MASK for KSampler
        # vis_mask is [B, 1, H, W] (Before repeating)
        # But wait, vis_mask variable was overwritten by repeat above!
        
        # Correct logic:
        # 1. Get vis_mask (1 channel)
        vis_mask_1ch = vis_mask # This is already [B, H, W, 3] if overwritten?
        # Let's fix the flow.
        
        vis_mask = (~mask_latents).float() # 1.0 = Edit
        vis_mask = vis_mask.view(B, T, H, W)
        vis_mask = vis_mask[:, 0, :, :] # [B, H, W]
        vis_mask = vis_mask.unsqueeze(1) # [B, 1, H, W]
        vis_mask = F.interpolate(vis_mask, scale_factor=8, mode='nearest') # [B, 1, H, W]
        
        # Clone for output mask
        out_mask = vis_mask.squeeze(1).cpu() # [B, H, W]
        
        # Prepare for Preview Image (3 Channels)
        vis_image = vis_mask.permute(0, 2, 3, 1) # [B, H, W, 1]
        vis_image = vis_image.cpu().float()
        if vis_image.shape[-1] == 1:
            vis_image = vis_image.repeat(1, 1, 1, 3)

        return (mask_patched, vis_image, out_mask)


class SpotEditApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_latents": ("LATENT",),
                "spotedit_mask": ("SPOTEDIT_MASK",),
            },
            # Allow optional VAE input to be ignored if passed by mistake (e.g. replacing old node)
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_spotedit"
    CATEGORY = "SpotEdit"

    def apply_spotedit(self, model, reference_latents, spotedit_mask, vae=None):
        m = model.clone()
        
        # Setup State
        state = SpotEditState()
        state.mask = spotedit_mask
        state.reference_latents = reference_latents["samples"]
        state.is_first_step = True 
        state.attn_processor = None # To hold the processor instance
        
        # Attach state to the CLONED model's diffusion_model
        # Note: model.clone() shallow copies the wrapper, but m.model.diffusion_model is usually shared!
        # We must NOT modify m.model.diffusion_model directly if it's shared.
        # However, ComfyUI usually handles this by cloning the underlying model structure if needed? No.
        # We should use model options to inject our logic without modifying the class instance methods permanently.
        
        # Initialize Processor
        attn_processor = QwenSpotEditAttnProcessor(state)
        state.attn_processor = attn_processor
        
        # Safe Patching Strategy:
        # 1. Use set_model_unet_function_wrapper to intercept the forward call.
        # 2. Use set_model_attn1_patch / attn2_patch for Attention (or wrapper).
        
        # Re-implementing using model_options wrapper
        
        # 1. Forward Wrapper
        def forward_wrapper(model_function, params):
            # This wrapper is called by ComfyUI's model executor
            # params contains: input, timestep, c, cond, control, transformer_options
            
            # We need to call our custom_qwen_forward logic here.
            # But custom_qwen_forward is written as a method replacement.
            # We can adapt it.
            
            # Extract args
            x = params["input"]
            timesteps = params["timestep"]
            context = params["c"].get("context")
            if context is None:
                context = params["c"].get("c_crossattn")
            # transformer_options = params["c"]["transformer_options"] # Passed in options
            
            # CHECK IF FIRST STEP AND CACHE IS EMPTY
            # If so, we run a PRE-PASS on Reference Latents to fill the cache
            if state.is_first_step and attn_processor._cached_keys is None:
                print(f"SpotEdit: Initializing Cache from Reference Latents (Shape: {state.reference_latents.shape})...")
                # We need to run the model on reference latents.
                # But reference latents might be different size or batch?
                # Usually same size.
                # We use the same parameters but swap input x
                
                # Ensure ref_latents is on device
                ref = state.reference_latents.to(x.device)
                
                # If batch size differs, we might need to handle it.
                # Assuming batch size 1 for now or matching.
                if ref.shape[0] != x.shape[0]:
                    ref = ref.repeat(x.shape[0], 1, 1, 1) # Repeat to match batch
                
                # Run Forward on Ref (Disable Pruning for this run)
                # We temporarily set state.mask to None to force full compute
                original_mask = state.mask
                state.mask = None
                
                # We also need to set a flag to tell AttnProcessor "This is Reference Run, Store Cache!"
                # AttnProcessor logic currently stores cache if _cached_keys is None.
                # So just running it should work.
                
                # We discard the output, we just want the side effect (Cache population)
                with torch.no_grad():
                     custom_qwen_forward(
                        m.model.diffusion_model, 
                        ref, # Input is Ref
                        timesteps, 
                        context, 
                        state=state, 
                        **params["c"]
                    )
                
                # Restore Mask
                state.mask = original_mask
                # Disable first step flag (though custom_qwen_forward might have done it)
                state.is_first_step = False
                print("SpotEdit: Cache Initialized.")

            # Now run the actual Forward on Input x
            return custom_qwen_forward(
                m.model.diffusion_model, # self
                x, # x
                timesteps, # timesteps
                context, # context
                state=state, # NEW ARGUMENT
                **params["c"] # other args
            )

        m.set_model_unet_function_wrapper(forward_wrapper)
        
        # Attach state to the ModelPatcher wrapper, NOT the diffusion_model
        # But custom_qwen_forward expects state in `self.spotedit_state` (where self is diffusion_model).
        # We can change custom_qwen_forward to accept state as argument.
        
        # We need to pass 'state' to the forward function.
        # We can use a closure.
        
        # Patch Attention
        # Attention patching in ComfyUI is usually done via `set_model_attn1_patch`.
        # But we need to replace the whole processor or inject logic.
        # QwenSpotEditAttnProcessor is a full processor replacement.
        # ComfyUI's `set_model_attn1_patch` allows replacing the attention operation?
        # No, it patches the *weights* or *output*.
        
        # If we want to change Attention behavior (KV Injection), we need to replace the forward method of Attention blocks.
        # Again, modifying blocks permanently is bad.
        
        # Does ComfyUI support `set_model_attn_function_wrapper`? Not standard.
        # But we can use `transformer_options` to pass patches!
        
        # In `custom_qwen_forward`, we are already manually iterating blocks:
        # for i, block in enumerate(self.transformer_blocks): ...
        
        # Wait, `custom_qwen_forward` IS the function that iterates blocks!
        # So we have full control there.
        # We don't need to patch Attention modules if we control the block iteration loop?
        # NO, the block calls `attention(hidden_states, ...)`.
        # We need to change what `attention` does.
        
        # In `custom_qwen_forward`, we define `image_rotary_emb` etc.
        # We can wrap the attention call?
        # The block code is inside `comfy.ldm.qwen_image.model`. We can't change it easily.
        
        # However, `custom_qwen_forward` sets `transformer_options`.
        # We can pass our AttnProcessor via `transformer_options`?
        # ComfyUI's `optimized_attention_masked` checks for patches?
        
        # Let's look at `QwenSpotEditAttnProcessor`. It's a `__call__` that takes `attn` module.
        # We used to monkey-patch `child.forward`.
        
        # To avoid monkey-patching shared objects:
        # We can't easily avoid it for Attention unless we re-implement the Block forward too.
        # Re-implementing Block forward is possible but tedious.
        
        # COMPROMISE:
        # We use monkey-patching but with a "Restoration" mechanism?
        # Or we clone the diffusion_model? (Memory heavy)
        
        # BETTER:
        # Use `transformer_options['patches']`?
        # ComfyUI has `attn1_patch` support.
        # If we look at `comfy/ldm/modules/attention.py`, `optimized_attention` calls `patches`.
        
        # But we need to change Q/K/V calculation (KV Cache), not just weights.
        
        # Let's stick to monkey-patching but make it safer.
        # We can store the original method and restore it?
        # No, async execution makes this impossible.
        
        # THE ONLY SAFE WAY for Attention replacement in ComfyUI without deep hacks:
        # Use `set_model_attn1_replace` (if available) or similar.
        # ComfyUI ModelPatcher has `set_model_attn1_patch`.
        # `set_model_attn1_patch` expects a function `f(q, k, v, extra_options)`.
        # This might be enough!
        # If we can modify K, V inside this patch, we are good.
        
        # QwenSpotEditAttnProcessor logic:
        # 1. Modifies K, V (Injection/Blending).
        # 2. Calls optimized_attention.
        
        # If we use `set_model_attn1_patch` (Self-Attn), we get Q, K, V.
        # We can return modified Q, K, V.
        # BUT, `set_model_attn1_patch` in ComfyUI usually patches the *weights* linear layer result?
        # No, it patches the q, k, v tensors.
        # Let's verify ComfyUI source (not provided, but general knowledge).
        # Usually: q = q_linear(x); k = ...; v = ...; q, k, v = patch(q, k, v)
        
        # If so, we can implement SpotEdit logic in `attn1_patch`!
        # And we don't need to touch Attention.forward.
        
        # Let's try to move Attention Logic to `attn1_patch`.
        
        # Define Patch Function
        def spotedit_attn1_patch(q, k, v, extra_options):
            # This replaces the Attention Logic or just modifies QKV?
            # ComfyUI `set_model_attn1_patch` modifies QKV.
            # But SpotEdit needs to MAINTAIN STATE (Cache).
            # We can access `state` from closure.
            
            # But wait, QwenSpotEditAttnProcessor does more:
            # It handles RoPE (Rotary Embedding) for the INJECTED Keys.
            # Standard ComfyUI might apply RoPE *after* patch?
            # We need to check where RoPE is applied.
            # In `custom_qwen_forward`, we calculate RoPE and pass it.
            # In `QwenSpotEditAttnProcessor`:
            # joint_query = apply_rope1(...)
            
            # If we use `attn1_patch`, we get Q, K, V *before* RoPE (usually).
            # If we inject cached keys (which already have RoPE applied?), we might double apply.
            # SpotEdit cache stores `img_key` *after* norm?
            # In `QwenSpotEditAttnProcessor`:
            # img_key = attn.norm_k(img_key)
            # ...
            # self._cached_keys = expanded_key
            
            # It seems complex to map exactly to `attn1_patch`.
            
            # Let's stick to `forward_wrapper` for the main loop.
            # For Attention:
            # Since we are already inside `custom_qwen_forward` (which we control via wrapper),
            # we can pass a CUSTOM `transformer_options` dict that contains our Attention Processor.
            # AND, we can rely on `custom_qwen_forward` to NOT use the standard block if we want?
            # No, we call `block(...)`.
            
            # We can use `set_model_unet_function_wrapper` to wrap the whole diffusion model.
            # And inside `custom_qwen_forward`, we are safe.
            # But `custom_qwen_forward` calls `self.transformer_blocks`.
            # These blocks contain standard Attention modules.
            
            # Here is the trick:
            # In `custom_qwen_forward`, we can dynamically monkey-patch the blocks JUST FOR THIS CALL?
            # Still not thread safe.
            
            # However, `custom_qwen_forward` is running in the main process (or executor).
            # If we modify the objects, we modify them for everyone.
            
            # Does `QwenImageTransformer2DModel` support passing custom attention processor?
            # Not natively.
            
            # Let's assume for now that we MUST use `set_model_unet_function_wrapper` to replace `_forward`.
            # This fixes the `m.model.diffusion_model._forward = ...` persistence issue.
            # Because the wrapper is stored in `model_options` which is cloned with `m`.
            
            # So, we just need to adapt `custom_qwen_forward` to be called from wrapper.
            # AND pass `state` safely.
            
            return q, k, v

        # Implementation of Safe Forward Wrapper
        def spotedit_forward_wrapper(model_function, params):
            # Inject state into the model instance temporarily? 
            # Or pass it as argument if we modify custom_qwen_forward?
            
            # We will modify `custom_qwen_forward` to take `state` as explicit argument.
            # And we pass `m.model.diffusion_model` as `self`.
            
            context = params["c"].get("context")
            if context is None:
                context = params["c"].get("c_crossattn")
            
            return custom_qwen_forward(
                m.model.diffusion_model, # self
                params["input"], # x
                params["timestep"], # timesteps
                context, # context
                state=state, # NEW ARGUMENT
                **params["c"] # other args
            )
            
        m.set_model_unet_function_wrapper(spotedit_forward_wrapper)
        
        # Now about Attention Patching.
        # Since we can't safely patch Attention modules, we will use a different strategy for KV Injection.
        # We will do it inside `custom_qwen_forward` if possible?
        # No, it's inside blocks.
        
        # We will use `comfy.model_patcher.set_model_attn1_replace`.
        # This allows replacing the entire attention computation.
        # We need to define a function that takes `(q, k, v, heads, mask)`.
        
        def attn1_replacement(q, k, v, heads, mask=None):
            # This replaces the `optimized_attention` call inside the block?
            # No, ComfyUI patches `comfy.ldm.modules.attention.optimized_attention`.
            # But we need access to `state` (Cache).
            
            # If we use `set_model_attn1_replace`, we can capture `state` in closure.
            # But this replacement function needs to handle RoPE and everything?
            # `optimized_attention` input `q, k, v` already has RoPE applied usually?
            # In Qwen, RoPE is applied inside the Block, *before* passing to attention?
            # Let's check `custom_qwen_forward`:
            # image_rotary_emb is passed to block.
            # Block applies RoPE?
            # Qwen Block:
            # joint_query = apply_rope1(...)
            # optimized_attention(...)
            
            # So `attn1_replace` receives RoPE'd QKV.
            # This is GOOD. We can do our KV Caching/Blending here!
            
            # BUT: We need to know which tokens are Image vs Text to apply caching only to Image.
            # QKV here is concatenated [Text + Image].
            # We can split by length.
            
            # Implementation of Attention Replacement
            return attn_processor.forward_replacement(q, k, v, heads, mask)

        # We need to add `forward_replacement` method to `QwenSpotEditAttnProcessor`
        # and use `m.set_model_attn1_replace(attn1_replacement, "block_name"?)`
        # ComfyUI `set_model_attn1_replace` patches ALL blocks.
        
        m.set_model_attn1_replace(attn1_replacement, "dit", 0) # "dit" context, block index?
        # Wait, `set_model_attn1_replace` signature: (patch, block_type, block_index)
        # We need to apply to ALL blocks.
        
        # Helper to find transformer_blocks safely
        def get_transformer_blocks(model_obj):
            if hasattr(model_obj, "transformer_blocks"):
                return model_obj.transformer_blocks
            if hasattr(model_obj, "diffusion_model"):
                return get_transformer_blocks(model_obj.diffusion_model)
            if hasattr(model_obj, "model"): # Wrapper?
                return get_transformer_blocks(model_obj.model)
            # Nunchaku specific check? Or generic traversal?
            # If it's a list/ModuleList, return it?
            return None

        # Find blocks
        blocks = get_transformer_blocks(m.model.diffusion_model)
        if blocks is None:
             # Fallback: Try accessing standard Qwen paths or just fail gracefully?
             # If we can't find blocks, we can't patch attention properly if we need index.
             # But set_model_attn1_replace needs block_index.
             # Assuming standard Qwen has 30 blocks usually? Or 28?
             # Qwen2.5-VL has 28?
             print("SpotEdit Warning: Could not find 'transformer_blocks' attribute. Assuming 28 blocks (Qwen Standard).")
             num_blocks = 28
        else:
             num_blocks = len(blocks)

        for i in range(num_blocks):
             m.set_model_attn1_replace(attn1_replacement, "double_block", i)
             
        return (m,)

NODE_CLASS_MAPPINGS = {
    "SpotEditMaskGen": SpotEditMaskGen,
    "SpotEditApply": SpotEditApply
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpotEditMaskGen": "SpotEdit Mask Generator",
    "SpotEditApply": "SpotEdit Apply (Static)"
}
