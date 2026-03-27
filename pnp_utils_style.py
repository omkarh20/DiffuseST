import torch
import os
import random
import numpy as np

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_time(model, t):
    # conv_module = model.unet.up_blocks[1].resnets[1]
    # setattr(conv_module, 't', t)
    res_dict = {0: [0,1,2], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            conv_module = model.unet.up_blocks[res].resnets[block]
            setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents

def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 2)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size] 
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size] 
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size] 
                k[2 * source_batch_size:] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.to_v(encoder_hidden_states)
                v = self.head_to_batch_dim(v)


            else :
                q = self.to_q(x)
                k = self.to_k(x)
                v = self.to_v(x)
                w = 0.8
                source_batch_size = int(q.shape[0] // 3)
                #第一部分content第二部分无条件的第三部分有条件的
                # inject unconditional
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                v[source_batch_size:2 * source_batch_size] = v[:source_batch_size]
                
                
                # inject conditional
                k[2 * source_batch_size:] = k[:source_batch_size]
                v[2 * source_batch_size:] = v[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.head_to_batch_dim(v)

            
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    # conv_module = model.unet.up_blocks[1].resnets[1]
    # conv_module.forward = conv_forward(conv_module)
    # setattr(conv_module, 'injection_schedule', injection_schedule)
    
    res_dict = {1: [0, 1, 2], 2: [0, 1, 2]}
    for res in res_dict:
        for block in res_dict[res]:
            conv_module = model.unet.up_blocks[res].resnets[block]
    
            conv_module.forward = conv_forward(conv_module)
  
            setattr(conv_module, 'injection_schedule', injection_schedule)


# =====================================================================
# PHASE 1: MASK AND BETA SCALING UTILITIES (Training-Free Enhancement)
# =====================================================================

def resize_mask_to_attention_shape(mask, target_height, target_width, device='cuda'):
    """
    Resize binary mask to match U-Net attention layer spatial dimensions.
    
    Args:
        mask: torch.Tensor, shape (H, W) or (1, 1, H, W), binary [0, 1]
        target_height: int, target spatial height
        target_width: int, target spatial width
        device: str, device to place resized mask on
    
    Returns:
        resized_mask: torch.Tensor, shape (1, 1, target_height, target_width), binary [0, 1]
    """
    # Ensure mask is 4D: (1, 1, H, W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    
    mask = mask.float().to(device)
    
    # Bilinear interpolation for smooth resizing
    if mask.shape[-2:] != (target_height, target_width):
        mask = torch.nn.functional.interpolate(
            mask, 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
    
    # Threshold back to binary [0, 1]
    mask = (mask > 0.5).float()
    
    return mask


def load_and_prepare_mask(mask_path, reference_image_h, reference_image_w, device='cuda'):
    """
    Load binary mask from file and prepare for attention control.
    
    Args:
        mask_path: str, path to mask file (PNG or NPY)
        reference_image_h: int, height of reference content image
        reference_image_w: int, width of reference content image
        device: str, device to place mask on
    
    Returns:
        mask: torch.Tensor, shape (1, 1, H, W), binary [0, 1]
    
    Raises:
        FileNotFoundError: if mask_path does not exist
        ValueError: if mask format is unsupported
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load mask based on file extension
    if mask_path.endswith('.png') or mask_path.endswith('.jpg'):
        from PIL import Image
        mask_pil = Image.open(mask_path).convert('L')  # Grayscale
        mask = torch.from_numpy(np.array(mask_pil, dtype=np.float32)) / 255.0
    elif mask_path.endswith('.npy'):
        mask = torch.from_numpy(np.load(mask_path).astype(np.float32))
    else:
        raise ValueError(f"Unsupported mask format: {mask_path}. Use .png, .jpg, or .npy")
    
    # Validate and resize to reference image size
    if mask.shape[0] != reference_image_h or mask.shape[1] != reference_image_w:
        print(f"[MASK] Auto-resizing mask from {mask.shape} to ({reference_image_h}, {reference_image_w})")
        mask = resize_mask_to_attention_shape(mask, reference_image_h, reference_image_w, device=device)
    else:
        mask = mask.unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Log mask statistics
    percent_masked = (mask == 1.0).float().mean().item() * 100
    print(f"[MASK] Loaded mask: {mask.shape}, {percent_masked:.1f}% pixels masked")
    
    return mask


def compute_attention_bias_from_mask(mask, attention_shape, device='cuda'):
    """
    Convert binary mask to attention bias matrix.
    
    Args:
        mask: torch.Tensor, binary mask, shape (1, 1, H, W)
        attention_shape: tuple, (sequence_length, sequence_length) for attention logits
        device: str, device to place bias on
    
    Returns:
        attention_bias: torch.Tensor, shape (1, 1, seq_len, seq_len), values in {0, -inf}
                       0 = attend to masked region, -inf = block attention to non-masked region
    """
    seq_len = attention_shape[0]
    
    # Flatten mask spatially: (1, 1, H, W) -> (H*W,)
    mask_flat = mask.reshape(-1)  # Shape: (H*W,)
    
    # Create attention bias: -inf for non-masked, 0 for masked
    # Expand to attention matrix shape: (seq_len, seq_len)
    attention_bias = torch.zeros(seq_len, seq_len, device=device)
    
    # For each position, set -inf for non-masked positions
    # This ensures softmax near-zero attention to style outside mask
    non_masked_indices = (mask_flat == 0).nonzero(as_tuple=True)[0]
    
    if len(non_masked_indices) > 0:
        # Set columns (key positions) that are non-masked to -inf
        attention_bias[:, non_masked_indices] = float('-inf')
    
    return attention_bias.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)


def register_attention_control_with_mask_and_scaling(model, injection_schedule, mask=None, beta=1.0, target_blocks=None):
    """
    Register enhanced attention control with mask and beta scaling support.
    
    Args:
        model: UNet model with attention layers
        injection_schedule: list/set of timesteps for style injection
        mask: torch.Tensor, binary spatial mask (1, 1, H, W), optional. If None, no spatial control.
        beta: float, style strength multiplier [0.0 = no style, 1.0 = full style], default 1.0
        target_blocks: list of block names to apply mask+beta, e.g., ['up_blocks.1', 'up_blocks.2'].
                      If None, applies to all blocks (backward compatible).
    """
    if target_blocks is None:
        target_blocks = None  # Apply to all blocks (backward compatible)
    
    def sa_forward_with_mask_and_scale(module, target_blocks_set):
        """
        Enhanced self-attention forward with mask injection and beta scaling.
        
        Simplified: 
        - During injection window: scale style K/V by beta
        - If mask present: apply attention bias to restrict style to masked regions
        """
        to_out = module.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = to_out[0]
        else:
            to_out = to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = module.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            
            if not is_cross and module.injection_schedule is not None and (
                    module.t in module.injection_schedule or module.t == 1000):
                # ===== INJECTION WINDOW (STYLE PHASE) =====
                q = module.to_q(x)
                k = module.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 2)
                
                # (Batch structure: [content, style])
                # Extract and scale style K/V by beta
                style_k = k[source_batch_size:2 * source_batch_size].clone()
                if module.beta is not None and module.beta < 1.0:
                    style_k = style_k * module.beta
                
                # Inject unconditional from content
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size] 
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size] 
                
                # Inject conditional from scaled style
                q[2 * source_batch_size:] = q[:source_batch_size] 
                k[2 * source_batch_size:] = style_k

                q = module.head_to_batch_dim(q)
                k = module.head_to_batch_dim(k)
                v = module.to_v(encoder_hidden_states)
                v = module.head_to_batch_dim(v)

            else:
                # ===== NON-INJECTION WINDOW (CONTENT PHASE) =====
                q = module.to_q(x)
                k = module.to_k(x)
                v = module.to_v(x)
                
                source_batch_size = int(q.shape[0] // 3)
                
                # Inject unconditional and conditional from content
                # (Batch structure: [content, unconditional, conditional])
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                v[source_batch_size:2 * source_batch_size] = v[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]
                v[2 * source_batch_size:] = v[:source_batch_size]

                q = module.head_to_batch_dim(q)
                k = module.head_to_batch_dim(k)
                v = module.head_to_batch_dim(v)

            # ===== ATTENTION COMPUTATION =====
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * module.scale

            # Apply spatial mask bias if mask is available and in injection window
            if module.mask is not None and module.t in module.injection_schedule:
                # Lazily compute and cache attention bias
                current_seq_len = sim.shape[-1]
                if not hasattr(module, '_attention_bias_cache') or module._attention_bias_cache is None or module._attention_bias_cache.shape[-1] != current_seq_len:
                    # Map sequence length to spatial dimensions (assume square spatial layout)
                    h_attn = int(np.sqrt(sequence_length))
                    w_attn = sequence_length // h_attn if h_attn > 0 else int(np.sqrt(sequence_length))
                    
                    # Resize mask to attention layer's spatial resolution
                    mask_resized = resize_mask_to_attention_shape(module.mask, h_attn, w_attn, device=sim.device)
                    
                    # Compute attention bias from spatial mask
                    attention_bias = compute_attention_bias_from_mask(mask_resized, (current_seq_len, current_seq_len), device=sim.device)
                    module._attention_bias_cache = attention_bias
                else:
                    attention_bias = module._attention_bias_cache
                
                # Apply attention bias to sim matrix
                # Shape: sim = (batch*heads, seq_len, seq_len), attention_bias = (1, 1, seq_len, seq_len)
                sim = sim + attention_bias.squeeze(0)  # Remove batch/head dimensions for broadcasting

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # Attention softmax
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = module.batch_to_head_dim(out)

            return to_out(out)

        return forward

    # Register enhanced attention forward
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    target_blocks_set = set(target_blocks) if target_blocks is not None else None
    
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward_with_mask_and_scale(module, target_blocks_set)
            setattr(module, 'injection_schedule', injection_schedule)
            setattr(module, 'mask', mask)
            setattr(module, 'beta', beta)
            # Cache for attention bias computation
            setattr(module, '_attention_bias_cache', None)