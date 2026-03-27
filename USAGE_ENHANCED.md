# DiffuseST Enhanced: Usage Guide

This document covers the new **Style Strength Control (Beta)** and **Region-Based Masking** features added to DiffuseST.

## Quick Start

### Default Usage (Backward Compatible)
```bash
# Standard DiffuseST (no changes to original behavior)
python run.py
```

### With Style Strength Control
```bash
# Reduce style intensity to 70% of full strength
python run.py --beta 0.7

# Subtle style application (30% strength)
python run.py --beta 0.3

# No style (0%)
python run.py --beta 0.0

# Full style (default, 100%)
python run.py --beta 1.0
```

### With Region-Based Masking
```bash
# Apply style only to masked region (binary PNG mask)
python run.py --mask_path masks/sky_mask.png

# Combine mask with reduced intensity
python run.py --mask_path masks/sky_mask.png --beta 0.8
```

---

## Features

### 1. Style Strength Control (`--beta`)

**What it does:** Continuously control the magnitude of style influence during synthesis, independent of the injection window timing (`--alpha`).

**Parameters:**
- `--beta` : float in range [0.0, 1.0]
  - `0.0` : No style applied (output ≈ content-only image)
  - `0.5` : 50% style blending
  - `1.0` : Full style (default, original DiffuseST behavior)

**How it works:**
- During the style injection window, style features (K/V matrices in attention) are multiplied by beta
- This reduces the magnitude/strength of style influence while preserving spatial localization
- Non-linear relationship: beta=0.5 does NOT produce 50% identical result to beta=1.0 (content structure affects interaction)

**Use cases:**
- Fine-tune style intensity after finding good content-style pair for image quality
- Subtle style hints without over-stylization
- A/B testing styles at different intensities

**Examples:**
```bash
# Gentle artsy style (for photorealistic content)
python run.py --alpha 0.1 --beta 0.4

# Strong style (for abstract transformations)
python run.py --alpha 0.2 --beta 1.0

# Barely-there style texture only
python run.py --alpha 0.05 --beta 0.2
```

---

### 2. Region-Based Masking (`--mask_path`)

**What it does:** Confine style application to user-defined spatial regions (e.g., apply "Starry Night" style only to the sky, keep subject photorealistic).

**Parameters:**
- `--mask_path` : string, path to binary mask file (PNG or NPY)
  - Binary mask: pixels = 1 (white) → apply style, pixels = 0 (black) → skip style
  - Format: PNG (8-bit grayscale, auto-thresholded at 0.5) or NPY (float array)
  - Spatial dimensions must match content image OR will be auto-resized
  - **Default:** None (no spatial masking, style applies globally)

**How it works:**
- Binary mask is converted to an "attention bias" matrix at each U-Net layer
- During attention computation, non-masked regions are given `-infinity` attention score
- After softmax, these regions have near-zero attention to style, suppressing style in non-masked areas
- **Result:** Style strictly limited to masked region, zero "style bleeding" across boundaries

**Advantages over simple regional prompts:**
- Direct spatial control (no text description needed)
- Sharp boundaries (no gradient smoothing artifacts)
- Efficient (single inference pass, no multi-region synthesis)

**Limitations:**
- Works best with clear, well-defined regions (not soft gradients initially)
- Very complex masks may still leak style at edges (diffusion process is soft)

**Use cases:**
- Selective style application: sky, background, foreground, clothing, etc.
- Preserve photorealism in portrait while stylizing background
- Apply different "vibes" to different image regions sequentially (run multiple times)
- Creative control: paint custom patterns, textures, or design overlays

**Examples:**

#### Example 1: Style Sky Only
```bash
# Create simple white/black mask in image editor
# White (255) = apply Starry Night, Black (0) = keep original sky color
python run.py --mask_path masks/sky_region.png --beta 0.8

# Optional: pair with alpha for timing control
python run.py --mask_path masks/sky_region.png --beta 0.8 --alpha 0.15
```

#### Example 2: Preserve Subject, Style Background
```bash
# Mask with subject (foreground) = black (0), background = white (255)
python run.py --mask_path masks/subject_background.png --beta 1.0
```

#### Example 3: Fine-Grained Control
```bash
# Create mask in Photoshop, GIMP, or Python:
# from PIL import Image
# import numpy as np
# mask = Image.new('L', (512, 512), 0)  # Black background
# # Draw white region(s) where style should apply
# mask.save('custom_mask.png')

python run.py --mask_path custom_mask.png --beta 0.6
```

---

## Creating Masks

### Option 1: Image Editor (Photoshop, GIMP, etc.)
1. Open content image
2. Create new layer → fill with black (0)
3. Draw/paint white (255) in regions where style should apply
4. Delete background layer, save as PNG (grayscale)

### Option 2: Python Script
```python
from PIL import Image
import numpy as np

# Create 512x512 black mask
mask = Image.new('L', (512, 512), 0)

# Option A: Draw rectangle (e.g., for sky)
from PIL import ImageDraw
draw = ImageDraw.Draw(mask)
draw.rectangle([(0, 0), (512, 256)], fill=255)  # Top half = white (masked)

# Option B: Use numpy for complex patterns
mask_arr = np.zeros((512, 512), dtype=np.uint8)
mask_arr[:256, :] = 255  # Top half = style region
mask = Image.fromarray(mask_arr, mode='L')

mask.save('sky_mask.png')
```

### Option 3: Alpha Channel from PNG
```python
# If you have PNG with alpha channel:
png_with_alpha = Image.open('image_with_alpha.png').convert('RGBA')
alpha_channel = png_with_alpha.split()[-1]  # Extract alpha
alpha_channel.save('mask_from_alpha.png')

# Then use:
python run.py --mask_path mask_from_alpha.png
```

---

## Advanced: Target Blocks

**Experimental feature** (default: applies to all U-Net attention blocks).

```bash
# Restrict mask+beta to specific U-Net blocks
python run.py --mask_path mask.png --target_blocks "up_blocks.0,up_blocks.1"
```

**Effect:**
- Only specified blocks inject masked style
- Other blocks continue normal style injection (no mask)
- Use case: Debug where style "leaks" or to target specific spatial frequencies

**Available blocks:** `up_blocks.0`, `up_blocks.1`, `up_blocks.2`, `up_blocks.3`, `mid_block`, `down_blocks.0`, `down_blocks.1`, etc.

---

## Combining Features

### Example: Professional Portrait Styling
```bash
# Apply oil painting style to background, keep face photorealistic
python run.py \
  --content_path images/content \
  --style_path images/style \
  --alpha 0.12 \
  --beta 0.7 \
  --mask_path masks/background_only.png \
  --ddpm_steps 250
```

### Example: Subtle Texture Overlay
```bash
# Apply light texture throughout (no mask), but at half intensity
python run.py \
  --beta 0.4 \
  --alpha 0.1 \
  --ddpm_steps 150
```

### Example: Strong Masked Foreground Effect
```bash
# Heavy style only in foreground region
python run.py \
  --beta 1.0 \
  --mask_path masks/foreground.png \
  --alpha 0.2 \
  --ddpm_steps 350
```

---

## Troubleshooting

### Issue: Mask not loading
```
Error: Mask file not found: path/to/mask.png
```
**Solution:**
- Verify file path (relative to DiffuseST/ folder)
- Check file exists: `ls path/to/mask.png`
- PNG must be grayscale (not RGB/RGBA)

### Issue: Style bleeding across mask boundary
```
Symptom: Style appears outside masked region
```
**Solutions:**
1. Increase `--alpha` (more injection window)
2. Increase `--ddpm_steps` (finer control)
3. Create mask with softer/anti-aliased edges (GIMP feather option)
4. Reduce `--beta` (weaker style magnitude)

### Issue: Style doesn't appear in mask region
```
Symptom: Masked region unchanged, very similar to content
```
**Solutions:**
1. Verify mask is correct (white = apply style, black = skip)
2. Increase `--beta` to 1.0 or higher (if supported)
3. Increase `--alpha` (more steps in injection window)
4. Use `--ddpm_steps 250+` for more detailed control

### Issue: Output is too similar to content
```
Symptom: Even with beta=1.0, style is faint
```
**Solutions:**
1. Increase `--alpha` to 0.15 or 0.2 (longer injection window)
2. Reduce `--ddpm_steps` (less diffusion noise, more direct inversion)
3. Check that style image is sufficiently different from content
4. Use a different content-style pair with stronger visual contrast

---

## Parameter Interaction

### Alpha vs. Beta (Confusion Alert!)
- **`--alpha`** (existing): Controls *which timesteps* style is applied (temporal window)
  - Default 0.1 = last 10% of diffusion steps
  - Higher = more injection window = stronger overall style
  
- **`--beta`** (new): Controls *how strongly* style features influence those timesteps (magnitude)
  - Default 1.0 = full style strength during injection window
  - Lower = weaker style during injection

**Interaction:** Use alpha for pacing, beta for intensity.

```bash
# Gentle, long application
python run.py --alpha 0.2 --beta 0.5

# Intense, brief application
python run.py --alpha 0.08 --beta 1.0

# Moderate both
python run.py --alpha 0.12 --beta 0.8
```

---

## Backward Compatibility

✅ **All existing workflows unchanged:**
```bash
# These work exactly as before (no mask, beta=1.0 by default)
python run.py
python run.py --alpha 0.15 --ddpm_steps 200
python run.py --seed 42
```

---

## Output

Results saved to:
- `DiffuseST/output/{extraction_path}/{content}+{style}.png`
- Latent caches saved to `output/latents_forward/` (reused on re-runs)

---

## Next Steps & Experimental Ideas

1. **Soft masking (future)**: Continuous [0, 1] masks instead of binary
2. **Multi-mask blending**: Apply different styles to different regions (requires sequential runs)
3. **Interactive UI**: Web interface for real-time mask painting and preview
4. **Per-block beta**: Different beta values for different U-Net blocks (fine-grained control)

---

For questions or issues, refer to README.md or project_guide.md.
