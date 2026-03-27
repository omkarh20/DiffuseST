<h1>DiffuseST: Unleashing the Capability of the Diffusion Model for Style Transfer</h1>

Ying Hu, [Chenyi Zhuang](https://chenyi-zhuang.github.io/), Pan Gao

[I2ML](https://i2-multimedia-lab.github.io/), Nanjing University of Aeronautics and Astronautics

[Paper](https://arxiv.org/abs/2410.15007)

### ⚙️ Setup and Usage
```bash
conda create --name DiffuseST python=3.8
conda activate DiffuseST

# Install requirements
pip install -r requirements.txt
```

Download the pre-trained [blipdiffusion](https://huggingface.co/salesforce/blipdiffusion). 

Put the content images in `images/content` and the style images in `images/style`.

```bash
# Run DiffuseST, alpha default to 0.1
python run.py

# Perform style injection for more steps
python run.py --alpha 0.2
```

---

### 🎛️ Enhanced Features (Beta & Regional Masking)

This implementation includes **training-free** enhancements for finer control:

#### **Style Strength Control (`--beta`)**
Control the magnitude of style influence continuously from 0% to 100%, independent of timing.

```bash
# Subtle style (30% strength)
python run.py --beta 0.3

# Balanced (70% strength)
python run.py --beta 0.7

# Full style (default)
python run.py --beta 1.0
```

#### **Region-Based Masking (`--mask_path`)**
Apply style only to user-defined spatial regions (e.g., style the sky, keep person photorealistic).

```bash
# Create a binary mask (white = apply style, black = skip)
# Then run:
python run.py --mask_path masks/sky_mask.png --beta 0.8

# Combine with alpha for timing control
python run.py --mask_path masks/bg_only.png --alpha 0.15 --beta 0.9
```

**How it works:**
- Binary masks restrict style injection to specific image regions via attention bias
- Sharp boundaries, no style bleeding
- Works efficiently in single pass (no multi-region synthesis required)

#### **Full Documentation**
Read [USAGE_ENHANCED.md](USAGE_ENHANCED.md) for:
- Detailed parameter explanations
- Creating masks (tutorial + Python code)
- Use case examples (portrait styling, texture overlays, etc.)
- Troubleshooting & parameter interactions
- Advanced: layer-specific targeting

---

### Examples with Enhanced Features

```bash
# Professional portrait: oil painting background, photorealistic face
python run.py \
  --mask_path masks/background.png \
  --beta 0.7 \
  --alpha 0.12

# Subtle texture: light style throughout
python run.py --beta 0.4 --alpha 0.1

# Strong masked effect: intense style in foreground only
python run.py \
  --mask_path masks/foreground.png \
  --beta 1.0 \
  --alpha 0.2
```

**Backward Compatible:** Existing scripts work unchanged. New features are optional.
