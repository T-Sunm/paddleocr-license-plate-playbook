"""
Camera sensors have many imperfections and tunable settings. 
1) Contrast, 
2) Brightness, 
3) JpegCompression 
and 
4) Pixelate. 

Contrast enables us to distinguish the different objects that compose an image. 
Brightness is directly affected by scene luminance. 
JpegCompression is the side effect of image compression. 
Pixelate is exhibited by increasing the resolution of an image.

Reference: https://github.com/hendrycks/robustness
Hacked together for STR by: Rowel Atienza
"""

from io import BytesIO

import numpy as np
import skimage as sk
from PIL import Image, ImageOps
from skimage import color

'''
    PIL resize (W,H)
    cv2 image is BGR
    PIL image is RGB
'''


class Contrast:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = [0.4, .3, .2, .1, .05]
        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = np.asarray(img) / 255.
        means = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - means) * c + means, 0, 1) * 255

        return Image.fromarray(img.astype(np.uint8))


class Brightness:
    """
    LowLight/Underexpose simulation: exposure + gamma + shadow crush.
    Simulates low-light camera conditions before compression.
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # OLD: HSV V += c approach (not realistic for underexposed footage)
        # c = [.1, .2, .3]
        # img = sk.color.rgb2hsv(img)
        # img[:, :, 2] = np.clip(img[:, :, 2] + c, 0, 1)
        # img = sk.color.hsv2rgb(img)

        # NEW: exposure + gamma + shadow crush (realistic LRLPR)
        # mag=0 is now very light
        exposure = [0.70, 0.50, 0.35]
        gamma = [1.1, 1.3, 1.6]
        black = [0.00, 0.02, 0.04]

        if mag < 0 or mag >= len(exposure):
            idx = len(exposure) - 1
        else:
            idx = mag

        x = np.asarray(img).astype(np.float32) / 255.0
        e = self.rng.uniform(exposure[idx] * 0.85, exposure[idx] * 1.15)
        x = x * e
        g = self.rng.uniform(gamma[idx] * 0.90, gamma[idx] * 1.10)
        x = np.power(np.clip(x, 0, 1), g)
        b = self.rng.uniform(black[idx] * 0.8, black[idx] * 1.2)
        x = np.clip(x - b, 0, 1)

        out = Image.fromarray((x * 255.0).astype(np.uint8))
        if len(img.getbands()) == 1:
            out = ImageOps.grayscale(out)
        return out


class JpegCompression:
    def __init__(self, rng=None, passes=10):
        self.rng = np.random.default_rng() if rng is None else rng
        self.passes = passes  # 1..3 for transcoding simulation

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # OLD: c = [25, 18, 15]
        # NEW: extended for LRLPR - extreme compression at mag=2
        # passes=2-3 with quality=1-5 creates heavy artifacts while preserving text shape
        q = [25, 10, 3]
        if mag < 0 or mag >= len(q):
            index = self.rng.integers(0, len(q))
        else:
            index = mag
        quality = q[index]

        x = img
        for _ in range(self.passes):
            buf = BytesIO()
            x.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            x = Image.open(buf).convert("RGB").copy()
        return x


class Pixelate:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        # OLD: c = [0.6, 0.5, 0.4]
        # NEW: extended for LRLPR - extreme downscale at mag=2
        # NEAREST upscale creates blocky effect matching real LR footage
        s = [0.5, 0.25, 0.12]
        if mag < 0 or mag >= len(s):
            index = self.rng.integers(0, len(s))
        else:
            index = mag
        c = s[index]

        # Downscale with BOX (average), upscale with NEAREST (blocky)
        x = img.resize((max(1, int(w * c)), max(1, int(h * c))), Image.BOX)
        return x.resize((w, h), Image.NEAREST)


class LRCompression:
    """
    Simulates real surveillance footage degradation:
    Downscale → JPEG compress at LR → Upscale
    
    This creates authentic LRLPR artifacts because compression happens
    at low resolution and artifacts get magnified during upscale.
    """
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        
        # 3 levels only (0, 1, 2) - mag=0 is lighter now
        lr_h = [20, 16, 14]
        q = [40, 25, 18]
        passes_list = [1, 1, 2]
        
        if mag < 0 or mag >= len(lr_h):
            index = self.rng.integers(0, len(lr_h))
        else:
            index = mag
        
        target_h = lr_h[index]
        quality = q[index]
        passes = passes_list[index]
        
        # 1. Downscale to target height (keep aspect ratio)
        scale = target_h / h
        lr_w = max(1, int(w * scale))
        lr_img = img.resize((lr_w, target_h), Image.BOX)
        
        # 2. JPEG compress at low resolution (with subsampling for color bleeding)
        for _ in range(passes):
            buf = BytesIO()
            lr_img.save(buf, format="JPEG", quality=quality, subsampling=2)
            buf.seek(0)
            lr_img = Image.open(buf).convert("RGB").copy()
        
        # 3. Upscale back - mix BILINEAR (80%) / NEAREST (20%)
        if self.rng.uniform(0, 1) < 0.8:
            return lr_img.resize((w, h), Image.BILINEAR)
        else:
            return lr_img.resize((w, h), Image.NEAREST)

# LowLight is now integrated into Brightness class above
