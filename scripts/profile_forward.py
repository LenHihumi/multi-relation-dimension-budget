"""Forward-pass ceiling for CLIP-L/14 on one A800: what images/s can the GPU actually do?

Separates "the pipeline is slow" from "the model is slow" so the encoder is optimised where the
time actually is.  Prints fp32 vs fp16 throughput for a few batch sizes; nothing is written.
"""
import time

import torch
from transformers import CLIPModel

from sb_paths import SB_ROOT, SB_OUT, SB_CACHE, SB_MODELS, REPO  # paths resolve relative to the deposit
dev = 'cuda'
model = CLIPModel.from_pretrained(SB_MODELS + '/clip-vit-large-patch14').to(dev).eval()
print('gpu', torch.cuda.get_device_name(0))

for dt in (torch.float32, torch.float16):
    m = model.to(dt)
    for bs in (128, 512):
        x = torch.randn(bs, 3, 224, 224, device=dev, dtype=dt)
        with torch.no_grad():
            for _ in range(3):
                m.get_image_features(pixel_values=x)
            torch.cuda.synchronize()
            t0 = time.time()
            n = 10
            for _ in range(n):
                m.get_image_features(pixel_values=x)
            torch.cuda.synchronize()
        el = time.time() - t0
        print('%-9s bs=%3d  %.1f img/s  (%.1f ms/batch)'
              % (str(dt).replace('torch.', ''), bs, n * bs / el, 1e3 * el / n))
