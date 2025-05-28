import torch
from PIL import Image
import numpy as np
import cv2
import os

import sys
zoedepth_path = "/Users/athenamo/Documents/GitHub/ZoeDepth"
if zoedepth_path not in sys.path:
    sys.path.insert(0, zoedepth_path)

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import pil_to_batched_tensor, save_raw_16bit, colorize

#Load ZoeDepth model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
conf = get_config("zoedepth", "infer")
zoe = build_model(conf).to(DEVICE)


#Load input soccer frame
img_path = "./Soccer.jpg"  # change to your image path
image = Image.open(img_path).convert("RGB")

#Run depth inference
depth_tensor = zoe.infer_pil(image, output_type="tensor")  # output: [1, 1, H, W]
depth_numpy = depth_tensor.squeeze().cpu().numpy()

#Normalize depth
depth_min, depth_max = depth_numpy.min(), depth_numpy.max()
depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min)
depth_uint8 = (depth_normalized * 255).astype(np.uint8)

#Mock U-Net segmentation mask 
h, w = depth_uint8.shape
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (w // 2, h // 2), min(h, w) // 4, 255, -1)  # circular "player" mask

#Apply mask to enhance depth map
foreground = cv2.bitwise_and(depth_uint8, depth_uint8, mask=mask)
background_mask = cv2.bitwise_not(mask)
blurred_background = cv2.GaussianBlur(depth_uint8, (25, 25), 0)
background = cv2.bitwise_and(blurred_background, blurred_background, mask=background_mask)
combined = cv2.add(foreground, background)

#Save results
os.makedirs("./output", exist_ok=True)
cv2.imwrite("./output/depth_raw.png", depth_uint8)
cv2.imwrite("./output/mask_used.png", mask)
cv2.imwrite("./output/depth_enhanced.png", combined)

#colorize
colored_output = cv2.applyColorMap(combined, cv2.COLORMAP_INFERNO)
cv2.imwrite("./output/depth_colored.png", colored_output)

print("✅ Enhanced depth maps saved in ./output/")
