
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# 1. Load the image
image = Image.open("assets/images/bottle2.jpg").convert("RGB")

# 2. Load SAM 3
print("Loading SAM 3 model...")
model = build_sam3_image_model()
processor = Sam3Processor(model)

# 3. Prepare the image
# 3. Prepare image and run segmentation
print("Preparing image and segmenting bottle...")

with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        inference_state = processor.set_image(image)

        output = processor.set_text_prompt(
            state=inference_state,
            prompt="bottle cap"
        )

# 4. Extract results
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]


# 5. Display the image and segmentation results
image_np = np.array(image)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_np)
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(image_np)

# Overlay every detected mask
for mask in masks:
    mask_np = mask.squeeze().detach().cpu().numpy()
    axes[1].imshow(
        np.ma.masked_where(mask_np < 0.5, mask_np),
        alpha=0.5,
        cmap="spring",
        vmin=0,
        vmax=1
    )

axes[1].set_title('SAM 3: "bottle cap"')
axes[1].axis("off")

plt.tight_layout()
os.makedirs("sam3/results", exist_ok=True)
plt.savefig("sam3/results/bottle_segmentation.png", dpi=150)
plt.show()
