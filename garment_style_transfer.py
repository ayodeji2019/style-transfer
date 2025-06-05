import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import tensorflow as tf
import tensorflow_hub as hub
import os

def process_style_transfer(CONTENT_PATH, STYLE_PATH, OUTPUT_PATH, SAM_CHECKPOINT_PATH="sam_vit_h_4b8939.pth"):
    # ----------- Load Content Image and Resize -----------
    # SAM Model Checkpoint to be downloaded from here: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    def load_and_resize(path, max_dim=512):
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img

    content_np = load_and_resize(CONTENT_PATH)
    style_np = load_and_resize(STYLE_PATH)

    # ----------- STEP 1: Segment Garment using SAM -----------
    print("Segmenting garment with SAM...")

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
    print("SAM running on:", next(sam.parameters()).device)
    predictor = SamPredictor(sam)

    input_image = content_np.copy()
    predictor.set_image(input_image)

    # Select a point roughly on the garment for SAM (centered click)
    point = np.array([[input_image.shape[1]//2, input_image.shape[0]//2]])
    labels = np.array([1])

    masks, scores, _ = predictor.predict(point_coords=point, point_labels=labels, multimask_output=True)
    garment_mask = masks[np.argmax(scores)]

    # ----------- STEP 2: Style Transfer Only on Garment Region -----------

    def to_tensorflow_img(np_img):
        img = tf.convert_to_tensor(np_img, dtype=tf.float32) / 255.0
        img = tf.image.resize_with_pad(img, 512, 512)
        return img[tf.newaxis, :]

    print("Preparing garment region for style transfer...")

    garment_mask = garment_mask.astype(np.uint8)
    garment_mask_3ch = np.stack([garment_mask]*3, axis=-1)

    masked_garment = np.where(garment_mask_3ch == 1, content_np, 0)

    ys, xs = np.where(garment_mask == 1)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped_garment = masked_garment[y_min:x_max+1, x_min:x_max+1]
    cropped_mask = garment_mask[y_min:x_max+1, x_min:x_max+1]

    cropped_garment_tf = to_tensorflow_img(cropped_garment)
    style_tf = to_tensorflow_img(style_np)

    print("Running style transfer on garment region...")
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_img = hub_model(cropped_garment_tf, style_tf)[0]
    stylized_np = (tf.squeeze(stylized_img).numpy() * 255).astype(np.uint8)

    stylized_np = cv2.resize(stylized_np, (cropped_garment.shape[1], cropped_garment.shape[0]))

    # ----------- STEP 3: Merge Stylized Region Back -----------
    print("Merging stylized garment with original image...")

    result_np = content_np.copy()
    cropped_mask_3ch = np.stack([cropped_mask]*3, axis=-1)

    result_np[y_min:x_max+1, x_min:x_max+1] = np.where(cropped_mask_3ch == 1, stylized_np, result_np[y_min:x_max+1, x_min:x_max+1])

    # Save the result
    cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))

    # Optionally show output
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(content_np)
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(style_np)
    plt.title("Style")

    plt.subplot(1, 3, 3)
    plt.imshow(result_np)
    plt.title("Stylized Garment")
    plt.axis('off')
    plt.show()