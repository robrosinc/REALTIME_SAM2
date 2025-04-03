import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)
from PIL import Image

clicked_coords = []
input_labels = []

def mouse_click_event(event, x, y, flags, param):  # Combined callback
    """Callback function to capture both left and right mouse clicks."""
    global clicked_coords, input_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords.append([x, y])
        input_labels.append(1)
        print(f"Left Clicked coordinates: {clicked_coords}")
        print(f"Labels: {input_labels}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_coords.append([x, y])
        input_labels.append(0)  # Or [0] if you want a list
        print(f"Right Clicked coordinates: {clicked_coords}")
        print(f"Labels: {input_labels}") # Print the labels after right click

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
instance_id_dir= "../../mask_dataset/data/6/41006849"

instance_path = os.path.join(instance_id_dir, "sparse", "0")
mask_npz_path = os.path.join(instance_path, "binary_masks.npz")
result_path = os.path.join(instance_path, "results")
image_dir = os.path.join(instance_id_dir, "images")

# scan all the JPEG frame names in this directory
frame_names = [
    p.split('.')[0]
    for p in os.listdir(image_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(p))

# take a look the first video frame
frame_idx = 0

image = cv2.imread(os.path.join(image_dir, frame_names[frame_idx]+".jpg"))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if torch.cuda.is_available():
    device = torch.device("cuda")

# model
checkpoint = "../checkpoints/efficienttam_s.pt"
model_cfg = "../efficient_track_anything/configs/efficienttam/efficienttam_s.yaml"

predictor = build_efficienttam_video_predictor(model_cfg, checkpoint, device=device)
cv2.namedWindow("Click to Segment", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Click to Segment", mouse_click_event)

print("Click on the image to select points for segmentation. Press 'q' to confirm.")

while True:
    display_image = image.copy()
    for coord in clicked_coords:
        cv2.circle(display_image, tuple(coord), 5, (0, 0, 255), -1)
    cv2.imshow("Click to Segment", display_image) 
    cv2.resizeWindow("Click to Segment", 1600, 1200) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_path=image_dir)
prompts = {}  # hold all the clicks we add for visualization

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = (
    2  # give a unique id to each object we interact with (it can be any integers)
)

# Let's add a positive click at (x, y) = (200, 300) to get started on the first object
points = np.array(clicked_coords, dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array(input_labels, np.int32)
prompts[ann_obj_id] = points, labels
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

video_segments = {}  # video_segments contains the per-frame segmentation results
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

# render the segmentation results every few frames
for out_frame_idx in range(0, len(frame_names)):
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        mask_data = np.load(mask_npz_path)
        binary_masks = mask_data["binary_masks"]

        ord_image = cv2.imread(os.path.join(image_dir, frame_names[out_frame_idx]+".jpg"))

        # Update the mask at the given index
        bool_mask = out_mask[0].astype(bool)
        binary_masks[out_frame_idx] = bool_mask

        # Save the updated mask back to .npz
        np.savez_compressed(mask_npz_path, binary_masks=binary_masks)
        print(f"Updated mask saved inside {mask_npz_path} for index {out_frame_idx}")

        # Apply dimming effect to unmasked areas (using the boolean mask directly)
        masked_img = ord_image.copy()
        masked_img[~bool_mask] = (masked_img[~bool_mask] * 0.3).astype(np.uint8) # Use ~ for NOT

        # Save the resulting masked image
        os.makedirs(result_path, exist_ok=True)
        result_image_path = os.path.join(result_path, f"masked_{frame_names[out_frame_idx]}"+".png")
        cv2.imwrite(result_image_path, masked_img)  # Save the dimmed image directly
        print(f"Masked image saved at: {result_image_path}")