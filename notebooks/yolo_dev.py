from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
import time
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
from ultralytics import YOLO

app = Flask(__name__)

# TAM Predictor initialization
tam_checkpoint = "../checkpoints/efficienttam_ti_512x512.pt"
model_cfg = "../efficient_track_anything/configs/efficienttam/efficienttam_ti_512x512.yaml"
predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint)

# YOLO model initialization
yolo_model = YOLO("block_sort_yolo.pt")
classes = [0, 1, 2, 3] # blue block, orange block, yellow box, navy box in order

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Cannot access the camera.")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Error: Could not read the initial frame.")

    # Load the first frame into the predictor
    predictor.load_first_frame(frame, len(classes))
    points = np.array([[-1,-1]], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)
    # predictor.set_num_class(len(classes))
    # Give dummy input (negative points) for each of the classes
    for cls in classes:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, out_mask_logits = predictor.add_new_points(
                frame_idx=0,
                obj_id=cls,
                points=points,
                labels=labels,
            )
    while True:
        new_input = False
        first_hit = True
        if not ret:
            break
        # Initialize Gathered_matrix with empty lists for points and labels for each class
        Gathered_matrix = {cls: {'points': [], 'labels': [], 'first_hit': []} for cls in classes}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_mask_logits = predictor.track(frame)
        # Perform YOLO object detection
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yolo_results = yolo_model(frame, stream=True, verbose=False)

        for result in yolo_results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes_detected = result.boxes.cls.cpu().numpy()  # Class labels

            for box, confidence, cls in zip(boxes, confidences, classes_detected):
                if (cls ==0 and confidence>0.2) or (cls ==2 and confidence>0.9): #or (cls ==3 and confidence>0.2):
                    new_input = True
                    
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{yolo_model.names[int(cls)]}: {confidence:.2f}"

                    # These drawing operations are delayed until after finalize_new_input or track
                    Gathered_matrix[int(cls)]['points'].append([(x1 + x2) / 2, (y1 + y2) / 2])
                    Gathered_matrix[int(cls)]['labels'].append(1)
                    Gathered_matrix[int(cls)]['first_hit'].append(first_hit)
                    first_hit = False

        # Gather items of each class and append them all at once as one prompt
        if new_input == True:
            for cls in classes:
                if Gathered_matrix[cls]['points']:
                    points = np.array(Gathered_matrix[cls]['points'], dtype=np.float32)
                    labels = np.array(Gathered_matrix[cls]['labels'], dtype=np.int32)
                    first_hit = np.array(Gathered_matrix[cls]['first_hit'], dtype=np.bool_)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictor.add_new_points_during_track(cls, points, labels, first_hit = first_hit[0], frame = frame)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, out_mask_logits = predictor.finalize_new_input()

        mask_logits = out_mask_logits.cpu().numpy()

        # Apply the mask and colors to the frame
        frame_with_mask = apply_mask_to_frame(frame, mask_logits[0:4])

        # Show where the prompt is
        for cls in classes:
            if Gathered_matrix[cls]['points']:
                for point, label in zip(Gathered_matrix[cls]['points'], Gathered_matrix[cls]['labels']):
                    # Get the center point of the detected object
                    center_x, center_y = point
                    cv2.putText(frame_with_mask, f"{label}", (int(center_x), int(center_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.line(frame_with_mask, (int(center_x), int(center_y)), (int(center_x), int(center_y)), (255, 255, 255), 10)

        # Encode the frame as a JPEG image
        _, buffer = cv2.imencode('.jpg', frame_with_mask)
        frame = buffer.tobytes()
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        ret, frame = cap.read()
    cap.release()



def apply_mask_to_frame(frame, mask_logits, colors=None):

    if isinstance(mask_logits, torch.Tensor):
        mask_logits = mask_logits.cpu().numpy()  # Convert to numpy array if it's a PyTorch tensor

    if colors is None:
        # Define distinct colors for up to 4 classes (extend as needed)
        colors = [
            (0, 255, 0),   # Red for class 0
            (255, 0, 0),   # Green for class 1
            (0, 0, 255),   # Blue for class 2
            (255, 255, 0)  # Yellow for class 3
        ]
    
    mask_colored = np.zeros_like(frame, dtype=np.uint8)

    # Iterate over each class and apply its respective color
    for class_idx in range(mask_logits.shape[0]):
        # Threshold to generate a binary mask for the class
        mask = (mask_logits[class_idx] > 0.0).astype(np.uint8) * 255

        # Apply the mask color to the frame
        for i in range(3):  # R, G, B channels
            mask_colored[:, :, i] = np.clip(
                mask_colored[:, :, i] + (mask * (colors[class_idx][i] / 255)).astype(np.uint8), 
                0, 
                255
            )
    # Blend the original frame with the colored masks
    return cv2.addWeighted(frame, 0.5, mask_colored, 0.5, 0)



@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('demo_index.html')


@app.route('/video_feed')
def video_feed():
    """Stream video frames to the client."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
