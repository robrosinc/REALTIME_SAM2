from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import numpy as np
import time
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor

app = Flask(__name__)

# TAM Predictor initialization
tam_checkpoint = "../checkpoints/efficienttam_ti_512x512.pt"
model_cfg = "../efficient_track_anything/configs/efficienttam/efficienttam_ti_512x512.yaml"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    raise RuntimeError("No CUDA or MPS device found")
predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint, device=device)
classes = [0, 1, 2, 3]  # Adjust number of classes

# Initialize global variables !!Not best practice!!
click_points = {cls: [] for cls in classes}  # Store clicked points for all classes
current_class = 0
reset = False
reset_class = 0
current_frame_idx = 0

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
    no_obj_points = np.array([[0,0]], dtype=np.float32)
    no_obj_labels = np.array([-1], dtype=np.int32)

    # Give empty points before starting the camera, because if number of classes change during tracking,
    # stack inside get_memory_cond_feat() will cause an error
    for cls in classes:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictor.add_new_points(
                frame_idx=0,
                obj_id=cls,
                points=no_obj_points,
                labels=no_obj_labels,
            )

    while True:
        global current_frame_idx, reset
        current_frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Whether there is a prompt in this frame
        new_input = False

        # Tells whether that class is the first prompt in this frame,
        # This is leftover from the yolo example, where yolo prompts multiple classes in a single frame
        first_hit = True
        
        # We perform track() regardless if a prompt exists or not. track() saves the results in output_dict['non_cond_frame'][current_frame_idx]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_mask_logits = predictor.track(frame)
        
        Gathered_matrix = {cls: {'points': [], 'labels': [], 'first_hit': []} for cls in classes}

        for cls in classes:
            if click_points[cls]:
                new_input = True
                for point in click_points[cls]:
                    Gathered_matrix[cls]['points'].append(point)
                    Gathered_matrix[cls]['labels'].append(1)
                    Gathered_matrix[cls]['first_hit'].append(first_hit)
                    first_hit = False
                click_points[cls] = []
        
        if new_input or reset:
            for cls in classes:
                if Gathered_matrix[cls]['points']:
                    points = np.array(Gathered_matrix[cls]['points'], dtype=np.float32)
                    labels = np.array(Gathered_matrix[cls]['labels'], dtype=np.int32)
                    first_hit = np.array(Gathered_matrix[cls]['first_hit'], dtype=np.bool_)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictor.add_new_points_during_track(cls, points, labels, first_hit=first_hit[0], frame=frame)
            
            if reset:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictor.add_new_points(
                        frame_idx=current_frame_idx,
                        obj_id=reset_class,
                        points=no_obj_points,
                        labels=no_obj_labels,
                        new_input=True
                    )
                reset = False
                # The resetting mechanism is same as new_input. We produce an empty mask for that object for the current frame,
                # overwrite the object slice in the consolidated output, delete the memory up to this frame and start from this frame's output.
                # Again the other objects are not affected as their output for this frame was created by track().


            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, out_mask_logits = predictor.finalize_new_input()
            # When new_input is true, memory up to this frame is deleted at the end of consolidate_temp_output_across_obj() inside finalize_new_input()
            # Because track() is called for every frame, the other objects' mask for the current frame stays, and the temp output for the prompted object becomes consolidated
            # We sort of 'tricked' the model into viewing the result of the prompted frame as the starting memory


        mask_logits = out_mask_logits.cpu().numpy()

        # Visualize masks
        frame_with_mask = apply_mask_to_frame(frame, mask_logits[0:4])
        
        # 클래스 정보 및 현재 선택된 클래스 표시
        cv2.putText(frame_with_mask, f"Selected Class: {current_class}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 각 클래스 표시
        for i, cls in enumerate(classes):
            color = (0, 255, 0) if cls == current_class else (200, 200, 200)
            cv2.putText(frame_with_mask, f"Class {cls}", (10, 60 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Visualize prompts
        for cls in classes:
            if Gathered_matrix[cls]['points']:
                for point in Gathered_matrix[cls]['points']:
                    center_x, center_y = point
                    # 클래스 색상 가져오기 (apply_mask_to_frame 함수의 colors와 일치)
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                    color = colors[cls] if cls < len(colors) else (255, 255, 255)
                    
                    cv2.circle(frame_with_mask, (int(center_x), int(center_y)), 5, color, -1)
                    cv2.circle(frame_with_mask, (int(center_x), int(center_y)), 10, color, 2)
                    cv2.putText(frame_with_mask, f"C{cls}", (int(center_x) + 15, int(center_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        _, buffer = cv2.imencode('.jpg', frame_with_mask)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

def apply_mask_to_frame(frame, mask_logits, colors=None):
    if isinstance(mask_logits, torch.Tensor):
        mask_logits = mask_logits.cpu().numpy()

    if colors is None:
        colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 255)
        ]
    
    mask_colored = np.zeros_like(frame, dtype=np.uint8)

    for class_idx in range(mask_logits.shape[0]):
        mask = (mask_logits[class_idx] > 0.0).astype(np.uint8) * 255

        for i in range(3):
            mask_colored[:, :, i] = np.clip(
                mask_colored[:, :, i] + (mask * (colors[class_idx][i] / 255)).astype(np.uint8), 
                0, 
                255
            )

    return cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

@app.route('/')
def index():
    return render_template('app_index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click', methods=['POST'])
def handle_click():
    data = request.json
    x = data['x']
    y = data['y']
    
    global current_class
    click_points[current_class].append([x, y])
    print(f"Click at (x: {x}, y: {y}) for Class {current_class}")
    return jsonify({'status': 'success', 'class': current_class,'coordinates': {'x': x, 'y': y}})


@app.route('/reset_class', methods=['POST'])
def reset_class():
    data = request.json
    class_num = data['class']

    global reset, reset_class
    if class_num in classes:
        reset = True
        reset_class = class_num
    else:
        return jsonify({'status': 'error', 'message': 'Invalid class'})

    return jsonify({'status': 'success', 'reset': reset, 'reset_class': reset_class})

@app.route('/change_class', methods=['POST'])
def change_class():
    data = request.json
    new_class = data['class']
    
    global current_class
    if new_class in classes:
        current_class = new_class
        return jsonify({'status': 'success', 'current_class': current_class})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid class'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
