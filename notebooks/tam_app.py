from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import numpy as np
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor

app = Flask(__name__)

# --- Model init (unchanged) ---
tam_checkpoint = "../checkpoints/efficienttam_ti_512x512.pt"
model_cfg = "../efficient_track_anything/configs/efficienttam/efficienttam_ti_512x512.yaml"
<<<<<<< Updated upstream
predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint)
classes = [0, 1, 2, 3]  # Adjust number of classes

# Initialize global variables !!Not best practice!!
click_points = {cls: [] for cls in classes}  # Store clicked points for all classes
=======
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    raise RuntimeError("No CUDA or MPS device found")

predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint, device=device)
classes = [0, 1, 2, 3]

# --- Global state ---
click_points = {cls: [] for cls in classes}    # point prompts per class
box_prompts   = {cls: [] for cls in classes}    # NEW: bbox prompts per class -> list of (x1,y1,x2,y2)
>>>>>>> Stashed changes
current_class = 0
reset_flag = False
reset_class_id = 0
current_frame_idx = 0

# modes: "click" | "text" | "dual" | "box"
current_mode = "click"

# Sticky text prompt (persists across frames until reset)
active_text_prompt = ""   # NEW

# For telemetry / overlay
text_prompts_log = []     # [{frame_idx, text, class_id, xy or box, source}]

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Cannot access the camera.")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read the initial frame.")

<<<<<<< Updated upstream
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
=======
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float32):
        predictor.load_first_frame(frame, len(classes))

    no_obj_points = np.array([[0, 0]], dtype=np.float32)
    no_obj_labels = np.array([-1], dtype=np.int32)
>>>>>>> Stashed changes

    while True:
        global current_frame_idx, reset_flag
        current_frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break

        new_input = False
        first_hit = True

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float32):
            _, out_mask_logits = predictor.track(frame)

        # Gather prompts issued before this frame renders
        Gathered = {cls: {'points': [], 'labels': [], 'boxes': [], 'first_hit': []} for cls in classes}

        for cls in classes:
            if click_points[cls]:
                new_input = True
                for pt in click_points[cls]:
                    Gathered[cls]['points'].append(pt)
                    Gathered[cls]['labels'].append(1)
                    Gathered[cls]['first_hit'].append(first_hit)
                    first_hit = False
                click_points[cls] = []

        for cls in classes:
            if box_prompts[cls]:
                new_input = True
                for (x1, y1, x2, y2) in box_prompts[cls]:
                    Gathered[cls]['boxes'].append([x1, y1, x2, y2])
                    Gathered[cls]['first_hit'].append(first_hit)
                    first_hit = False
                box_prompts[cls] = []

        if new_input or reset_flag:
            for cls in classes:
                if Gathered[cls]['points']:
                    points = np.array(Gathered[cls]['points'], dtype=np.float32)
                    labels = np.array(Gathered[cls]['labels'], dtype=np.int32)
                    first_hit_arr = np.array(Gathered[cls]['first_hit'], dtype=np.bool_)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float32):
                        predictor.add_new_prompts_during_track(
                            cls, points, labels, first_hit=first_hit_arr[0], frame=frame
                        )
                elif Gathered[cls]['boxes']:
                    boxes = np.array(Gathered[cls]['boxes'], dtype=np.float32)
                    first_hit_arr = np.array(Gathered[cls]['first_hit'], dtype=np.bool_)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float32):
                        predictor.add_new_prompts_during_track(
                            cls, boxes=boxes, first_hit=first_hit_arr[0], frame=frame
                        )

            if reset_flag:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float32):
                    predictor.add_new_prompts(
                        frame_idx=current_frame_idx,
                        obj_id=reset_class_id,
                        points=no_obj_points,
                        labels=no_obj_labels,
                        new_input=True
                    )
                reset_flag = False

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='cuda' else torch.float32):
                _, _, out_mask_logits = predictor.finalize_new_input()

        mask_logits = out_mask_logits.cpu().numpy()
        frame_with_mask = apply_mask_to_frame(frame, mask_logits[0:4])
<<<<<<< Updated upstream
        
        # 클래스 정보 및 현재 선택된 클래스 표시
        cv2.putText(frame_with_mask, f"Selected Class: {current_class}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 각 클래스 표시
=======

        # HUD
        cv2.putText(frame_with_mask, f"Selected Class: {current_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
>>>>>>> Stashed changes
        for i, cls in enumerate(classes):
            color = (0,255,0) if cls == current_class else (200,200,200)
            cv2.putText(frame_with_mask, f"Class {cls}", (10, 60 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
<<<<<<< Updated upstream

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


=======
        cv2.putText(frame_with_mask, f"Mode: {current_mode}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

        # Show active sticky text prompt
        if active_text_prompt:
            cv2.putText(frame_with_mask, f'Text: "{active_text_prompt[:46]}" (sticky)', (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

        # Show last submitted prompt event
        if text_prompts_log:
            last = text_prompts_log[-1]
            snippet = (last.get("text") or "")[:34]
            src = last.get("source")
            cv2.putText(frame_with_mask, f'Last: "{snippet}" @F{last.get("frame_idx")} ({src})', (10, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)

        # Encode and stream
>>>>>>> Stashed changes
        _, buffer = cv2.imencode('.jpg', frame_with_mask)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def apply_mask_to_frame(frame, mask_logits, colors=None):
    if isinstance(mask_logits, torch.Tensor):
        mask_logits = mask_logits.cpu().numpy()
    if colors is None:
        colors = [(0,255,0),(0,0,255),(255,0,0),(0,255,255)]
    mask_colored = np.zeros_like(frame, dtype=np.uint8)
    for class_idx in range(mask_logits.shape[0]):
        mask = (mask_logits[class_idx] > 0.0).astype(np.uint8) * 255
        for i in range(3):
            mask_colored[:,:,i] = np.clip(
                mask_colored[:,:,i] + (mask * (colors[class_idx][i] / 255)).astype(np.uint8),
                0, 255
            )
    return cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

# ---------------- Routes ----------------

@app.route('/')
def index():
    return render_template('app_index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_mode')
def get_mode():
    # Also send current sticky text for UI convenience
    return jsonify({"mode": current_mode, "active_text": active_text_prompt})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    data = request.json or {}
    mode = data.get('mode')
    if mode not in ('click', 'text', 'dual', 'box'):
        return jsonify({'status':'error', 'message':'Invalid mode'}), 400
    current_mode = mode
    return jsonify({'status':'success', 'mode': current_mode})

@app.route('/click', methods=['POST'])
def handle_click():
    data = request.json or {}
    x = float(data.get('x'))
    y = float(data.get('y'))
    maybe_text = data.get('text')  # may be None

    global current_class, current_frame_idx, active_text_prompt

    # Click/dual still generate point prompts
    if current_mode in ('click', 'dual'):
        click_points[current_class].append([x, y])

    # In dual mode, pair the text: prefer supplied text; fallback to sticky text if present
    applied_text = None
    if current_mode == 'dual':
        use_text = (maybe_text or "").strip() or active_text_prompt
        if use_text:
            text_prompts_log.append({
                "frame_idx": current_frame_idx,
                "text": use_text,
                "class_id": current_class,
                "xy": (x, y),
                "source": "dual-click"
            })
            applied_text = use_text

    return jsonify({
        'status': 'success',
        'class': current_class,
        'coordinates': {'x': x, 'y': y},
        'frame_idx': current_frame_idx,
        'applied_text': applied_text
    })

@app.route('/box', methods=['POST'])
def handle_box():
    """Receive bounding-box prompts from the UI (box mode)."""
    data = request.json or {}
    x1 = float(data.get('x1'))
    y1 = float(data.get('y1'))
    x2 = float(data.get('x2'))
    y2 = float(data.get('y2'))

    global current_class, current_frame_idx
    box_prompts[current_class].append((x1, y1, x2, y2))
    # Telemetry
    text_prompts_log.append({
        "frame_idx": current_frame_idx,
        "text": active_text_prompt or "",
        "class_id": current_class,
        "box": (x1, y1, x2, y2),
        "source": "box"
    })
    return jsonify({'status':'success', 'class': current_class, 'frame_idx': current_frame_idx})

@app.route('/text_prompt', methods=['POST'])
def text_prompt():
    """
    Set/replace the sticky text prompt. It remains in effect for all subsequent frames
    until a reset is triggered, at which point it becomes '' again.
    """
    data = request.json or {}
    text = (data.get('text') or "").strip()
    global active_text_prompt, current_frame_idx, current_class
    if not text:
        return jsonify({'status':'error', 'message':'Empty text'}), 400

    active_text_prompt = text  # <-- STICKY
    text_prompts_log.append({
        "frame_idx": current_frame_idx,
        "text": active_text_prompt,
        "class_id": current_class,
        "xy": None,
        "source": "text-sticky"
    })
    # TODO: If your predictor supports language conditioning, apply `active_text_prompt` here.

    return jsonify({'status':'success', 'frame_idx': current_frame_idx, 'active_text': active_text_prompt})

@app.route('/reset_class', methods=['POST'])
def reset_class():
    data = request.json or {}
    class_num = data.get('class')

    global reset_flag, reset_class_id, active_text_prompt
    if class_num in classes:
        reset_flag = True
        reset_class_id = class_num
        active_text_prompt = ""   # <-- clear sticky text on reset
    else:
        return jsonify({'status': 'error', 'message': 'Invalid class'})

    return jsonify({'status': 'success', 'reset': reset_flag, 'reset_class': reset_class_id, 'active_text': active_text_prompt})

@app.route('/change_class', methods=['POST'])
def change_class():
    data = request.json or {}
    new_class = data.get('class')

    global current_class
    if new_class in classes:
        current_class = new_class
        return jsonify({'status': 'success', 'current_class': current_class})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid class'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

