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
predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint)
# Configuration
classes = [0, 1, 2, 3]  # 4개의 클래스 유지
click_points = {cls: [] for cls in classes}  # 각 클래스별 클릭 포인트 저장
current_class = 0
current_frame = 0

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
    points = np.array([[0,0]], dtype=np.float32)
    labels = np.array([-1], dtype=np.int32)

    # 초기화: 각 클래스에 대해 빈 포인트 설정
    for cls in classes:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, out_mask_logits = predictor.add_new_points(
                frame_idx=0,
                obj_id=cls,
                points=points,
                labels=labels,
            )

    while True:
        global current_frame
        current_frame += 1
        ret, frame = cap.read()
        if not ret:
            break

        new_input = False
        first_hit = True
        # torch.cuda.synchronize()
        # start_time = time.time()
        
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
        
        if new_input:
            print("item detected")
            for cls in classes:
                if Gathered_matrix[cls]['points']:
                    points = np.array(Gathered_matrix[cls]['points'], dtype=np.float32)
                    labels = np.array(Gathered_matrix[cls]['labels'], dtype=np.int32)
                    first_hit = np.array(Gathered_matrix[cls]['first_hit'], dtype=np.bool_)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictor.add_new_points_during_track(cls, points, labels, first_hit=first_hit[0], frame=frame)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, out_mask_logits = predictor.finalize_new_input()

        mask_logits = out_mask_logits.cpu().numpy()

        # 마스크 시각화
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

        # torch.cuda.synchronize()
        # elapsed_time = time.time()-start_time
        # print(elapsed_time)

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
    
    # 현재 선택된 클래스에 클릭 포인트 추가
    global current_class
    click_points[current_class].append([x, y])
    print(f"Click at (x: {x}, y: {y}) for Class {current_class}")
    return jsonify({'status': 'success', 'class': current_class,'coordinates': {'x': x, 'y': y}})


@app.route('/reset_class', methods=['POST'])
def reset_class():
    data = request.json
    class_num = data['class']

    if class_num not in classes:
        return jsonify({'status': 'error', 'message': 'Invalid class'})
    
    # 해당 클래스만 초기화
    points = np.array([[0,0]], dtype=np.float32)
    labels = np.array([-1], dtype=np.int32)
    
    global current_frame, predictor
    print("inside request", current_frame)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, _, _ = predictor.add_new_points(
            frame_idx=current_frame,
            obj_id=class_num,
            points=points,
            labels=labels,
            new_input=True
        )
    
    return jsonify({'status': 'success', 'class': class_num})

@app.route('/change_class', methods=['POST'])
def change_class():
    """클래스 변경 처리"""
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
