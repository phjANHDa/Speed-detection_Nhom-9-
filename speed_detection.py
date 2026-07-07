import argparse
import math
import os
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

CAMERA_PRESETS = {
    "cam1": {
        "name": "CAM 1 - Highway",
        "input": "video/highway.mp4",
        "output": "video/ket_qua_cam1.avi",
        "red_line": ((172, 198), (774, 198)),
        "green_line": ((8, 268), (927, 268)),
        "red_y": 198,
        "green_y": 268,
        "distance_m": 15,
        "lane_guides": [
            ((170, 170), (150, 330)),
            ((340, 165), (330, 330)),
            ((610, 165), (620, 330)),
            ((800, 170), (835, 330)),
        ],
    },
}

LANE_LIMITS_LEFT_TO_RIGHT = [80, 80, 90, 90, 80, 80]
LANE_BOUNDARIES_X = [0, 170, 340, 510, 610, 800, 1020]


class Tracker:
    def __init__(self, max_distance=35):
        self.center_points = {}
        self.id_count = 0
        self.max_distance = max_distance

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for object_id, point in self.center_points.items():
                dist = math.hypot(cx - point[0], cy - point[1])
                if dist < self.max_distance:
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            new_center_points[object_id] = self.center_points[object_id]

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle speed detection with YOLOv8.")
    parser.add_argument("--camera", choices=CAMERA_PRESETS.keys(), default="cam1", help="Camera preset.")
    parser.add_argument("--input", default=None, help="Input video path. Overrides selected camera preset.")
    parser.add_argument("--output", default=None, help="Output video path. Overrides selected camera preset.")
    parser.add_argument("--weights", default="yolov8s.pt", help="YOLO weights path.")
    parser.add_argument("--device", default="auto", help="Device to run YOLO on: auto, cpu, 0, cuda:0.")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames.")
    return parser.parse_args()


def draw_label(frame, text, origin, font_scale=0.65, fg=(255, 255, 255), bg=(24, 24, 28), thickness=1):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x - 8, y - text_h - 8), (x + text_w + 8, y + baseline + 8), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def get_lane_info(cx):
    for lane_index, speed_limit in enumerate(LANE_LIMITS_LEFT_TO_RIGHT):
        left_x = LANE_BOUNDARIES_X[lane_index]
        right_x = LANE_BOUNDARIES_X[lane_index + 1]
        if left_x <= cx < right_x:
            return f"Lane {lane_index + 1}", speed_limit

    return "Lane 6", LANE_LIMITS_LEFT_TO_RIGHT[-1]


def draw_warning_board(frame, violations, total_violations):
    panel_color = (10, 10, 12)
    border_color = (0, 80, 255)
    title_color = (0, 215, 255)
    text_color = (230, 230, 230)
    alert_color = (0, 0, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (356, 178), panel_color, -1)
    frame[:] = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
    cv2.rectangle(frame, (12, 12), (356, 178), border_color, 1)
    cv2.putText(frame, "BANG CANH BAO QUA TOC DO", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, title_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Tong vi pham: {total_violations}", (28, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

    recent = violations[-4:]
    if not recent:
        cv2.putText(frame, "Chua co xe vuot toc do", (28, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 255, 120), 1, cv2.LINE_AA)
        return

    for index, item in enumerate(recent):
        y = 94 + index * 20
        text = f"Xe {item['id']:>2} | {item['speed']:>3} km/h | GT {item['limit']}"
        cv2.putText(frame, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, alert_color, 1, cv2.LINE_AA)


def main():
    args = parse_args()
    preset = CAMERA_PRESETS[args.camera]

    input_path = Path(args.input or preset["input"])
    output_path = Path(args.output or preset["output"])
    weights_path = Path(args.weights)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Camera: {args.camera} ({preset['name']})")
    print(f"Using device: {device}")

    model = YOLO(str(weights_path))
    tracker = Tracker()
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    count = 0
    di_xuong = {}
    di_len = {}
    dem_xe_xuong = []
    dem_xe_len = []
    toc_do_da_do = {}
    lane_da_do = {}
    vi_pham = {}
    bang_canh_bao = []

    moc_do_y = preset["red_y"]
    moc_xanh_la_y = preset["green_y"]
    offset = 6
    distance_m = preset["distance_m"]

    if args.save_frames:
        os.makedirs("detected_frames", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (1020, 500))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame, device=device, verbose=False)
        boxes = results[0].boxes.data.detach().cpu().numpy()

        vehicle_boxes = []
        for row in boxes:
            x1, y1, x2, y2, _, class_id = row
            class_name = COCO_CLASSES[int(class_id)]
            if class_name in {"car", "bus", "truck", "motorcycle"}:
                vehicle_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        bbox_id = tracker.update(vehicle_boxes)

        for bbox in bbox_id:
            x3, y3, x4, y4, object_id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            if moc_do_y < cy + offset and moc_do_y > cy - offset:
                di_xuong[object_id] = time.time()

            if object_id in di_xuong and moc_xanh_la_y < cy + offset and moc_xanh_la_y > cy - offset:
                elapsed_time = time.time() - di_xuong[object_id]
                if object_id not in dem_xe_xuong and elapsed_time > 0:
                    dem_xe_xuong.append(object_id)
                    speed_kmh = (distance_m / elapsed_time) * 3.6
                    toc_do_da_do[object_id] = speed_kmh
                    lane_name, speed_limit = get_lane_info(cx)
                    lane_da_do[object_id] = (lane_name, speed_limit)
                    display_speed = int(speed_kmh)
                    if display_speed > speed_limit:
                        vi_pham[object_id] = True
                        bang_canh_bao.append({"id": object_id, "speed": display_speed, "limit": speed_limit})

            if moc_xanh_la_y < cy + offset and moc_xanh_la_y > cy - offset:
                di_len[object_id] = time.time()

            if object_id in di_len and moc_do_y < cy + offset and moc_do_y > cy - offset:
                elapsed_time = time.time() - di_len[object_id]
                if object_id not in dem_xe_len and elapsed_time > 0:
                    dem_xe_len.append(object_id)
                    speed_kmh = (distance_m / elapsed_time) * 3.6
                    toc_do_da_do[object_id] = speed_kmh
                    lane_name, speed_limit = get_lane_info(cx)
                    lane_da_do[object_id] = (lane_name, speed_limit)
                    display_speed = int(speed_kmh)
                    if display_speed > speed_limit:
                        vi_pham[object_id] = True
                        bang_canh_bao.append({"id": object_id, "speed": display_speed, "limit": speed_limit})

            if object_id in toc_do_da_do:
                speed_kmh = toc_do_da_do[object_id]
                _, speed_limit = lane_da_do.get(object_id, get_lane_info(cx))
                is_violation = vi_pham.get(object_id, False)
                blink_on = (count // 5) % 2 == 0
                box_color = (0, 0, 255) if is_violation and blink_on else (0, 255, 0)
                speed_color = (0, 0, 255) if is_violation else (0, 255, 255)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), box_color, 2)
                cv2.putText(frame, f"Xe {object_id}", (x3, y3 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{int(speed_kmh)}/{speed_limit}", (x3, y4 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, speed_color, 1, cv2.LINE_AA)

        red_color = (0, 0, 255)
        green_color = (0, 255, 0)
        lane_color = (210, 210, 210)

        draw_warning_board(frame, bang_canh_bao, len(bang_canh_bao))
        cv2.putText(frame, f"Xe xuong: {len(dem_xe_xuong)} | Xe len: {len(dem_xe_len)}", (28, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

        for guide_start, guide_end in preset["lane_guides"]:
            cv2.line(frame, guide_start, guide_end, lane_color, 1, cv2.LINE_AA)

        cv2.line(frame, preset["red_line"][0], preset["red_line"][1], red_color, 1)
        cv2.line(frame, preset["green_line"][0], preset["green_line"][1], green_color, 1)

        if args.save_frames:
            cv2.imwrite(f"detected_frames/frame_{count}.jpg", frame)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
