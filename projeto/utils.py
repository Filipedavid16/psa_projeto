import cv2
import math
import numpy as np
import re


def get_center(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def color_from_id(track_id):
    rng = np.random.default_rng(seed=track_id)
    return tuple(int(c) for c in rng.integers(60, 256, size=3))


def save_face_image(frame, bbox, save_path, margin=20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    x1m = max(0, x1 - margin)
    y1m = max(0, y1 - margin)
    x2m = min(w, x2 + margin)
    y2m = min(h, y2 + margin)

    face_crop = frame[y1m:y2m, x1m:x2m]

    if face_crop.size == 0:
        return False

    cv2.imwrite(str(save_path), face_crop)
    return True


def sanitize_folder_name(name):
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name