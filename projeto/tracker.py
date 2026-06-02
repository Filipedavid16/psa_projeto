from config import PENDENTES_DIR, MAX_DISTANCE, MAX_LOST_FRAMES
from utils import distance, color_from_id


class TrackManager:
    def __init__(self):
        self.next_id = 1
        self.frame_count = 0
        self.tracks = {}

    def update(self, detected_faces):
        self.frame_count += 1

        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["lost"] += 1

        current_assignments = {}
        used_track_ids = set()
        used_detection_indices = set()

        all_pairs = []
        for det_idx, det in enumerate(detected_faces):
            for track_id, track_data in self.tracks.items():
                dist = distance(det["center"], track_data["center"])
                all_pairs.append((dist, det_idx, track_id))

        all_pairs.sort(key=lambda x: x[0])

        for dist, det_idx, track_id in all_pairs:
            if dist > MAX_DISTANCE:
                continue
            if det_idx in used_detection_indices:
                continue
            if track_id in used_track_ids:
                continue

            current_assignments[det_idx] = track_id
            used_detection_indices.add(det_idx)
            used_track_ids.add(track_id)

        for det_idx, track_id in current_assignments.items():
            det = detected_faces[det_idx]

            was_lost = self.tracks[track_id]["lost"] > 0

            self.tracks[track_id]["bbox"] = det["bbox"]
            self.tracks[track_id]["center"] = det["center"]
            self.tracks[track_id]["confidence"] = det["confidence"]
            self.tracks[track_id]["lost"] = 0

        for det_idx, det in enumerate(detected_faces):
            if det_idx in current_assignments:
                continue

            track_id = self.next_id
            self.next_id += 1

            folder = PENDENTES_DIR / f"ID_{track_id}"
            folder.mkdir(exist_ok=True)

            self.tracks[track_id] = {
                "bbox": det["bbox"],
                "center": det["center"],
                "confidence": det["confidence"],
                "lost": 0,
                "color": color_from_id(track_id),
                "saved_count": 0,
                "last_saved_frame": -999,
                "folder": folder,
                "pending_announced": False,
                "label": "Desconhecido",
                "registered": False,
                "last_recognition_frame": -999,
                "best_distance": None,
                "recognized_at": None,
                "greeting_mode": False,
                "just_reappeared": False
            }

            current_assignments[det_idx] = track_id
            print(f"Novo rosto detetado: ID {track_id}")

        ids_to_remove = []
        for track_id, track_data in self.tracks.items():
            if track_data["lost"] > MAX_LOST_FRAMES:
                ids_to_remove.append(track_id)

        for track_id in ids_to_remove:
            del self.tracks[track_id]

        return current_assignments