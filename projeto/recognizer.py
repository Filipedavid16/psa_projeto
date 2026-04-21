import cv2
import numpy as np
import shutil
import threading
import face_recognition
import time

from config import FACE_DB_DIR, PENDENTES_DIR, MAX_PHOTOS_PER_ID, RECOGNITION_TOLERANCE
from utils import sanitize_folder_name
from audio_manager import ensure_greeting_audio_file


class FaceRegistry:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.db_lock = threading.Lock()
        self.db_loading = False

    def get_pending_ids(self):
        pending_ids = []

        for folder in PENDENTES_DIR.glob("ID_*"):
            if not folder.is_dir():
                continue

            try:
                track_id = int(folder.name.replace("ID_", ""))
            except ValueError:
                continue

            photos = list(folder.glob("*.jpg"))
            if len(photos) >= MAX_PHOTOS_PER_ID:
                pending_ids.append(track_id)

        return sorted(pending_ids)

    def load_known_faces(self):
        local_encodings = []
        local_names = []

        valid_exts = {".jpg", ".jpeg", ".png"}
        persons_loaded = set()
        images_loaded = 0

        try:
            for person_folder in FACE_DB_DIR.iterdir():
                if not person_folder.is_dir():
                    continue

                person_name = person_folder.name

                for image_path in person_folder.iterdir():
                    if not image_path.is_file():
                        continue
                    if image_path.suffix.lower() not in valid_exts:
                        continue

                    try:
                        image = face_recognition.load_image_file(str(image_path))
                        encodings = face_recognition.face_encodings(image)

                        if not encodings:
                            print(f"Aviso: sem rosto válido em {image_path.name}")
                            continue

                        local_encodings.append(encodings[0])
                        local_names.append(person_name)
                        persons_loaded.add(person_name)
                        images_loaded += 1

                    except Exception as e:
                        print(f"Erro ao carregar {image_path}: {e}")

            with self.db_lock:
                self.known_face_encodings = local_encodings
                self.known_face_names = local_names

            print(
                f"Base de dados carregada: {images_loaded} foto(s) válidas "
                f"de {len(persons_loaded)} pessoa(s)."
            )

        finally:
            self.db_loading = False

    def reload_known_faces_async(self):
        if self.db_loading:
            return

        self.db_loading = True
        threading.Thread(target=self.load_known_faces, daemon=True).start()

    def recognize_face(self, frame, bbox, margin=25):
        with self.db_lock:
            encodings_db = self.known_face_encodings.copy()
            names_db = self.known_face_names.copy()

        if not encodings_db:
            return None, None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        x1m = max(0, x1 - margin)
        y1m = max(0, y1 - margin)
        x2m = min(w, x2 + margin)
        y2m = min(h, y2 + margin)

        face_crop = frame[y1m:y2m, x1m:x2m]
        if face_crop.size == 0:
            return None, None

        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb_crop)
        if not encodings:
            return None, None

        face_encoding = encodings[0]

        distances = face_recognition.face_distance(encodings_db, face_encoding)
        if len(distances) == 0:
            return None, None

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        if best_distance <= RECOGNITION_TOLERANCE:
            return names_db[best_idx], best_distance

        return None, best_distance

    def register_pending_id(self, track_id, person_name, tracks):
        safe_name = sanitize_folder_name(person_name)
        if not safe_name:
            print("Nome inválido.")
            return False

        pending_folder = PENDENTES_DIR / f"ID_{track_id}"

        if not pending_folder.exists() or not pending_folder.is_dir():
            print(f"Não existe pasta pendente para o ID {track_id}.")
            return False

        photos = sorted(pending_folder.glob("*.jpg"))

        if len(photos) < MAX_PHOTOS_PER_ID:
            print(f"O ID {track_id} ainda não tem {MAX_PHOTOS_PER_ID} fotos guardadas.")
            return False

        person_folder = FACE_DB_DIR / safe_name
        person_folder.mkdir(exist_ok=True)

        existing_photos = sorted(person_folder.glob("*.jpg"))
        next_index = len(existing_photos) + 1

        moved_count = 0
        for photo in photos:
            while True:
                dest_name = f"{safe_name}_{next_index:03d}.jpg"
                dest_path = person_folder / dest_name
                if not dest_path.exists():
                    break
                next_index += 1

            shutil.move(str(photo), str(dest_path))
            moved_count += 1
            next_index += 1

        try:
            pending_folder.rmdir()
        except OSError:
            pass

        if track_id in tracks:
            tracks[track_id]["label"] = person_name
            tracks[track_id]["registered"] = True
            tracks[track_id]["best_distance"] = 0.0
            tracks[track_id]["recognized_at"] = time.time()
            tracks[track_id]["greeting_mode"] = True

        self.reload_known_faces_async()
        ensure_greeting_audio_file(person_name)

        print(f"ID {track_id} registado como '{person_name}'.")
        print(f"{moved_count} fotos movidas para: {person_folder}")
        return True
    
    def delete_pending_id(self, track_id, tracks):
        pending_folder = PENDENTES_DIR / f"ID_{track_id}"

        if pending_folder.exists() and pending_folder.is_dir():
            shutil.rmtree(pending_folder)

        # Se o track ainda existir em memória, reinicia-o
        if track_id in tracks:
            new_folder = PENDENTES_DIR / f"ID_{track_id}"
            new_folder.mkdir(exist_ok=True)

            tracks[track_id]["folder"] = new_folder
            tracks[track_id]["saved_count"] = 0
            tracks[track_id]["last_saved_frame"] = -999
            tracks[track_id]["pending_announced"] = False
            tracks[track_id]["label"] = "Desconhecido"
            tracks[track_id]["registered"] = False
            tracks[track_id]["best_distance"] = None
            tracks[track_id]["last_recognition_frame"] = -999

        print(f"Registo pendente do ID {track_id} eliminado.")
        return True
    def cleanup_recognized_track(self, track_id, tracks):
        if track_id not in tracks:
            return

        track_data = tracks[track_id]
        pending_folder = track_data.get("folder")

        try:
            if pending_folder is not None and pending_folder.exists():
                shutil.rmtree(pending_folder)
        except Exception as e:
            print(f"Erro ao limpar pendentes do ID {track_id}: {e}")

        track_data["saved_count"] = 0
        track_data["last_saved_frame"] = -999
        track_data["pending_announced"] = False