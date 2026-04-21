import cv2
import tkinter as tk
import time

from config import (
    MAIN_WINDOW,
    REGISTER_WINDOW,
    MAX_PHOTOS_PER_ID,
    SAVE_INTERVAL_FRAMES,
    RECOGNITION_INTERVAL_FRAMES,
)
from detector import FaceDetector
from tracker import TrackManager
from recognizer import FaceRegistry
from register_ui import RegisterUI
from utils import save_face_image
from audio_manager import AudioManager


def try_recognize_faces(frame, detected_faces, assignments, tracker, registry,audio):
    for det_idx, det in enumerate(detected_faces):
        track_id = assignments[det_idx]
        track_data = tracker.tracks[track_id]

        if track_data["registered"]:
            continue

        frames_since_recognition = tracker.frame_count - track_data["last_recognition_frame"]
        if frames_since_recognition < RECOGNITION_INTERVAL_FRAMES:
            continue

        name, best_distance = registry.recognize_face(frame, det["bbox"])

        track_data["last_recognition_frame"] = tracker.frame_count
        track_data["best_distance"] = best_distance

        if name is not None:
            track_data["label"] = name
            track_data["registered"] = True
            track_data["recognized_at"] = time.time()
            track_data["greeting_mode"] = True

            registry.cleanup_recognized_track(track_id, tracker.tracks)

            print(f"ID {track_id} reconhecido automaticamente como '{name}'.")
            audio.play_person_greeting(name)


def save_unknown_faces(frame, detected_faces, assignments, tracker):
    for det_idx, det in enumerate(detected_faces):
        track_id = assignments[det_idx]
        track_data = tracker.tracks[track_id]

        if track_data["registered"]:
            continue

        if track_data["saved_count"] < MAX_PHOTOS_PER_ID:
            frames_since_last_save = tracker.frame_count - track_data["last_saved_frame"]

            if frames_since_last_save >= SAVE_INTERVAL_FRAMES:
                photo_number = track_data["saved_count"] + 1
                save_path = track_data["folder"] / f"foto_{photo_number}.jpg"

                saved_ok = save_face_image(frame, det["bbox"], save_path, margin=20)

                if saved_ok:
                    track_data["saved_count"] += 1
                    track_data["last_saved_frame"] = tracker.frame_count

                    print(
                        f"ID {track_id}: foto "
                        f"{track_data['saved_count']}/{MAX_PHOTOS_PER_ID} guardada."
                    )

                    if (
                        track_data["saved_count"] == MAX_PHOTOS_PER_ID
                        and not track_data["pending_announced"]):
            
                        track_data["pending_announced"] = True
                        print(f"ID {track_id} pronto para registo.")


def draw_tracks(frame, detected_faces, assignments, tracker):
    for det_idx, det in enumerate(detected_faces):
        track_id = assignments[det_idx]
        track_data = tracker.tracks[track_id]

        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cor = track_data["color"]
        saved_count = track_data["saved_count"]
        label = track_data["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        texto_2 = None

        if track_data["registered"]:
            recognized_at = track_data.get("recognized_at")

            if track_data.get("greeting_mode") and recognized_at is not None:
                elapsed = time.time() - recognized_at

                if elapsed < 1.0:
                    texto_1 = f"{label}"
                    texto_2 = f"Conf: {conf:.2f} | Reconhecido"
                else:
                    texto_1 = f"Ola, {label}"
                    texto_2 = None
            else:
                texto_1 = f"{label}"
                texto_2 = f"Conf: {conf:.2f} | Reconhecido"
        else:
            texto_1 = f"{label} | ID {track_id}"
            texto_2 = f"Conf: {conf:.2f} | Fotos: {saved_count}/{MAX_PHOTOS_PER_ID}"

        text_y1 = y1 - 30 if y1 > 50 else y1 + 20
        text_y2 = y1 - 10 if y1 > 30 else y1 + 45

        cv2.putText(
            frame,
            texto_1,
            (x1, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            cor,
            2
        )

        if texto_2 is not None:
            cv2.putText(
                frame,
                texto_2,
                (x1, text_y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cor,
                2
            )


def main():
    detector = FaceDetector()
    tracker = TrackManager()
    registry = FaceRegistry()
    ui = RegisterUI()
    audio = AudioManager()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: nao foi possivel abrir a camara.")
        return

    registry.reload_known_faces_async()

    cv2.namedWindow(MAIN_WINDOW)

    print("Camara aberta.")
    print("  1 -> sair")
    print("  2 -> abrir/fechar janela de registo")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro ao ler imagem da camara.")
            break

        detected_faces = detector.detect(frame)
        assignments = tracker.update(detected_faces)

        try_recognize_faces(frame, detected_faces, assignments, tracker, registry, audio)
        save_unknown_faces(frame, detected_faces, assignments, tracker)
        draw_tracks(frame, detected_faces, assignments, tracker)

        if registry.db_loading:
            cv2.putText(frame, "A atualizar base de dados...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(MAIN_WINDOW, frame)

        if ui.is_open:
            cv2.namedWindow(REGISTER_WINDOW, cv2.WINDOW_NORMAL)

            cv2.setMouseCallback(
                REGISTER_WINDOW,
                ui.mouse_callback,
                {"registry": registry, "tracks": tracker.tracks}
            )

            ui.draw(registry)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            break
        elif key == ord("2"):
            ui.toggle()
        elif ui.is_open:
            ui.process_key(key, registry, tracker.tracks)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()