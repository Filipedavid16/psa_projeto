import cv2
import numpy as np
from pathlib import Path
import math
import shutil
import re
import threading
import face_recognition

# =========================================================
# CAMINHOS DOS FICHEIROS DO MODELO
# =========================================================
BASE_DIR = Path(_file_).resolve().parents[2]

proto_path = str(BASE_DIR / "modelo" / "deploy.prototxt.txt")
model_path = str(BASE_DIR / "modelo" / "res10_300x300_ssd_iter_140000.caffemodel")

# Pastas de trabalho
PENDENTES_DIR = BASE_DIR / "pendentes"
FACE_DB_DIR = BASE_DIR / "face_db"

PENDENTES_DIR.mkdir(exist_ok=True)
FACE_DB_DIR.mkdir(exist_ok=True)

# =========================================================
# CARREGAR O MODELO DNN DE DETEÇÃO FACIAL
# =========================================================
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# =========================================================
# PARÂMETROS
# =========================================================
CONFIDENCE_THRESHOLD = 0.5
MAX_DISTANCE = 160
MAX_LOST_FRAMES = 20
MAX_PHOTOS_PER_ID = 3
SAVE_INTERVAL_FRAMES = 15

# Reconhecimento
RECOGNITION_INTERVAL_FRAMES = 8
RECOGNITION_TOLERANCE = 0.50   # mais baixo = mais rigoroso

MAIN_WINDOW = "Deteção + Tracking + Registo"
REGISTER_WINDOW = "Registo"

# =========================================================
# ESTRUTURAS DE DADOS
# =========================================================
next_id = 1
frame_count = 0
tracks = {}

register_window_open = False
selected_pending_id = None
typed_name = ""
pending_click_areas = []  # [(x1, y1, x2, y2, track_id), ...]

# Base de dados em memória
known_face_encodings = []
known_face_names = []

# Controlo de acesso e carregamento da base de dados
db_lock = threading.Lock()
db_loading = False

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def get_center(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def color_from_id(track_id):
    np.random.seed(track_id)
    return tuple(int(c) for c in np.random.randint(60, 256, size=3))

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

def get_pending_ids():
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

def load_known_faces():
    global known_face_encodings, known_face_names, db_loading

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

        with db_lock:
            known_face_encodings = local_encodings
            known_face_names = local_names

        print(f"Base de dados carregada: {images_loaded} foto(s) válidas de {len(persons_loaded)} pessoa(s).")

    finally:
        db_loading = False

def reload_known_faces_async():
    global db_loading

    if db_loading:
        return

    db_loading = True
    threading.Thread(target=load_known_faces, daemon=True).start()

def recognize_face(frame, bbox, margin=25):
    """
    Tenta reconhecer uma face com base nas fotos já guardadas em face_db.
    Devolve (nome, distancia) ou (None, melhor_distancia).
    """
    with db_lock:
        encodings_db = known_face_encodings.copy()
        names_db = known_face_names.copy()

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

def register_pending_id(track_id, person_name):
    global selected_pending_id, typed_name

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

    # Atualizar logo o track atual
    if track_id in tracks:
        tracks[track_id]["label"] = person_name
        tracks[track_id]["registered"] = True
        tracks[track_id]["best_distance"] = 0.0

    # Recarregar a base de dados em background
    reload_known_faces_async()

    print(f"ID {track_id} registado como '{person_name}'.")
    print(f"{moved_count} fotos movidas para: {person_folder}")

    selected_pending_id = None
    typed_name = ""
    return True

def draw_register_window():
    global pending_click_areas

    width = 700
    height = 500
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)

    cv2.putText(canvas, "Janela de Registo", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, "Clica num ID pendente, escreve o nome e carrega Enter",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(canvas, "Teclas: Enter = registar | Backspace = apagar | Esc = limpar | r = fechar",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    pending_ids = get_pending_ids()
    pending_click_areas = []

    cv2.putText(canvas, "IDs pendentes:", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    start_y = 175
    row_h = 40

    if pending_ids:
        for idx, track_id in enumerate(pending_ids):
            y1 = start_y + idx * row_h
            y2 = y1 + 30
            x1 = 20
            x2 = 260

            if track_id == selected_pending_id:
                cv2.rectangle(canvas, (x1, y1 - 20), (x2, y2), (0, 180, 255), -1)
                text_color = (0, 0, 0)
            else:
                cv2.rectangle(canvas, (x1, y1 - 20), (x2, y2), (70, 70, 70), -1)
                text_color = (255, 255, 255)

            texto = f"ID {track_id}"
            cv2.putText(canvas, texto, (x1 + 10, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            pending_click_areas.append((x1, y1 - 20, x2, y2, track_id))
    else:
        cv2.putText(canvas, "Não há IDs pendentes com 3 fotos completas.",
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.putText(canvas, "ID selecionado:", (350, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    selected_text = f"ID {selected_pending_id}" if selected_pending_id is not None else "Nenhum"
    cv2.putText(canvas, selected_text, (350, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(canvas, "Nome:", (350, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(canvas, (350, 270), (660, 320), (255, 255, 255), 2)
    cv2.putText(canvas, typed_name, (360, 305),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(canvas, "Depois de escrever o nome, carrega Enter.",
                (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.putText(canvas, "Exemplo: Ana, Joao Silva, Maria",
                (350, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow(REGISTER_WINDOW, canvas)

def register_mouse_callback(event, x, y, flags, param):
    global selected_pending_id, typed_name

    if event == cv2.EVENT_LBUTTONDOWN:
        for x1, y1, x2, y2, track_id in pending_click_areas:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_pending_id = track_id
                typed_name = ""
                break

def process_register_key(key):
    global typed_name, selected_pending_id

    if key == -1:
        return

    if key == 13:  # Enter
        if selected_pending_id is not None and typed_name.strip():
            register_pending_id(selected_pending_id, typed_name.strip())
        return

    if key in (8, 127):  # Backspace
        typed_name = typed_name[:-1]
        return

    if key == 27:  # ESC
        typed_name = ""
        return

    if key in (9, 10):
        return

    if 32 <= key <= 126:
        typed_name += chr(key)

# =========================================================
# ABRIR A WEBCAM
# =========================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: não foi possível abrir a câmara.")
    exit()

# Carregar a base de dados ao arrancar
reload_known_faces_async()

cv2.namedWindow(MAIN_WINDOW)
print("Câmara aberta.")
print("Teclas:")
print("  q -> sair")
print("  r -> abrir/fechar janela de registo")

# =========================================================
# CICLO PRINCIPAL
# =========================================================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao ler imagem da câmara.")
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # -----------------------------------------------------
    # 1) DETEÇÃO DE CARAS
    # -----------------------------------------------------
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    detected_faces = []

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])

        if conf < CONFIDENCE_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        center = get_center(x1, y1, x2, y2)

        detected_faces.append({
            "bbox": (x1, y1, x2, y2),
            "center": center,
            "confidence": conf
        })

    # -----------------------------------------------------
    # 2) MARCAR TODOS OS TRACKS COMO "PERDIDOS"
    # -----------------------------------------------------
    for track_id in list(tracks.keys()):
        tracks[track_id]["lost"] += 1

    # -----------------------------------------------------
    # 3) ASSOCIAR DETEÇÕES A IDs EXISTENTES
    # -----------------------------------------------------
    current_assignments = {}
    used_track_ids = set()
    used_detection_indices = set()

    all_pairs = []
    for det_idx, det in enumerate(detected_faces):
        for track_id, track_data in tracks.items():
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

    # -----------------------------------------------------
    # 4) ATUALIZAR TRACKS EXISTENTES
    # -----------------------------------------------------
    for det_idx, track_id in current_assignments.items():
        det = detected_faces[det_idx]
        tracks[track_id]["bbox"] = det["bbox"]
        tracks[track_id]["center"] = det["center"]
        tracks[track_id]["confidence"] = det["confidence"]
        tracks[track_id]["lost"] = 0

    # -----------------------------------------------------
    # 5) CRIAR NOVOS IDs
    # -----------------------------------------------------
    for det_idx, det in enumerate(detected_faces):
        if det_idx in current_assignments:
            continue

        track_id = next_id
        next_id += 1

        folder = PENDENTES_DIR / f"ID_{track_id}"
        folder.mkdir(exist_ok=True)

        tracks[track_id] = {
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
            "best_distance": None
        }

        current_assignments[det_idx] = track_id
        print(f"Novo rosto detetado: ID {track_id}")

    # -----------------------------------------------------
    # 6) APAGAR TRACKS DESAPARECIDOS
    # -----------------------------------------------------
    ids_to_remove = []
    for track_id, track_data in tracks.items():
        if track_data["lost"] > MAX_LOST_FRAMES:
            ids_to_remove.append(track_id)

    for track_id in ids_to_remove:
        del tracks[track_id]

    # -----------------------------------------------------
    # 7) TENTAR RECONHECER CADA CARA
    # -----------------------------------------------------
    for det_idx, det in enumerate(detected_faces):
        track_id = current_assignments[det_idx]
        track_data = tracks[track_id]

        if track_data["registered"]:
            continue

        frames_since_recognition = frame_count - track_data["last_recognition_frame"]
        if frames_since_recognition < RECOGNITION_INTERVAL_FRAMES:
            continue

        name, best_distance = recognize_face(frame, det["bbox"])

        track_data["last_recognition_frame"] = frame_count
        track_data["best_distance"] = best_distance

        if name is not None:
            track_data["label"] = name
            track_data["registered"] = True

            # Se a pasta pendente existir e ainda estiver vazia, tenta limpá-la
            try:
                if track_data["folder"].exists() and not any(track_data["folder"].iterdir()):
                    track_data["folder"].rmdir()
            except OSError:
                pass

            print(f"ID {track_id} reconhecido automaticamente como '{name}'.")

    # -----------------------------------------------------
    # 8) GUARDAR AUTOMATICAMENTE ATÉ 3 FOTOS POR ID DESCONHECIDO
    # -----------------------------------------------------
    for det_idx, det in enumerate(detected_faces):
        track_id = current_assignments[det_idx]
        track_data = tracks[track_id]

        if track_data["registered"]:
            continue

        if track_data["saved_count"] < MAX_PHOTOS_PER_ID:
            frames_since_last_save = frame_count - track_data["last_saved_frame"]

            if frames_since_last_save >= SAVE_INTERVAL_FRAMES:
                photo_number = track_data["saved_count"] + 1
                save_path = track_data["folder"] / f"foto_{photo_number}.jpg"

                saved_ok = save_face_image(frame, det["bbox"], save_path, margin=20)

                if saved_ok:
                    track_data["saved_count"] += 1
                    track_data["last_saved_frame"] = frame_count

                    print(f"ID {track_id}: foto {track_data['saved_count']}/{MAX_PHOTOS_PER_ID} guardada.")

                    if track_data["saved_count"] == MAX_PHOTOS_PER_ID and not track_data["pending_announced"]:
                        track_data["pending_announced"] = True
                        print(f"ID {track_id} pronto para registo.")

    # -----------------------------------------------------
    # 9) DESENHAR CAIXAS E TEXTO
    # -----------------------------------------------------
    for det_idx, det in enumerate(detected_faces):
        track_id = current_assignments[det_idx]
        track_data = tracks[track_id]

        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cor = track_data["color"]
        saved_count = track_data["saved_count"]
        label = track_data["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        texto_1 = f"{label} | ID {track_id}"

        if track_data["registered"]:
            if track_data["best_distance"] is not None:
                texto_2 = f"Conf: {conf:.2f} | Dist: {track_data['best_distance']:.3f}"
            else:
                texto_2 = f"Conf: {conf:.2f} | Reconhecido"
        else:
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

        cv2.putText(
            frame,
            texto_2,
            (x1, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cor,
            2
        )

    # -----------------------------------------------------
    # 10) AJUDA VISUAL
    # -----------------------------------------------------
    cv2.putText(frame, "q = sair | r = registo", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if db_loading:
        cv2.putText(frame, "A atualizar base de dados...", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # -----------------------------------------------------
    # 11) MOSTRAR IMAGEM PRINCIPAL
    # -----------------------------------------------------
    cv2.imshow(MAIN_WINDOW, frame)

    if register_window_open:
        cv2.namedWindow(REGISTER_WINDOW)
        cv2.setMouseCallback(REGISTER_WINDOW, register_mouse_callback)
        draw_register_window()

    # -----------------------------------------------------
    # 12) TECLADO
    # -----------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        register_window_open = not register_window_open
        if not register_window_open:
            try:
                cv2.destroyWindow(REGISTER_WINDOW)
            except cv2.error:
                pass

    elif register_window_open:
        process_register_key(key)

# =========================================================
# FECHAR TUDO
# =========================================================
cap.release()
cv2.destroyAllWindows();