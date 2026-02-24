import os, cv2, json, time, numpy as np
import asyncio, edge_tts, tempfile
from threading import Thread
from queue import Queue

import pygame
pygame.mixer.init()

from insightface.app import FaceAnalysis

print("A correr em:", os.getcwd())

# =========================
# CONFIG
# =========================
DATA_DIR = "face_db"
EMB_PATH = os.path.join(DATA_DIR, "embeddings.json")
os.makedirs(DATA_DIR, exist_ok=True)

WINDOW_NAME = "Vigilancia - Reconhecimento"

buttons = {
    "add":   (10, 50, 170, 95),
    "train": (180, 50, 340, 95),
    "quit":  (350, 50, 470, 95),
}

# Webcam
CAM_W, CAM_H = 640, 360

# InsightFace
DET_SIZE = (320, 320)            # melhor deteção/embedding (se ficar lento usa 224 ou 160)
DETECT_EVERY = 4                 # 4..8 (mais baixo = mais rápido a atualizar)
DET_SCORE_MIN = 0.45             # rejeita deteções fracas

# Reconhecimento
SAMPLES_PER_PERSON = 20          # mais amostras = melhor
MAX_EMBS_PER_PERSON = 20
SIM_THRESHOLD = 0.38             # começa aqui (0.34..0.42)
GAP_MIN = 0.04                   # 0.03..0.08 (maior = menos confusões)
STREAK_TO_ACCEPT = 3             # 2..4

FACE_MIN_SIZE = 60               # px

# Low-light (tecla L)
LOW_LIGHT_FIX = True
BRIGHTNESS = 18
CONTRAST = 1.20
GAMMA = 1.35
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# TTS
VOICE = "pt-PT-RaquelNeural"

# View / follow (zoom + seguimento)
VIEW_W, VIEW_H = 640, 360
FOLLOW_MARGIN = 1.9              # maior = menos zoom
SMOOTH_POS = 0.10
SMOOTH_ZOOM = 0.08
DEADZONE = 6

# =========================
# TTS (não bloquear)
# =========================
def speak(text: str):
    async def _run():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            path = f.name
        try:
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(path)
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        finally:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            if os.path.exists(path):
                os.remove(path)

    asyncio.run(_run())

tts_q = Queue()

def tts_worker():
    while True:
        text = tts_q.get()
        if text is None:
            break
        try:
            speak(text)
        except Exception as e:
            print("Erro no TTS:", e)
        finally:
            tts_q.task_done()

Thread(target=tts_worker, daemon=True).start()

# =========================
# Low light enhancement
# =========================
def apply_gamma(bgr, gamma=1.3):
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.arange(256) / 255.0) ** inv * 255.0
    return cv2.LUT(bgr, table.astype(np.uint8))

def enhance_low_light(bgr):
    out = cv2.convertScaleAbs(bgr, alpha=CONTRAST, beta=BRIGHTNESS)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    out = apply_gamma(out, GAMMA)
    return out

# =========================
# DB (múltiplos embeddings)
# =========================
def load_db():
    if not os.path.exists(EMB_PATH):
        return []
    try:
        with open(EMB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        people = []
        for p in data.get("people", []):
            embs = []
            if "embs" in p:
                for e in p["embs"]:
                    arr = np.array(e, dtype=np.float32)
                    arr = arr / (np.linalg.norm(arr) + 1e-8)
                    embs.append(arr)
            elif "emb" in p:  # compat antigo
                arr = np.array(p["emb"], dtype=np.float32)
                arr = arr / (np.linalg.norm(arr) + 1e-8)
                embs.append(arr)

            if embs:
                people.append({"name": p["name"], "embs": embs})

        return people
    except Exception as e:
        print("Erro a ler DB:", e)
        return []

def save_db(people):
    data = {
        "people": [
            {"name": p["name"], "embs": [e.astype(float).tolist() for e in p["embs"]]}
            for p in people
        ]
    }
    with open(EMB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return 1.0 - float(np.dot(a, b) / denom)

def identify(emb, people):
    """Best-of por pessoa + gap test."""
    if emb is None or not people:
        return "Unknown", 999.0

    scores = []
    for p in people:
        best_d = 999.0
        for e in p["embs"]:
            d = cosine_distance(emb, e)
            if d < best_d:
                best_d = d
        scores.append((best_d, p["name"]))

    scores.sort(key=lambda x: x[0])
    best_dist, best_name = scores[0]
    second_dist = scores[1][0] if len(scores) > 1 else 999.0

    if best_dist <= SIM_THRESHOLD and (second_dist - best_dist) >= GAP_MIN:
        return best_name, best_dist
    return "Unknown", best_dist

# =========================
# UI
# =========================
def inside(rect, x, y):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_button(frame, rect, text):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, text, (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_ui(frame, ui_message, pending_name, mode, captured, total):
    draw_button(frame, buttons["add"], "Adicionar")
    draw_button(frame, buttons["train"], "Recarregar")
    draw_button(frame, buttons["quit"], "Sair")

    if ui_message:
        cv2.putText(frame, ui_message, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if mode == "add_person":
        cv2.putText(frame, "Escreve o nome e ENTER (ESC cancela):", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Nome: {pending_name}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    if mode == "capturing":
        cv2.putText(frame, f"Captura: {captured}/{total}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

clicked_action = None
def mouse_callback(event, x, y, flags, param):
    global clicked_action
    if event == cv2.EVENT_LBUTTONDOWN:
        if inside(buttons["add"], x, y):
            clicked_action = "add"
        elif inside(buttons["train"], x, y):
            clicked_action = "train"
        elif inside(buttons["quit"], x, y):
            clicked_action = "quit"

# =========================
# Follow/Zoom (suave)
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def compute_follow_target(frame, bbox):
    H, W = frame.shape[:2]
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    cx = x1 + fw / 2.0
    cy = y1 + fh / 2.0

    crop_w = fw * FOLLOW_MARGIN
    crop_h = fh * FOLLOW_MARGIN

    target_aspect = VIEW_W / VIEW_H
    cur_aspect = crop_w / crop_h
    if cur_aspect > target_aspect:
        crop_h = crop_w / target_aspect
    else:
        crop_w = crop_h * target_aspect

    crop_w = clamp(crop_w, W * 0.35, W * 1.0)
    crop_h = clamp(crop_h, H * 0.35, H * 1.0)

    return (cx, cy, crop_w, crop_h)

def make_view_follow(frame, bbox, state):
    H, W = frame.shape[:2]
    target = compute_follow_target(frame, bbox)
    if target is None:
        return cv2.resize(frame, (VIEW_W, VIEW_H))

    cx, cy, tw, th = target
    if state["cx"] is None:
        state["cx"], state["cy"], state["w"], state["h"] = cx, cy, tw, th
    else:
        if abs(cx - state["cx"]) > DEADZONE:
            state["cx"] += (cx - state["cx"]) * SMOOTH_POS
        if abs(cy - state["cy"]) > DEADZONE:
            state["cy"] += (cy - state["cy"]) * SMOOTH_POS
        state["w"] += (tw - state["w"]) * SMOOTH_ZOOM
        state["h"] += (th - state["h"]) * SMOOTH_ZOOM

    x1 = int(state["cx"] - state["w"] / 2)
    y1 = int(state["cy"] - state["h"] / 2)
    x2 = int(x1 + state["w"])
    y2 = int(y1 + state["h"])

    x1 = clamp(x1, 0, W - 2)
    y1 = clamp(y1, 0, H - 2)
    x2 = clamp(x2, x1 + 1, W)
    y2 = clamp(y2, y1 + 1, H)

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (VIEW_W, VIEW_H))

# =========================
# InsightFace init
# =========================
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=DET_SIZE)

# =========================
# MAIN
# =========================
people = load_db()
print("Pessoas na DB:", [p["name"] for p in people])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Não consegui abrir a webcam.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

mode = "recognize"
ui_message = ""
pending_name = ""

# captura
capture_embs = []
current_name = None
captured = 0

# estado tracking
frame_idx = 0
last_bbox = None
last_best = ("Unknown", 999.0)
follow_state = {"cx": None, "cy": None, "w": None, "h": None}

# estabilização
streak_name = None
streak_count = 0
stable_name = "Unknown"
stable_dist = 999.0

already_greeted = set()

# FPS
fps_t0 = time.time()
fps_frames = 0
fps_val = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    fps_frames += 1
    dt = time.time() - fps_t0
    if dt >= 1.0:
        fps_val = fps_frames / dt
        fps_frames = 0
        fps_t0 = time.time()

    # Tecla
    key = cv2.waitKey(1) & 0xFF

    # Toggle low-light
    if key in (ord('l'), ord('L')):
        LOW_LIGHT_FIX = not LOW_LIGHT_FIX
        ui_message = f"Low-light: {'ON' if LOW_LIGHT_FIX else 'OFF'}"

    # garantir callback (evita perder cliques)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # Botões
    if clicked_action == "quit":
        break
    elif clicked_action == "train":
        people = load_db()
        ui_message = f"DB recarregada ({len(people)} pessoas)."
        clicked_action = None
    elif clicked_action == "add":
        mode = "add_person"
        pending_name = ""
        ui_message = "Adicionar pessoa: escreve o nome."
        clicked_action = None

    # Input nome
    if mode == "add_person":
        if key == 27:  # ESC
            mode = "recognize"
            ui_message = "Cancelado."
        elif key in (10, 13):  # ENTER
            nm = pending_name.strip()
            if not nm:
                ui_message = "Nome vazio."
            else:
                current_name = nm
                capture_embs = []
                captured = 0
                mode = "capturing"
                ui_message = f"A capturar {SAMPLES_PER_PERSON} amostras de {nm}..."
        elif key in (8, 127):  # backspace
            pending_name = pending_name[:-1]
        elif 32 <= key <= 126:
            pending_name += chr(key)

    # Frame para reconhecimento (melhorado)
    frame_rec = enhance_low_light(frame) if LOW_LIGHT_FIX else frame

    display_text = "Sem face"

    # =========================
    # DETEÇÃO + EMBEDDING (uma vez só)
    # =========================
    if mode in ("recognize", "capturing"):
        frame_idx += 1
        if frame_idx % DETECT_EVERY == 0 or last_bbox is None:
            faces = app.get(frame_rec)  # <- embedding vem aqui

            # escolhe a face maior com score decente
            best_face = None
            if faces:
                faces_sorted = sorted(
                    faces,
                    key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                    reverse=True
                )
                for f in faces_sorted:
                    if getattr(f, "det_score", 1.0) >= DET_SCORE_MIN:
                        best_face = f
                        break

            if best_face is None:
                last_bbox = None
                last_best = ("Unknown", 999.0)
                emb_now = None
            else:
                x1, y1, x2, y2 = map(int, best_face.bbox)
                if (x2 - x1) < FACE_MIN_SIZE or (y2 - y1) < FACE_MIN_SIZE:
                    last_bbox = None
                    last_best = ("Unknown", 999.0)
                    emb_now = None
                else:
                    last_bbox = (x1, y1, x2, y2)
                    emb_now = best_face.embedding.astype(np.float32)
                    emb_now = emb_now / (np.linalg.norm(emb_now) + 1e-8)

        else:
            # sem atualização neste frame
            emb_now = None

        # =========================
        # RECOGNIZE / CAPTURE
        # =========================
        if mode == "recognize":
            # usa o último embedding válido (se emb_now é None, mantém o último resultado)
            if emb_now is not None:
                best_name, best_dist = identify(emb_now, people)
                last_best = (best_name, best_dist)

            best_name, best_dist = last_best

            # estabilização
            if best_name == streak_name:
                streak_count += 1
            else:
                streak_name = best_name
                streak_count = 1

            if streak_count >= STREAK_TO_ACCEPT:
                stable_name = best_name
                stable_dist = best_dist

            if stable_name != "Unknown":
                display_text = f"Ola, {stable_name} (d={stable_dist:.2f})"
                if stable_name not in already_greeted:
                    tts_q.put(f"Olá, {stable_name}")
                    already_greeted.add(stable_name)
            else:
                display_text = f"Desconhecido (d={best_dist:.2f})"

        elif mode == "capturing":
            display_text = "A capturar..."
            # só adiciona embedding quando existe
            if emb_now is not None:
                capture_embs.append(emb_now)
                captured = len(capture_embs)

            if captured >= SAMPLES_PER_PERSON:
                # limita e guarda
                embs = capture_embs[:MAX_EMBS_PER_PERSON]
                people = [p for p in people if p["name"].lower() != current_name.lower()]
                people.append({"name": current_name, "embs": embs})
                save_db(people)

                ui_message = f"{current_name} guardado! ({len(people)} pessoas na DB)"
                mode = "recognize"
                current_name = None
                capture_embs = []
                captured = 0
                clicked_action = None

    # =========================
    # RENDER FINAL (view -> overlays -> UI)
    # =========================
    view_frame = make_view_follow(frame, last_bbox if mode != "add_person" else None, follow_state)

    cv2.putText(view_frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(view_frame, f"FPS: {fps_val:.1f}", (VIEW_W - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(view_frame, f"Low-light: {'ON' if LOW_LIGHT_FIX else 'OFF'} (L)", (10, VIEW_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    draw_ui(view_frame, ui_message, pending_name, mode, captured, SAMPLES_PER_PERSON)
    cv2.imshow(WINDOW_NAME, view_frame)

    if key in (ord('q'), ord('Q')):
        break

cap.release()
tts_q.put(None)
cv2.destroyAllWindows()
