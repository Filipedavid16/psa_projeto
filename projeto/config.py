from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PROTO_PATH = BASE_DIR / "modelo" / "deploy.prototxt.txt"
MODEL_PATH = BASE_DIR / "modelo" / "res10_300x300_ssd_iter_140000.caffemodel"

PENDENTES_DIR = BASE_DIR / "pendentes"
FACE_DB_DIR = BASE_DIR / "face_db"

PENDENTES_DIR.mkdir(exist_ok=True)
FACE_DB_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5
MAX_DISTANCE = 160
MAX_LOST_FRAMES = 20
MAX_PHOTOS_PER_ID = 3
SAVE_INTERVAL_FRAMES = 15

RECOGNITION_INTERVAL_FRAMES = 8
RECOGNITION_TOLERANCE = 0.50

MAIN_WINDOW = "Detecao + Tracking + Registo"
REGISTER_WINDOW = "Registo"