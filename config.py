import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Directories
IMAGE_ROOT = "known_faces"
EMBEDDING_DIR = "embeddings"
EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, "embeddings.pkl")

# Models
YOLO_MODEL_PATH = 'model/best.pt'

# Face Recognition Settings
SIMILARITY_THRESHOLD = 0.5

# Sensitive credentials (prefer setting these as environment variables or a .env file)
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("APP_PASSWORD", "")