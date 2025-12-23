"""
Configuration globale du projet Drone Agri AI
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === CHEMINS ===
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Créer les dossiers s'ils n'existent pas
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# === MODÈLE ===
MODEL_CONFIG = {
    "input_size": (224, 224),
    "num_classes": 38,  # PlantVillage classes
    "backbone": "efficientnet_b0",  # Léger pour Raspberry
    "confidence_threshold": 0.7,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
}

# === CLASSES DE PLANTES ===
# Sera chargé dynamiquement depuis le dataset
PLANT_CLASSES = []

# === STADES DE CROISSANCE ===
GROWTH_STAGES = {
    "seedling": {"min_size_ratio": 0.0, "max_size_ratio": 0.2, "description": "Semis"},
    "vegetative": {"min_size_ratio": 0.2, "max_size_ratio": 0.5, "description": "Végétatif"},
    "flowering": {"min_size_ratio": 0.5, "max_size_ratio": 0.8, "description": "Floraison"},
    "mature": {"min_size_ratio": 0.8, "max_size_ratio": 1.0, "description": "Mature"},
}

# === FIREBASE ===
FIREBASE_CONFIG = {
    "credential_path": os.getenv("FIREBASE_CREDENTIAL_PATH", "firebase-key.json"),
    "database_url": os.getenv("FIREBASE_DATABASE_URL", ""),
    "storage_bucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
}

# === RASPBERRY PI ===
RASPBERRY_CONFIG = {
    "camera_resolution": (640, 480),
    "camera_framerate": 30,
    "inference_interval_ms": 100,  # 10 FPS pour l'inférence
    "offline_storage_path": "/home/pi/drone-agri-ai/offline_data",
    "max_offline_storage_mb": 500,
}

# === SERVEUR ===
SERVER_CONFIG = {
    "sync_interval_seconds": 30,
    "retry_attempts": 3,
    "timeout_seconds": 10,
    "use_compression": True,
}

# === SÉCURITÉ ===
SECURITY_CONFIG = {
    "enable_encryption": True,
    "api_key": os.getenv("API_KEY", ""),
    "use_https": True,
}