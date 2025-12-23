"""
Utilitaires généraux pour le projet
"""

import os
import json
import hashlib
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import cv2
from loguru import logger


def get_system_info() -> Dict[str, Any]:
    """Retourne les informations système"""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "hostname": platform.node()
    }
    
    # Raspberry Pi spécifique
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                if "Raspberry" in cpuinfo:
                    info["device"] = "Raspberry Pi"
                    # Extraire le modèle
                    for line in cpuinfo.split("\n"):
                        if line.startswith("Model"):
                            info["model"] = line.split(":")[1].strip()
                            break
        except:
            pass
    
    return info


def get_device_id() -> str:
    """Génère un ID unique pour l'appareil"""
    system_info = get_system_info()
    unique_string = f"{system_info['hostname']}-{system_info['platform']}-{system_info['architecture']}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:16]


def format_file_size(size_bytes: int) -> str:
    """Formate une taille en bytes en format lisible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def calculate_image_hash(image: np.ndarray) -> str:
    """Calcule un hash pour une image"""
    return hashlib.md5(image.tobytes()).hexdigest()


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: tuple,
    padding_color: tuple = (0, 0, 0)
) -> np.ndarray:
    """
    Redimensionne une image en conservant le ratio d'aspect
    avec padding si nécessaire
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculer le ratio
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    # Redimensionner
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Créer le canvas avec padding
    canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    
    # Centrer l'image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    position: tuple,
    font_scale: float = 0.6,
    font_thickness: int = 1,
    text_color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
    padding: int = 5
) -> np.ndarray:
    """Dessine du texte avec un fond coloré"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculer la taille du texte
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Dessiner le rectangle de fond
    x, y = position
    cv2.rectangle(
        image,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Dessiner le texte
    cv2.putText(image, text, position, font, font_scale, text_color, font_thickness)
    
    return image


def create_grid_visualization(
    images: List[np.ndarray],
    labels: List[str] = None,
    grid_size: tuple = None,
    cell_size: tuple = (200, 200),
    padding: int = 10
) -> np.ndarray:
    """Crée une grille de visualisation d'images"""
    n = len(images)
    if n == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    
    # Calculer la taille de la grille
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
    
    # Créer le canvas
    canvas_h = rows * (cell_size[1] + padding) + padding
    canvas_w = cols * (cell_size[0] + padding) + padding
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    
    # Placer les images
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        x = col * (cell_size[0] + padding) + padding
        y = row * (cell_size[1] + padding) + padding
        
        # Redimensionner l'image
        resized = resize_with_aspect_ratio(img, cell_size)
        
        # Placer dans le canvas
        canvas[y:y + cell_size[1], x:x + cell_size[0]] = resized
        
        # Ajouter le label
        if labels and i < len(labels):
            draw_text_with_background(
                canvas,
                labels[i][:30],
                (x + 5, y + cell_size[1] - 10),
                font_scale=0.4
            )
    
    return canvas


def load_json(path: Union[str, Path]) -> Dict:
    """Charge un fichier JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, path: Union[str, Path], indent: int = 2):
    """Sauvegarde des données en JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Crée un répertoire s'il n'existe pas"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Retourne un timestamp formaté"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_iso_timestamp() -> str:
    """Retourne un timestamp ISO"""
    return datetime.now().isoformat()


class Timer:
    """Context manager pour mesurer le temps d'exécution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.end = None
        self.elapsed = None
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end = datetime.now()
        self.elapsed = (self.end - self.start).total_seconds()
        logger.debug(f"{self.name}: {self.elapsed:.3f}s")


class MovingAverage:
    """Calcule une moyenne mobile"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> float:
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return self.get()
    
    def get(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self):
        self.values = []


class FPSCounter:
    """Compteur de FPS"""
    
    def __init__(self, avg_frames: int = 30):
        self.avg_frames = avg_frames
        self.times = []
        self.last_time = None
    
    def tick(self) -> float:
        import time
        current = time.perf_counter()
        
        if self.last_time is not None:
            self.times.append(current - self.last_time)
            if len(self.times) > self.avg_frames:
                self.times.pop(0)
        
        self.last_time = current
        return self.get_fps()
    
    def get_fps(self) -> float:
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


def validate_image(image: np.ndarray) -> bool:
    """Valide qu'une image est correcte"""
    if image is None:
        return False
    if not isinstance(image, np.ndarray):
        return False
    if len(image.shape) < 2:
        return False
    if image.size == 0:
        return False
    return True


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalise une image pour l'affichage"""
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Dénormaliser si nécessaire
        if image.min() < 0 or image.max() <= 1:
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
    return image