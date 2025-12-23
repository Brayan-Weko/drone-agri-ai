"""
Gestionnaire de caméra pour Raspberry Pi
Support: PiCamera2, USB Camera, CSI Camera
"""

import threading
import time
from typing import Tuple, Optional
from collections import deque

import numpy as np
import cv2
from loguru import logger

# Essayer d'importer picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    logger.warning("picamera2 non disponible")


class CameraHandler:
    """
    Gestionnaire de caméra thread-safe avec buffer
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        framerate: int = 30,
        camera_type: str = "auto"  # "auto", "picamera", "usb"
    ):
        self.resolution = resolution
        self.framerate = framerate
        self.camera_type = camera_type
        
        self.camera = None
        self.running = False
        
        # Buffer de frames thread-safe
        self.frame_buffer = deque(maxlen=3)
        self._lock = threading.Lock()
        
        # Thread de capture
        self._capture_thread = None
        
        self._detect_camera()
    
    def _detect_camera(self):
        """Détecte et initialise la caméra disponible"""
        
        if self.camera_type == "auto":
            # Essayer PiCamera d'abord
            if PICAMERA2_AVAILABLE:
                try:
                    self._init_picamera()
                    self.camera_type = "picamera"
                    logger.info("Caméra Pi détectée")
                    return
                except Exception as e:
                    logger.warning(f"PiCamera non disponible: {e}")
            
            # Fallback vers USB
            try:
                self._init_usb_camera()
                self.camera_type = "usb"
                logger.info("Caméra USB détectée")
                return
            except Exception as e:
                logger.warning(f"Caméra USB non disponible: {e}")
            
            raise RuntimeError("Aucune caméra disponible")
        
        elif self.camera_type == "picamera":
            self._init_picamera()
        else:
            self._init_usb_camera()
    
    def _init_picamera(self):
        """Initialise la PiCamera2"""
        self.camera = Picamera2()
        
        config = self.camera.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        self.camera.configure(config)
    
    def _init_usb_camera(self):
        """Initialise une caméra USB"""
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            raise RuntimeError("Impossible d'ouvrir la caméra USB")
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, self.framerate)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def start(self):
        """Démarre la capture"""
        if self.running:
            return
        
        self.running = True
        
        if self.camera_type == "picamera":
            self.camera.start()
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        logger.info("Capture démarrée")
    
    def stop(self):
        """Arrête la capture"""
        self.running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        
        if self.camera_type == "picamera":
            self.camera.stop()
            self.camera.close()
        elif self.camera_type == "usb":
            self.camera.release()
        
        logger.info("Capture arrêtée")
    
    def _capture_loop(self):
        """Boucle de capture dans un thread séparé"""
        while self.running:
            try:
                if self.camera_type == "picamera":
                    frame = self.camera.capture_array()
                    # Convertir RGB -> BGR pour OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        continue
                
                with self._lock:
                    self.frame_buffer.append(frame)
                
            except Exception as e:
                logger.error(f"Erreur capture: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Récupère la dernière frame capturée"""
        with self._lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
        return None
    
    def get_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """Récupère la frame avec son timestamp"""
        frame = self.get_frame()
        return frame, time.time()
    
    def capture_still(self) -> Optional[np.ndarray]:
        """Capture une image haute résolution"""
        if self.camera_type == "picamera":
            # Capturer en pleine résolution
            return self.camera.capture_array()
        else:
            # Prendre la frame courante
            return self.get_frame()


class SimulatedCamera:
    """
    Caméra simulée pour les tests sans matériel
    """
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.resolution = resolution
        self.frame_count = 0
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def get_frame(self) -> np.ndarray:
        """Génère une frame de test"""
        # Créer une image avec du bruit
        frame = np.random.randint(0, 255, (*self.resolution[::-1], 3), dtype=np.uint8)
        
        # Ajouter un cercle vert (simule une plante)
        center = (self.resolution[0] // 2, self.resolution[1] // 2)
        cv2.circle(frame, center, 100, (0, 200, 0), -1)
        
        self.frame_count += 1
        return frame