#!/usr/bin/env python3
"""
Script principal pour le drone agricole
Exécution sur Raspberry Pi avec caméra et communication Pixhawk
"""

import os
import sys
import time
import signal
import argparse
import threading
from datetime import datetime
from pathlib import Path

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from loguru import logger

# Configuration du logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    "/var/log/drone-agri-ai/drone.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)

from src.config import RASPBERRY_CONFIG, MODEL_CONFIG
from src.plant_analyzer import PlantAnalyzer
from src.server_sync import FirebaseSync
from raspberry.camera_handler import CameraHandler
from raspberry.pixhawk_comm import PixhawkCommunicator


class DroneAgriAI:
    """
    Système principal d'IA pour drone agricole
    """
    
    def __init__(self, args):
        self.args = args
        self.running = False
        
        # Composants
        self.camera = None
        self.analyzer = None
        self.sync = None
        self.pixhawk = None
        
        # Stats
        self.stats = {
            "frames_processed": 0,
            "plants_detected": 0,
            "issues_found": 0,
            "start_time": None
        }
        
        # Thread de traitement
        self._process_thread = None
        
        # Initialiser les composants
        self._initialize()
    
    def _initialize(self):
        """Initialise tous les composants"""
        
        logger.info("=== Initialisation Drone Agri AI ===")
        
        # Analyseur de plantes
        logger.info("Chargement du modèle IA...")
        self.analyzer = PlantAnalyzer(
            model_path=self.args.model_path,
            use_coral=self.args.use_coral
        )
        
        # Caméra
        logger.info("Initialisation caméra...")
        self.camera = CameraHandler(
            resolution=RASPBERRY_CONFIG["camera_resolution"],
            framerate=RASPBERRY_CONFIG["camera_framerate"]
        )
        
        # Synchronisation Firebase
        logger.info("Initialisation synchronisation...")
        self.sync = FirebaseSync()
        self.sync.start_sync_worker()
        
        # Communication Pixhawk (optionnel)
        if self.args.enable_pixhawk:
            logger.info("Initialisation Pixhawk...")
            self.pixhawk = PixhawkCommunicator(port=self.args.pixhawk_port)
        
        logger.info("Initialisation terminée ✓")
    
    def start(self):
        """Démarre le système"""
        
        self.running = True
        self.stats["start_time"] = datetime.now()
        
        # Gérer les signaux
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Démarrer la caméra
        self.camera.start()
        
        # Boucle principale
        logger.info("=== Système démarré ===")
        logger.info(f"Mode: {'Coral TPU' if self.args.use_coral else 'CPU'}")
        logger.info(f"Résolution: {RASPBERRY_CONFIG['camera_resolution']}")
        logger.info(f"Intervalle inférence: {RASPBERRY_CONFIG['inference_interval_ms']}ms")
        
        self._main_loop()
    
    def _main_loop(self):
        """Boucle principale de traitement"""
        
        last_inference_time = 0
        inference_interval = RASPBERRY_CONFIG["inference_interval_ms"] / 1000.0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Capturer une frame
                frame = self.camera.get_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Vérifier l'intervalle d'inférence
                if current_time - last_inference_time >= inference_interval:
                    
                    # Analyser la frame
                    result, vis_frame = self.analyzer.analyze(
                        frame, 
                        return_visualization=True
                    )
                    
                    last_inference_time = current_time
                    self.stats["frames_processed"] += 1
                    
                    # Traiter le résultat
                    if result.is_plant:
                        self.stats["plants_detected"] += 1
                        
                        if result.health_status != "healthy":
                            self.stats["issues_found"] += 1
                            logger.warning(
                                f"Problème détecté: {result.plant_species} - "
                                f"{result.health_status} ({result.health_score:.0f}%)"
                            )
                        
                        # Envoyer au serveur
                        self.sync.queue_analysis(result.to_dict())
                        
                        # Envoyer au Pixhawk si nécessaire
                        if self.pixhawk and result.health_status == "critical":
                            self._notify_pixhawk(result)
                    
                    # Afficher si mode verbose
                    if self.args.verbose:
                        self._log_result(result)
                    
                    # Sauvegarder visualisation si demandé
                    if self.args.save_frames and vis_frame is not None:
                        self._save_visualization(vis_frame, result)
                
                # Limiter la charge CPU
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Erreur boucle principale: {e}")
                time.sleep(0.1)
        
        self._cleanup()
    
    def _notify_pixhawk(self, result):
        """Envoie une notification au Pixhawk"""
        if self.pixhawk:
            message = {
                "type": "PLANT_ISSUE",
                "severity": result.health_status,
                "species": result.plant_species,
                "recommendations": result.recommendations[:1]  # Premier conseil
            }
            self.pixhawk.send_message(message)
    
    def _log_result(self, result):
        """Affiche le résultat de l'analyse"""
        logger.info(
            f"Frame {self.stats['frames_processed']} | "
            f"Plante: {result.is_plant} | "
            f"Espèce: {result.plant_species[:20]} | "
            f"Santé: {result.health_score:.0f}% | "
            f"Temps: {result.inference_time_ms:.1f}ms"
        )
    
    def _save_visualization(self, frame, result):
        """Sauvegarde une frame annotée"""
        output_dir = Path("/home/pi/drone-agri-ai/captures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{result.analysis_id}.jpg"
        cv2.imwrite(str(output_dir / filename), frame)
    
    def _signal_handler(self, signum, frame):
        """Gère les signaux d'arrêt"""
        logger.info("Signal d'arrêt reçu...")
        self.running = False
    
    def _cleanup(self):
        """Nettoie les ressources"""
        logger.info("Arrêt du système...")
        
        if self.camera:
            self.camera.stop()
        
        if self.sync:
            self.sync.stop_sync_worker()
        
        if self.pixhawk:
            self.pixhawk.close()
        
        # Afficher les stats
        duration = (datetime.now() - self.stats["start_time"]).total_seconds()
        logger.info("=== Statistiques de session ===")
        logger.info(f"Durée: {duration:.1f} secondes")
        logger.info(f"Frames traitées: {self.stats['frames_processed']}")
        logger.info(f"Plantes détectées: {self.stats['plants_detected']}")
        logger.info(f"Problèmes trouvés: {self.stats['issues_found']}")
        logger.info(f"FPS moyen: {self.stats['frames_processed'] / max(duration, 1):.1f}")
        
        sync_status = self.sync.get_sync_status() if self.sync else {}
        logger.info(f"Items en attente sync: {sync_status.get('pending_offline', 0)}")
        
        logger.info("Arrêt terminé ✓")


def main():
    parser = argparse.ArgumentParser(description="Drone Agri AI - Analyse de plantes")
    
    parser.add_argument(
        "--model-path",
        default="/home/pi/drone-agri-ai/models/plant_model.tflite",
        help="Chemin vers le modèle TFLite"
    )
    parser.add_argument(
        "--use-coral",
        action="store_true",
        help="Utiliser l'accélérateur Coral Edge TPU"
    )
    parser.add_argument(
        "--enable-pixhawk",
        action="store_true",
        help="Activer la communication Pixhawk"
    )
    parser.add_argument(
        "--pixhawk-port",
        default="/dev/ttyAMA0",
        help="Port série du Pixhawk"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbose"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Sauvegarder les frames analysées"
    )
    
    args = parser.parse_args()
    
    # Créer et démarrer le système
    system = DroneAgriAI(args)
    system.start()


if __name__ == "__main__":
    main()