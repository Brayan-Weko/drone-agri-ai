"""
Module de synchronisation avec le serveur Firebase
Gestion du mode hors-ligne et reprise de connexion
"""

import os
import json
import asyncio
import aiohttp
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import threading
import queue
import time

from loguru import logger
from dotenv import load_dotenv

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("Firebase non disponible - mode offline uniquement")

from .config import FIREBASE_CONFIG, SERVER_CONFIG, SECURITY_CONFIG
from .offline_storage import OfflineStorage

load_dotenv()


@dataclass
class SyncItem:
    """Item à synchroniser"""
    id: str
    data: Dict
    timestamp: str
    priority: int = 0  # 0 = normal, 1 = haute priorité
    retry_count: int = 0


class FirebaseSync:
    """
    Gestionnaire de synchronisation Firebase
    - Upload asynchrone des analyses
    - Gestion du mode hors-ligne
    - File d'attente persistante
    - Reprise automatique
    """
    
    def __init__(self):
        self.db = None
        self.bucket = None
        self.is_connected = False
        self.offline_storage = OfflineStorage()
        
        # File d'attente en mémoire
        self.sync_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # Thread de synchronisation
        self._sync_thread = None
        self._stop_event = threading.Event()
        
        # Initialiser Firebase
        self._init_firebase()
    
    def _init_firebase(self):
        """Initialise la connexion Firebase"""
        
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase SDK non installé")
            return
        
        try:
            cred_path = FIREBASE_CONFIG.get("credential_path")
            
            if not cred_path or not os.path.exists(cred_path):
                logger.warning(f"Fichier credentials non trouvé: {cred_path}")
                return
            
            # Initialiser l'app Firebase
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': FIREBASE_CONFIG.get("storage_bucket")
                })
            
            self.db = firestore.client()
            self.bucket = storage.bucket()
            self.is_connected = True
            
            logger.info("Firebase initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation Firebase: {e}")
            self.is_connected = False
    
    def start_sync_worker(self):
        """Démarre le worker de synchronisation en arrière-plan"""
        
        if self._sync_thread and self._sync_thread.is_alive():
            logger.warning("Worker déjà en cours d'exécution")
            return
        
        self._stop_event.clear()
        self._sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self._sync_thread.start()
        logger.info("Worker de synchronisation démarré")
    
    def stop_sync_worker(self):
        """Arrête le worker de synchronisation"""
        self._stop_event.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        logger.info("Worker de synchronisation arrêté")
    
    def _sync_worker(self):
        """Worker qui traite la file de synchronisation"""
        
        while not self._stop_event.is_set():
            try:
                # Vérifier la connectivité
                if not self._check_connectivity():
                    time.sleep(SERVER_CONFIG["sync_interval_seconds"])
                    continue
                
                # Traiter les items en attente offline d'abord
                self._process_offline_queue()
                
                # Traiter la file en mémoire
                try:
                    priority, item = self.sync_queue.get(timeout=1)
                    self._sync_item(item)
                    self.sync_queue.task_done()
                except queue.Empty:
                    pass
                
            except Exception as e:
                logger.error(f"Erreur worker sync: {e}")
                time.sleep(5)
    
    def _check_connectivity(self) -> bool:
        """Vérifie la connectivité internet"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            
            if not self.is_connected:
                self._init_firebase()
            
            return self.is_connected
        except OSError:
            return False
    
    def _process_offline_queue(self):
        """Traite les items stockés offline"""
        
        pending_items = self.offline_storage.get_pending_items()
        
        for item_id, item_data in pending_items:
            try:
                sync_item = SyncItem(
                    id=item_id,
                    data=item_data,
                    timestamp=item_data.get('timestamp', datetime.now().isoformat()),
                    priority=1  # Priorité haute pour rattrapage
                )
                
                if self._sync_item(sync_item):
                    self.offline_storage.mark_synced(item_id)
                    logger.info(f"Item offline synchronisé: {item_id}")
                    
            except Exception as e:
                logger.error(f"Erreur sync item offline {item_id}: {e}")
    
    def _sync_item(self, item: SyncItem) -> bool:
        """Synchronise un item vers Firebase"""
        
        if not self.is_connected or not self.db:
            # Sauvegarder offline
            self.offline_storage.save(item.id, item.data)
            return False
        
        try:
            # Préparer les données
            data = item.data.copy()
            data['_synced_at'] = firestore.SERVER_TIMESTAMP
            data['_device_id'] = self._get_device_id()
            
            # Upload vers Firestore
            doc_ref = self.db.collection('plant_analyses').document(item.id)
            doc_ref.set(data)
            
            logger.debug(f"Item synchronisé: {item.id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur sync Firestore: {e}")
            
            item.retry_count += 1
            if item.retry_count < SERVER_CONFIG["retry_attempts"]:
                # Remettre dans la file
                self.sync_queue.put((item.priority, item))
            else:
                # Sauvegarder offline après trop d'échecs
                self.offline_storage.save(item.id, item.data)
            
            return False
    
    def queue_analysis(self, analysis_result: Dict, priority: int = 0):
        """Ajoute une analyse à la file de synchronisation"""
        
        item = SyncItem(
            id=analysis_result.get('analysis_id', f"analysis_{datetime.now().timestamp()}"),
            data=analysis_result,
            timestamp=analysis_result.get('timestamp', datetime.now().isoformat()),
            priority=priority
        )
        
        # Si pas de connexion, sauvegarder directement offline
        if not self.is_connected:
            self.offline_storage.save(item.id, item.data)
            logger.debug(f"Sauvegardé offline: {item.id}")
        else:
            self.sync_queue.put((priority, item))
    
    async def upload_image(
        self, 
        image_data: bytes, 
        filename: str
    ) -> Optional[str]:
        """Upload une image vers Firebase Storage"""
        
        if not self.bucket:
            logger.warning("Storage non disponible")
            return None
        
        try:
            blob = self.bucket.blob(f"plant_images/{filename}")
            blob.upload_from_string(image_data, content_type='image/jpeg')
            
            # Rendre public et obtenir l'URL
            blob.make_public()
            return blob.public_url
            
        except Exception as e:
            logger.error(f"Erreur upload image: {e}")
            return None
    
    def get_analysis_history(
        self, 
        limit: int = 100,
        plant_species: str = None
    ) -> List[Dict]:
        """Récupère l'historique des analyses"""
        
        if not self.db:
            # Retourner les données offline
            return self.offline_storage.get_all_local()
        
        try:
            query = self.db.collection('plant_analyses')
            
            if plant_species:
                query = query.where('plant_species', '==', plant_species)
            
            query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)
            query = query.limit(limit)
            
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
            
        except Exception as e:
            logger.error(f"Erreur récupération historique: {e}")
            return []
    
    def _get_device_id(self) -> str:
        """Génère un ID unique pour cet appareil"""
        try:
            # Sur Raspberry Pi, utiliser l'ID matériel
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('Serial'):
                        return hashlib.md5(line.encode()).hexdigest()[:16]
        except:
            pass
        
        # Fallback: générer un ID basé sur le hostname
        import socket
        hostname = socket.gethostname()
        return hashlib.md5(hostname.encode()).hexdigest()[:16]
    
    def get_sync_status(self) -> Dict:
        """Retourne le statut de synchronisation"""
        return {
            "is_connected": self.is_connected,
            "pending_in_memory": self.sync_queue.qsize(),
            "pending_offline": self.offline_storage.count_pending(),
            "total_synced": self.offline_storage.count_synced(),
            "last_sync": self.offline_storage.get_last_sync_time()
        }


class SecureDataTransfer:
    """Gestion du transfert sécurisé des données"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or SECURITY_CONFIG.get("api_key")
    
    def encrypt_payload(self, data: Dict) -> bytes:
        """Chiffre les données avant envoi"""
        # Implémentation basique - en production, utiliser cryptography
        import base64
        json_data = json.dumps(data)
        return base64.b64encode(json_data.encode())
    
    def get_auth_headers(self) -> Dict:
        """Retourne les headers d'authentification"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Device-ID": self._get_device_id()
        }
    
    def _get_device_id(self) -> str:
        import socket
        return hashlib.md5(socket.gethostname().encode()).hexdigest()[:16]