"""
Gestion du stockage hors-ligne pour le mode déconnecté
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import threading

from loguru import logger

from .config import RASPBERRY_CONFIG


class OfflineStorage:
    """
    Stockage SQLite pour les données en mode hors-ligne
    Thread-safe pour utilisation avec le worker de sync
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            RASPBERRY_CONFIG.get("offline_storage_path", "/tmp"),
            "offline_data.db"
        )
        
        # Créer le dossier parent si nécessaire
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialise la base de données"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    synced INTEGER DEFAULT 0,
                    synced_at TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_synced 
                ON analyses(synced)
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    details TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Base offline initialisée: {self.db_path}")
    
    def save(self, item_id: str, data: Dict) -> bool:
        """Sauvegarde une analyse en local"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO analyses (id, data, timestamp, synced)
                    VALUES (?, ?, ?, 0)
                ''', (
                    item_id,
                    json.dumps(data),
                    data.get('timestamp', datetime.now().isoformat())
                ))
                
                conn.commit()
                conn.close()
                
                self._check_storage_limit()
                return True
                
            except Exception as e:
                logger.error(f"Erreur sauvegarde offline: {e}")
                return False
    
    def get_pending_items(self, limit: int = 50) -> List[Tuple[str, Dict]]:
        """Récupère les items non synchronisés"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, data FROM analyses 
                    WHERE synced = 0 
                    ORDER BY timestamp ASC
                    LIMIT ?
                ''', (limit,))
                
                results = cursor.fetchall()
                conn.close()
                
                return [(row[0], json.loads(row[1])) for row in results]
                
            except Exception as e:
                logger.error(f"Erreur lecture pending: {e}")
                return []
    
    def mark_synced(self, item_id: str) -> bool:
        """Marque un item comme synchronisé"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE analyses 
                    SET synced = 1, synced_at = ?
                    WHERE id = ?
                ''', (datetime.now().isoformat(), item_id))
                
                # Logger l'événement
                cursor.execute('''
                    INSERT INTO sync_log (event_type, details)
                    VALUES ('SYNC_SUCCESS', ?)
                ''', (item_id,))
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logger.error(f"Erreur marquage synced: {e}")
                return False
    
    def count_pending(self) -> int:
        """Compte les items en attente de sync"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM analyses WHERE synced = 0')
                count = cursor.fetchone()[0]
                conn.close()
                return count
            except:
                return 0
    
    def count_synced(self) -> int:
        """Compte les items synchronisés"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM analyses WHERE synced = 1')
                count = cursor.fetchone()[0]
                conn.close()
                return count
            except:
                return 0
    
    def get_last_sync_time(self) -> Optional[str]:
        """Récupère le timestamp de la dernière sync"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT MAX(synced_at) FROM analyses WHERE synced = 1
                ''')
                result = cursor.fetchone()[0]
                conn.close()
                return result
            except:
                return None
    
    def get_all_local(self, limit: int = 100) -> List[Dict]:
        """Récupère toutes les analyses locales"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT data FROM analyses 
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                results = cursor.fetchall()
                conn.close()
                
                return [json.loads(row[0]) for row in results]
                
            except Exception as e:
                logger.error(f"Erreur lecture all: {e}")
                return []
    
    def _check_storage_limit(self):
        """Vérifie et nettoie si la limite de stockage est atteinte"""
        max_size_bytes = RASPBERRY_CONFIG.get("max_offline_storage_mb", 500) * 1024 * 1024
        
        try:
            current_size = os.path.getsize(self.db_path)
            
            if current_size > max_size_bytes:
                logger.warning(f"Limite stockage atteinte: {current_size / 1024 / 1024:.1f} MB")
                self._cleanup_old_synced()
                
        except:
            pass
    
    def _cleanup_old_synced(self, keep_recent: int = 1000):
        """Supprime les anciens items déjà synchronisés"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Garder les N plus récents
            cursor.execute('''
                DELETE FROM analyses 
                WHERE synced = 1 
                AND id NOT IN (
                    SELECT id FROM analyses 
                    WHERE synced = 1 
                    ORDER BY synced_at DESC 
                    LIMIT ?
                )
            ''', (keep_recent,))
            
            deleted = cursor.rowcount
            conn.commit()
            
            # Vacuum pour récupérer l'espace
            cursor.execute('VACUUM')
            conn.close()
            
            logger.info(f"Nettoyage: {deleted} anciens items supprimés")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage: {e}")
    
    def export_to_json(self, output_path: str) -> bool:
        """Exporte toutes les données vers un fichier JSON"""
        try:
            data = self.get_all_local(limit=10000)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exporté {len(data)} items vers {output_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur export: {e}")
            return False
    
    def import_from_json(self, input_path: str) -> int:
        """Importe des données depuis un fichier JSON"""
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            count = 0
            for item in data:
                item_id = item.get('analysis_id', f"import_{count}")
                if self.save(item_id, item):
                    count += 1
            
            logger.info(f"Importé {count} items depuis {input_path}")
            return count
            
        except Exception as e:
            logger.error(f"Erreur import: {e}")
            return 0