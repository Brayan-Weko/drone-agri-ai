"""
Communication avec le Pixhawk via MAVLink
"""

import json
import threading
import time
from typing import Dict, Optional, Callable

from loguru import logger

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    logger.warning("pymavlink non disponible")


class PixhawkCommunicator:
    """
    Gère la communication bidirectionnelle avec le Pixhawk
    """
    
    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 57600
    ):
        self.port = port
        self.baudrate = baudrate
        self.connection = None
        self.running = False
        
        self._recv_thread = None
        self._message_handlers: Dict[str, Callable] = {}
        
        if MAVLINK_AVAILABLE:
            self._connect()
    
    def _connect(self):
        """Établit la connexion MAVLink"""
        try:
            self.connection = mavutil.mavlink_connection(
                self.port,
                baud=self.baudrate
            )
            
            # Attendre le heartbeat
            self.connection.wait_heartbeat(timeout=10)
            logger.info(f"Pixhawk connecté (system {self.connection.target_system})")
            
            # Démarrer la réception
            self.running = True
            self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._recv_thread.start()
            
        except Exception as e:
            logger.error(f"Erreur connexion Pixhawk: {e}")
            self.connection = None
    
    def _receive_loop(self):
        """Boucle de réception des messages MAVLink"""
        while self.running and self.connection:
            try:
                msg = self.connection.recv_match(blocking=True, timeout=1)
                if msg:
                    self._handle_message(msg)
            except Exception as e:
                logger.error(f"Erreur réception MAVLink: {e}")
    
    def _handle_message(self, msg):
        """Traite un message MAVLink reçu"""
        msg_type = msg.get_type()
        
        # Ignorer les heartbeats
        if msg_type == "HEARTBEAT":
            return
        
        # GPS Position
        if msg_type == "GLOBAL_POSITION_INT":
            self._current_position = {
                "lat": msg.lat / 1e7,
                "lon": msg.lon / 1e7,
                "alt": msg.alt / 1000,  # mm -> m
                "heading": msg.hdg / 100  # centidegrees -> degrees
            }
        
        # Appeler les handlers enregistrés
        if msg_type in self._message_handlers:
            self._message_handlers[msg_type](msg)
    
    def send_message(self, data: Dict):
        """
        Envoie un message au Pixhawk
        Utilise STATUSTEXT pour envoyer des informations textuelles
        """
        if not self.connection:
            logger.warning("Pixhawk non connecté")
            return
        
        try:
            # Formater le message
            text = json.dumps(data)[:50]  # Limite STATUSTEXT
            
            self.connection.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_INFO,
                text.encode('utf-8')
            )
            
            logger.debug(f"Message envoyé au Pixhawk: {text}")
            
        except Exception as e:
            logger.error(f"Erreur envoi Pixhawk: {e}")
    
    def send_spray_command(self, activate: bool, duration_ms: int = 500):
        """
        Envoie une commande de pulvérisation
        Utilise un canal RC virtuel ou un GPIO
        """
        if not self.connection:
            return
        
        # Valeur PWM: 1000 = off, 2000 = on
        pwm_value = 2000 if activate else 1000
        
        try:
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,  # confirmation
                9,  # servo channel (AUX1 = channel 9)
                pwm_value,
                0, 0, 0, 0, 0
            )
            
            if activate and duration_ms > 0:
                # Programmer l'arrêt automatique
                threading.Timer(
                    duration_ms / 1000,
                    lambda: self.send_spray_command(False, 0)
                ).start()
            
            logger.info(f"Pulvérisation: {'ON' if activate else 'OFF'}")
            
        except Exception as e:
            logger.error(f"Erreur commande spray: {e}")
    
    def get_current_position(self) -> Optional[Dict]:
        """Retourne la position GPS actuelle"""
        return getattr(self, '_current_position', None)
    
    def register_handler(self, msg_type: str, handler: Callable):
        """Enregistre un handler pour un type de message"""
        self._message_handlers[msg_type] = handler
    
    def close(self):
        """Ferme la connexion"""
        self.running = False
        
        if self._recv_thread:
            self._recv_thread.join(timeout=2)
        
        if self.connection:
            self.connection.close()
        
        logger.info("Connexion Pixhawk fermée")