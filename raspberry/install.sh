#!/bin/bash
# Script d'installation pour Raspberry Pi 4
# Usage: sudo bash install.sh

set -e

echo "=========================================="
echo "  Installation Drone Agri AI"
echo "  Raspberry Pi 4"
echo "=========================================="

# Vérifier si on est root
if [ "$EUID" -ne 0 ]; then
    echo "Veuillez exécuter avec sudo"
    exit 1
fi

# Mettre à jour le système
echo "[1/8] Mise à jour du système..."
apt-get update && apt-get upgrade -y

# Installer les dépendances système
echo "[2/8] Installation des dépendances système..."
apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    libharfbuzz0b \
    libwebp6 \
    libtiff5 \
    libjasper1 \
    libilmbase23 \
    libopenexr23 \
    libgstreamer1.0-0 \
    libavcodec-extra58 \
    libavformat58 \
    libswscale5 \
    git

# Installer picamera2
echo "[3/8] Installation de picamera2..."
apt-get install -y python3-picamera2

# Créer le répertoire du projet
echo "[4/8] Configuration du projet..."
PROJECT_DIR="/home/pi/drone-agri-ai"
mkdir -p $PROJECT_DIR
mkdir -p $PROJECT_DIR/models
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/offline_data
mkdir -p $PROJECT_DIR/captures
mkdir -p /var/log/drone-agri-ai

# Créer l'environnement virtuel
echo "[5/8] Création de l'environnement virtuel..."
python3 -m venv $PROJECT_DIR/venv
source $PROJECT_DIR/venv/bin/activate

# Installer les dépendances Python
echo "[6/8] Installation des dépendances Python..."
pip install --upgrade pip
pip install -r $PROJECT_DIR/requirements-raspberry.txt

# Installer TFLite runtime optimisé
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl || pip install tflite-runtime

# Configurer le service systemd
echo "[7/8] Configuration du service systemd..."
cat > /etc/systemd/system/drone-agri-ai.service << EOF
[Unit]
Description=Drone Agri AI Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin:/usr/bin
ExecStart=$PROJECT_DIR/venv/bin/python raspberry/main.py --verbose
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable drone-agri-ai

# Configurer les permissions
echo "[8/8] Configuration des permissions..."
chown -R pi:pi $PROJECT_DIR
chmod +x $PROJECT_DIR/raspberry/main.py

# Configurer le port série pour Pixhawk
if ! grep -q "enable_uart=1" /boot/config.txt; then
    echo "enable_uart=1" >> /boot/config.txt
fi

# Désactiver la console série
systemctl disable serial-getty@ttyAMA0.service 2>/dev/null || true

echo ""
echo "=========================================="
echo "  Installation terminée!"
echo "=========================================="
echo ""
echo "Prochaines étapes:"
echo "1. Copier le modèle TFLite dans: $PROJECT_DIR/models/"
echo "2. Configurer Firebase: créer firebase-key.json"
echo "3. Créer .env avec les variables d'environnement"
echo "4. Redémarrer: sudo reboot"
echo "5. Démarrer le service: sudo systemctl start drone-agri-ai"
echo "6. Voir les logs: sudo journalctl -u drone-agri-ai -f"
echo ""