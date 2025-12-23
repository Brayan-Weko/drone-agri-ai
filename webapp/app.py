"""
Application Flask pour tester le modèle via navigateur
Compatible PC, tablette, téléphone
"""

import os
import io
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from loguru import logger

# Ajouter le chemin parent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plant_analyzer import PlantAnalyzer
from src.server_sync import FirebaseSync

app = Flask(__name__)
CORS(app)

# Initialiser l'analyseur
analyzer = None
sync = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        model_path = os.environ.get(
            'MODEL_PATH', 
            str(Path(__file__).parent.parent / 'models' / 'plant_model.tflite')
        )
        analyzer = PlantAnalyzer(model_path=model_path)
    return analyzer

def get_sync():
    global sync
    if sync is None:
        sync = FirebaseSync()
    return sync


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint d'analyse d'image"""
    try:
        # Récupérer l'image
        if 'image' in request.files:
            # Upload de fichier
            file = request.files['image']