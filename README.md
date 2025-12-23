# 🌱 Drone Agri AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Système d'intelligence artificielle embarqué pour drone agricole** - Analyse en temps réel de la santé des plantes avec déploiement sur Raspberry Pi.

![Demo](docs/demo.gif)

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Entraînement](#-entraînement)
- [Déploiement](#-déploiement)
- [API](#-api)
- [Tests](#-tests)
- [Contribution](#-contribution)

## ✨ Fonctionnalités

### 🔍 Analyse multi-tâches
- **Détection plante/non-plante** - Filtre les images non pertinentes
- **Identification d'espèce** - 38 classes de plantes (extensible)
- **Diagnostic de santé** - Score de santé 0-100%
- **Stade de croissance** - Semis, Végétatif, Floraison, Mature
- **Recommandations** - Conseils personnalisés basés sur l'analyse

### ⚡ Performance
- Inférence < 100ms sur Raspberry Pi 4
- Support Coral Edge TPU pour accélération 10x
- Mode hors-ligne avec synchronisation différée
- Optimisation TensorFlow Lite (FP16/INT8)

### 🔒 Sécurité
- Chiffrement des données en transit
- Authentification API
- Stockage local sécurisé

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      DRONE AGRICOLE                        │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐     ┌─────────────┐    │
│  │   Caméra    │───▶│ Raspberry  │◀──▶│  Pixhawk    │    │
│  │  Pi Camera  │    │   Pi 4      │     │  (Flight)   │    │
│  └─────────────┘    └──────┬──────┘     └─────────────┘    │
│                            │                               │
│                     ┌──────▼──────┐                        │
│                     │  TFLite +   │                        │
│                     │ Coral TPU   │                        │
│                     └──────┬──────┘                        │
└────────────────────────────┼───────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │    Firebase     │
                    │  (Cloud Sync)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Application   │
                    │    Mobile/Web   │
                    └─────────────────┘
```

## 🚀 Installation

### Prérequis
- Python 3.9+
- pip ou conda
- Git

### Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/drone-agri-ai.git
cd drone-agri-ai

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Installation Raspberry Pi

```bash
# Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/drone-agri-ai.git
cd drone-agri-ai

# Exécuter le script d'installation
sudo bash raspberry/install.sh
```

## 🎓 Entraînement

### Option 1: Google Colab (Recommandé)

1. Ouvrir [Google Colab](https://colab.research.google.com)
2. Cloner le dépôt dans Colab:
   ```python
   !git clone https://github.com/VOTRE_USERNAME/drone-agri-ai.git
   %cd drone-agri-ai
   ```
3. Exécuter les notebooks dans l'ordre:
   - `01_data_preparation.ipynb`
   - `02_model_training.ipynb`
   - `03_model_optimization.ipynb`
   - `04_testing_evaluation.ipynb`

### Option 2: Local

```bash
# Télécharger les datasets
python scripts/download_data.py

# Lancer l'entraînement
python scripts/train.py --epochs 50 --batch_size 32
```

## 📱 Déploiement

### Sur Raspberry Pi

1. Copier le modèle TFLite:
   ```bash
   scp models/plant_model.tflite pi@<IP>:/home/pi/drone-agri-ai/models/
   ```

2. Configurer Firebase (optionnel):
   ```bash
   scp firebase-key.json pi@<IP>:/home/pi/drone-agri-ai/
   ```

3. Démarrer le service:
   ```bash
   sudo systemctl start drone-agri-ai
   ```

### Application Web de test

```bash
cd webapp
python app.py
# Ouvrir http://localhost:5000
```

## 📖 API

### Endpoint d'analyse

```http
POST /analyze
Content-Type: multipart/form-data

file: <image_file>
```

**Réponse:**
```json
{
  "analysis_id": "analysis_20240101_120000",
  "is_plant": true,
  "plant_species": "Tomato",
  "condition": "Early blight",
  "health_score": 65.5,
  "health_status": "warning",
  "growth_stage": "flowering",
  "recommendations": [
    "Appliquer un fongicide préventif",
    "Améliorer la circulation d'air"
  ],
  "inference_time_ms": 87.3
}
```

## 🧪 Tests

```bash
# Tous les tests
python -m pytest tests/

# Tests spécifiques
python -m pytest tests/test_model.py -v

# Avec couverture
python -m pytest --cov=src tests/
```

## 📊 Performances

| Modèle | Taille | Temps (Pi4) | Accuracy |
|--------|--------|-------------|----------|
| FP32   | 25 MB  | 250 ms      | 95.2%    |
| FP16   | 13 MB  | 150 ms      | 95.0%    |
| INT8   | 7 MB   | 80 ms       | 94.1%    |
| Coral  | 7 MB   | 15 ms       | 94.1%    |

## 🗂 Structure du projet

```
drone-agri-ai/
├── notebooks/          # Notebooks Jupyter/Colab
├── src/                # Code source principal
├── raspberry/          # Scripts Raspberry Pi
├── webapp/             # Application web de test
├── models/             # Modèles entraînés
├── data/               # Datasets (non versionné)
├── tests/              # Tests unitaires
├── docs/               # Documentation
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 License

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE).

## 🙏 Remerciements

- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- [TensorFlow Team](https://www.tensorflow.org/)
- [ArduPilot Community](https://ardupilot.org/)

---

**Développé avec ❤️ par Brayan Weko pour l'agriculture de précision**