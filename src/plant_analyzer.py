"""
Analyseur de plantes - Module principal d'inférence et d'analyse
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import cv2
from PIL import Image
from loguru import logger

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except ImportError:
    import tensorflow as tf
    TFLITE_RUNTIME = False

from .config import (
    MODEL_CONFIG, 
    MODELS_DIR, 
    GROWTH_STAGES,
    RASPBERRY_CONFIG
)
from .data_loader import ImagePreprocessor


@dataclass
class AnalysisResult:
    """Résultat d'analyse d'une plante"""
    
    # Identifiants
    analysis_id: str
    timestamp: str
    
    # Détection
    is_plant: bool
    is_plant_confidence: float
    
    # Classification
    plant_species: str
    species_confidence: float
    condition: str
    
    # Santé
    health_score: float  # 0-100
    health_status: str  # "healthy", "warning", "critical"
    
    # Croissance
    growth_stage: str
    growth_stage_confidence: float
    expected_characteristics: Dict
    
    # Diagnostic
    issues_detected: List[str]
    recommendations: List[str]
    
    # Métadonnées
    inference_time_ms: float
    model_version: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class PlantAnalyzer:
    """
    Analyseur de plantes temps réel pour drone agricole
    Optimisé pour Raspberry Pi avec TensorFlow Lite
    """
    
    def __init__(
        self,
        model_path: str = None,
        class_mapping_path: str = None,
        use_coral: bool = False
    ):
        self.model_path = model_path or str(MODELS_DIR / "plant_model.tflite")
        self.class_mapping_path = class_mapping_path or str(MODELS_DIR / "class_mapping.json")
        self.use_coral = use_coral
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_mapping = None
        self.preprocessor = ImagePreprocessor(MODEL_CONFIG["input_size"])
        
        self.model_version = "1.0.0"
        self._load_model()
        self._load_class_mapping()
    
    def _load_model(self):
        """Charge le modèle TFLite"""
        
        try:
            if self.use_coral:
                # Utiliser Edge TPU
                try:
                    from pycoral.utils.edgetpu import make_interpreter
                    self.interpreter = make_interpreter(self.model_path)
                    logger.info("Modèle chargé sur Coral Edge TPU")
                except ImportError:
                    logger.warning("Coral non disponible, utilisation CPU")
                    self._load_cpu_interpreter()
            else:
                self._load_cpu_interpreter()
            
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"Modèle chargé: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise
    
    def _load_cpu_interpreter(self):
        """Charge l'interpréteur CPU"""
        if TFLITE_RUNTIME:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
    
    def _load_class_mapping(self):
        """Charge le mapping des classes"""
        try:
            with open(self.class_mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
            logger.info(f"Classes chargées: {len(self.class_mapping['class_names'])}")
        except FileNotFoundError:
            logger.warning("Fichier de mapping non trouvé, utilisation par défaut")
            self.class_mapping = {
                "class_names": [],
                "class_info": {}
            }
    
    def analyze(
        self, 
        image: np.ndarray,
        return_visualization: bool = False
    ) -> Tuple[AnalysisResult, Optional[np.ndarray]]:
        """
        Analyse une image de plante
        
        Args:
            image: Image BGR ou RGB (numpy array)
            return_visualization: Si True, retourne aussi l'image annotée
            
        Returns:
            Tuple (AnalysisResult, image annotée optionnelle)
        """
        
        start_time = time.perf_counter()
        
        # Prétraitement
        input_data = self.preprocessor.preprocess(image)
        
        # Inférence
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Récupérer les sorties
        outputs = {}
        for detail in self.output_details:
            name = detail['name']
            tensor = self.interpreter.get_tensor(detail['index'])
            outputs[name] = tensor
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Parser les résultats
        result = self._parse_outputs(outputs, inference_time)
        
        # Générer visualisation si demandé
        vis_image = None
        if return_visualization:
            vis_image = self._create_visualization(image, result)
        
        return result, vis_image
    
    def _parse_outputs(
        self, 
        outputs: Dict[str, np.ndarray],
        inference_time: float
    ) -> AnalysisResult:
        """Parse les sorties du modèle"""
        
        # Générer ID unique
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # === Détection plante ===
        is_plant_score = float(outputs.get('is_plant', [[0.5]])[0][0])
        is_plant = is_plant_score > MODEL_CONFIG["confidence_threshold"]
        
        # === Classification ===
        class_probs = outputs.get('classification', [[]])[0]
        if len(class_probs) > 0:
            class_idx = int(np.argmax(class_probs))
            class_confidence = float(class_probs[class_idx])
            
            if class_idx < len(self.class_mapping.get('class_names', [])):
                class_name = self.class_mapping['class_names'][class_idx]
                class_info = self.class_mapping.get('class_info', {}).get(class_name, {})
            else:
                class_name = f"class_{class_idx}"
                class_info = {}
        else:
            class_name = "unknown"
            class_confidence = 0.0
            class_info = {}
        
        plant_species = class_info.get('plant', class_name)
        condition = class_info.get('condition', 'unknown')
        
        # === Score de santé ===
        health_score = float(outputs.get('health_score', [[0.5]])[0][0]) * 100
        health_status = self._get_health_status(health_score)
        
        # === Stade de croissance ===
        growth_probs = outputs.get('growth_stage', [[0.25, 0.25, 0.25, 0.25]])[0]
        growth_idx = int(np.argmax(growth_probs))
        growth_confidence = float(growth_probs[growth_idx])
        
        growth_stages_list = list(GROWTH_STAGES.keys())
        growth_stage = growth_stages_list[growth_idx] if growth_idx < len(growth_stages_list) else "unknown"
        
        # === Caractéristiques attendues ===
        expected = self._get_expected_characteristics(plant_species, growth_stage)
        
        # === Diagnostic ===
        issues, recommendations = self._generate_diagnosis(
            plant_species=plant_species,
            condition=condition,
            health_score=health_score,
            growth_stage=growth_stage,
            is_healthy=class_info.get('is_healthy', True)
        )
        
        return AnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            is_plant=is_plant,
            is_plant_confidence=is_plant_score,
            plant_species=plant_species,
            species_confidence=class_confidence,
            condition=condition,
            health_score=health_score,
            health_status=health_status,
            growth_stage=growth_stage,
            growth_stage_confidence=growth_confidence,
            expected_characteristics=expected,
            issues_detected=issues,
            recommendations=recommendations,
            inference_time_ms=inference_time,
            model_version=self.model_version
        )
    
    def _get_health_status(self, score: float) -> str:
        """Détermine le statut de santé basé sur le score"""
        if score >= 80:
            return "healthy"
        elif score >= 50:
            return "warning"
        else:
            return "critical"
    
    def _get_expected_characteristics(
        self, 
        species: str, 
        growth_stage: str
    ) -> Dict:
        """
        Retourne les caractéristiques attendues pour une plante
        à un stade de croissance donné
        """
        
        # Base de connaissances (à enrichir)
        plant_knowledge = {
            "Tomato": {
                "seedling": {
                    "height_cm": "5-15",
                    "leaves": "2-4 vraies feuilles",
                    "color": "Vert clair",
                    "needs": "Lumière indirecte, arrosage modéré"
                },
                "vegetative": {
                    "height_cm": "30-60",
                    "leaves": "10-20 feuilles",
                    "color": "Vert foncé",
                    "needs": "Tuteurage, fertilisation azotée"
                },
                "flowering": {
                    "height_cm": "60-100",
                    "leaves": "20+ feuilles",
                    "color": "Vert foncé, fleurs jaunes",
                    "needs": "Pollinisation, fertilisation P-K"
                },
                "mature": {
                    "height_cm": "100-180",
                    "leaves": "Feuillage dense",
                    "color": "Vert + fruits rouges",
                    "needs": "Récolte régulière, arrosage constant"
                }
            },
            "Potato": {
                "seedling": {
                    "height_cm": "5-10",
                    "leaves": "Premières pousses",
                    "color": "Vert pâle",
                    "needs": "Buttage initial"
                },
                "vegetative": {
                    "height_cm": "20-40",
                    "leaves": "Feuillage développé",
                    "color": "Vert dense",
                    "needs": "Buttage, arrosage régulier"
                },
                "flowering": {
                    "height_cm": "40-60",
                    "leaves": "Maximum de feuillage",
                    "color": "Vert + fleurs blanches/violettes",
                    "needs": "Réduire azote, maintenir humidité"
                },
                "mature": {
                    "height_cm": "40-60 (déclin)",
                    "leaves": "Jaunissement naturel",
                    "color": "Jaune/brun",
                    "needs": "Arrêter arrosage, préparer récolte"
                }
            }
        }
        
        # Chercher dans la base
        species_clean = species.split()[0] if species else ""
        if species_clean in plant_knowledge:
            stage_info = plant_knowledge[species_clean].get(growth_stage, {})
            return stage_info
        
        # Retour par défaut
        stage_info = GROWTH_STAGES.get(growth_stage, {})
        return {
            "description": stage_info.get("description", growth_stage),
            "general_advice": "Consultez un guide spécifique pour cette espèce"
        }
    
    def _generate_diagnosis(
        self,
        plant_species: str,
        condition: str,
        health_score: float,
        growth_stage: str,
        is_healthy: bool
    ) -> Tuple[List[str], List[str]]:
        """Génère le diagnostic et les recommandations"""
        
        issues = []
        recommendations = []
        
        # Analyse de la condition
        if not is_healthy and condition != "healthy":
            condition_clean = condition.lower()
            
            # Maladies courantes et recommandations
            disease_map = {
                "late blight": {
                    "issue": "Mildiou détecté - infection fongique grave",
                    "reco": [
                        "Appliquer un fongicide à base de cuivre immédiatement",
                        "Retirer les feuilles infectées",
                        "Améliorer la circulation d'air",
                        "Éviter l'arrosage par aspersion"
                    ]
                },
                "early blight": {
                    "issue": "Alternariose détectée",
                    "reco": [
                        "Appliquer un fongicide préventif",
                        "Pailler le sol pour éviter les éclaboussures",
                        "Rotation des cultures recommandée"
                    ]
                },
                "bacterial spot": {
                    "issue": "Taches bactériennes détectées",
                    "reco": [
                        "Éviter l'arrosage sur le feuillage",
                        "Appliquer du cuivre en traitement préventif",
                        "Éliminer les plants très infectés"
                    ]
                },
                "yellow leaf curl": {
                    "issue": "Virus de l'enroulement des feuilles",
                    "reco": [
                        "Contrôler les aleurodes (vecteurs)",
                        "Retirer les plants infectés",
                        "Utiliser des variétés résistantes"
                    ]
                },
                "mosaic virus": {
                    "issue": "Virus de la mosaïque détecté",
                    "reco": [
                        "Aucun traitement curatif",
                        "Détruire les plants infectés",
                        "Désinfecter les outils"
                    ]
                },
                "septoria": {
                    "issue": "Septoriose détectée",
                    "reco": [
                        "Retirer les feuilles basses infectées",
                        "Appliquer un fongicide",
                        "Améliorer l'espacement des plants"
                    ]
                },
                "spider mites": {
                    "issue": "Infestation d'acariens",
                    "reco": [
                        "Pulvériser de l'eau pour déloger les acariens",
                        "Utiliser un acaricide biologique",
                        "Augmenter l'humidité ambiante"
                    ]
                }
            }
            
            # Chercher la maladie correspondante
            for disease_key, disease_info in disease_map.items():
                if disease_key in condition_clean:
                    issues.append(disease_info["issue"])
                    recommendations.extend(disease_info["reco"])
                    break
            else:
                # Maladie non reconnue spécifiquement
                issues.append(f"Problème détecté: {condition}")
                recommendations.append("Consulter un expert agricole pour diagnostic précis")
        
        # Analyse du score de santé
        if health_score < 50:
            issues.append(f"Score de santé critique: {health_score:.0f}%")
            recommendations.append("Intervention urgente recommandée")
        elif health_score < 80:
            issues.append(f"Score de santé moyen: {health_score:.0f}%")
            recommendations.append("Surveillance accrue conseillée")
        
        # Si aucun problème
        if not issues:
            recommendations.append("Plante en bonne santé - Maintenir les soins actuels")
            recommendations.append(f"Stade {growth_stage}: suivre les pratiques standard")
        
        return issues, recommendations
    
    def _create_visualization(
        self, 
        image: np.ndarray, 
        result: AnalysisResult
    ) -> np.ndarray:
        """Crée une image annotée avec les résultats"""
        
        vis = image.copy()
        h, w = vis.shape[:2]
        
        # Couleur selon le statut
        status_colors = {
            "healthy": (0, 255, 0),    # Vert
            "warning": (0, 165, 255),  # Orange
            "critical": (0, 0, 255)    # Rouge
        }
        color = status_colors.get(result.health_status, (128, 128, 128))
        
        # Cadre autour de l'image
        cv2.rectangle(vis, (0, 0), (w-1, h-1), color, 3)
        
        # Zone d'information en haut
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)
        
        # Textes
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Espèce et condition
        text1 = f"{result.plant_species} - {result.condition}"
        cv2.putText(vis, text1, (10, 25), font, 0.6, (255, 255, 255), 2)
        
        # Score de santé
        text2 = f"Sante: {result.health_score:.0f}% ({result.health_status})"
        cv2.putText(vis, text2, (10, 50), font, 0.5, color, 2)
        
        # Stade de croissance
        text3 = f"Stade: {result.growth_stage} ({result.growth_stage_confidence*100:.0f}%)"
        cv2.putText(vis, text3, (10, 70), font, 0.4, (200, 200, 200), 1)
        
        return vis
    
    def analyze_batch(
        self, 
        images: List[np.ndarray]
    ) -> List[AnalysisResult]:
        """Analyse un batch d'images"""
        results = []
        for img in images:
            result, _ = self.analyze(img, return_visualization=False)
            results.append(result)
        return results


# === Point d'entrée pour tests rapides ===
if __name__ == "__main__":
    # Test avec une image
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is not None:
            analyzer = PlantAnalyzer()
            result, vis = analyzer.analyze(image, return_visualization=True)
            
            print("\n=== Résultat d'analyse ===")
            print(result.to_json())
            
            if vis is not None:
                cv2.imwrite("analysis_result.jpg", vis)
                print("\nVisualisation sauvegardée: analysis_result.jpg")
        else:
            print(f"Impossible de charger l'image: {image_path}")
    else:
        print("Usage: python plant_analyzer.py <chemin_image>")