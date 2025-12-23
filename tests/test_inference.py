"""
Tests du module d'inférence
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImagePreprocessor(unittest.TestCase):
    """Tests du préprocesseur d'images"""
    
    def setUp(self):
        from src.data_loader import ImagePreprocessor
        self.preprocessor = ImagePreprocessor(input_size=(224, 224))
    
    def test_preprocess_shape(self):
        """Test de la forme de sortie"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = self.preprocessor.preprocess(image)
        
        self.assertEqual(processed.shape, (1, 224, 224, 3))
    
    def test_preprocess_dtype(self):
        """Test du type de données"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = self.preprocessor.preprocess(image)
        
        self.assertEqual(processed.dtype, np.float32)
    
    def test_preprocess_normalization(self):
        """Test de la normalisation"""
        image = np.full((224, 224, 3), 128, dtype=np.uint8)
        processed = self.preprocessor.preprocess(image)
        
        # Après normalisation, les valeurs ne devraient plus être dans [0, 255]
        self.assertTrue(np.max(processed) <= 3)  # Environ [-2, 2] après normalisation
        self.assertTrue(np.min(processed) >= -3)
    
    def test_preprocess_batch(self):
        """Test du prétraitement par batch"""
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        processed = self.preprocessor.preprocess_batch(images)
        
        self.assertEqual(processed.shape, (5, 224, 224, 3))


class TestTFLiteEngine(unittest.TestCase):
    """Tests du moteur TFLite"""
    
    @classmethod
    def setUpClass(cls):
        """Créer un modèle TFLite de test"""
        import tensorflow as tf
        
        # Créer un modèle simple
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(8, 3, padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Convertir en TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Sauvegarder
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.temp_dir, 'test_model.tflite')
        with open(cls.model_path, 'wb') as f:
            f.write(tflite_model)
    
    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir)
    
    def test_engine_creation(self):
        """Test de création du moteur"""
        from src.inference import TFLiteEngine
        engine = TFLiteEngine(self.model_path)
        self.assertIsNotNone(engine.interpreter)
    
    def test_engine_predict(self):
        """Test de prédiction"""
        from src.inference import TFLiteEngine
        engine = TFLiteEngine(self.model_path)
        
        input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        outputs = engine.predict(input_data)
        
        self.assertIsInstance(outputs, dict)
    
    def test_engine_run(self):
        """Test du pipeline complet"""
        from src.inference import TFLiteEngine
        engine = TFLiteEngine(self.model_path)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = engine.run(image)
        
        self.assertIsNotNone(result.outputs)
        self.assertGreater(result.inference_time_ms, 0)


class TestPlantAnalyzer(unittest.TestCase):
    """Tests de l'analyseur de plantes"""
    
    def test_analysis_result_structure(self):
        """Test de la structure du résultat"""
        from src.plant_analyzer import AnalysisResult
        
        result = AnalysisResult(
            analysis_id="test_001",
            timestamp="2024-01-01T00:00:00",
            is_plant=True,
            is_plant_confidence=0.95,
            plant_species="Tomato",
            species_confidence=0.87,
            condition="healthy",
            health_score=85.0,
            health_status="healthy",
            growth_stage="vegetative",
            growth_stage_confidence=0.75,
            expected_characteristics={"height": "30-60cm"},
            issues_detected=[],
            recommendations=["Continue watering"],
            inference_time_ms=45.2,
            model_version="1.0.0"
        )
        
        # Tester la conversion en dict
        result_dict = result.to_dict()
        self.assertIn('analysis_id', result_dict)
        self.assertIn('is_plant', result_dict)
        self.assertIn('health_score', result_dict)
        
        # Tester la conversion en JSON
        json_str = result.to_json()
        self.assertIn('Tomato', json_str)
    
    def test_health_status_classification(self):
        """Test de la classification du statut de santé"""
        from src.plant_analyzer import PlantAnalyzer
        
        # Créer un analyseur mock
        class MockAnalyzer(PlantAnalyzer):
            def __init__(self):
                pass
            
        analyzer = MockAnalyzer()
        analyzer._get_health_status = PlantAnalyzer._get_health_status
        
        self.assertEqual(analyzer._get_health_status(analyzer, 90), "healthy")
        self.assertEqual(analyzer._get_health_status(analyzer, 70), "warning")
        self.assertEqual(analyzer._get_health_status(analyzer, 30), "critical")


class TestBenchmark(unittest.TestCase):
    """Tests du benchmark"""
    
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(8, 3, padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10)
        ])
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.temp_dir, 'test_model.tflite')
        with open(cls.model_path, 'wb') as f:
            f.write(tflite_model)
    
    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir)
    
    def test_benchmark_run(self):
        """Test de l'exécution du benchmark"""
        from src.inference import TFLiteEngine, InferenceBenchmark
        
        engine = TFLiteEngine(self.model_path)
        benchmark = InferenceBenchmark(engine)
        
        stats = benchmark.run(num_iterations=10, warmup=2)
        
        self.assertIn('mean_ms', stats)
        self.assertIn('fps', stats)
        self.assertGreater(stats['fps'], 0)


if __name__ == '__main__':
    unittest.main()