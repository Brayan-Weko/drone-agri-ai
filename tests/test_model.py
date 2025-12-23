"""
Tests unitaires pour le modèle de classification de plantes
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Ajouter le chemin parent
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf


class TestModelArchitecture(unittest.TestCase):
    """Tests de l'architecture du modèle"""
    
    @classmethod
    def setUpClass(cls):
        """Charger le modèle une fois pour tous les tests"""
        from src.model import PlantClassificationModel
        cls.model_builder = PlantClassificationModel(num_classes=38)
        cls.model = cls.model_builder.build(compile_model=True)
    
    def test_model_creation(self):
        """Test de création du modèle"""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, tf.keras.Model)
    
    def test_input_shape(self):
        """Test de la forme d'entrée"""
        input_shape = self.model.input_shape
        self.assertEqual(input_shape, (None, 224, 224, 3))
    
    def test_output_shapes(self):
        """Test des formes de sortie"""
        outputs = self.model.output
        self.assertIsInstance(outputs, dict)
        
        # Vérifier chaque sortie
        self.assertIn('is_plant', outputs)
        self.assertIn('classification', outputs)
        self.assertIn('health_score', outputs)
        self.assertIn('growth_stage', outputs)
    
    def test_forward_pass(self):
        """Test d'un passage forward"""
        # Créer une entrée aléatoire
        batch_size = 4
        input_data = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
        
        # Prédire
        outputs = self.model.predict(input_data, verbose=0)
        
        # Vérifier les sorties
        self.assertEqual(outputs['is_plant'].shape, (batch_size, 1))
        self.assertEqual(outputs['classification'].shape, (batch_size, 38))
        self.assertEqual(outputs['health_score'].shape, (batch_size, 1))
        self.assertEqual(outputs['growth_stage'].shape, (batch_size, 4))
    
    def test_output_ranges(self):
        """Test des plages de valeurs des sorties"""
        input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        outputs = self.model.predict(input_data, verbose=0)
        
        # is_plant: sigmoid -> [0, 1]
        self.assertTrue(np.all(outputs['is_plant'] >= 0))
        self.assertTrue(np.all(outputs['is_plant'] <= 1))
        
        # classification: softmax -> somme = 1
        self.assertAlmostEqual(np.sum(outputs['classification']), 1.0, places=5)
        
        # health_score: sigmoid -> [0, 1]
        self.assertTrue(np.all(outputs['health_score'] >= 0))
        self.assertTrue(np.all(outputs['health_score'] <= 1))
        
        # growth_stage: softmax -> somme = 1
        self.assertAlmostEqual(np.sum(outputs['growth_stage']), 1.0, places=5)
    
    def test_model_trainable_params(self):
        """Test du nombre de paramètres entraînables"""
        trainable_params = np.sum([
            np.prod(v.get_shape()) 
            for v in self.model.trainable_weights
        ])
        
        # Le modèle devrait avoir au moins 1M de paramètres entraînables
        self.assertGreater(trainable_params, 1_000_000)
        
        # Mais pas trop (pour Raspberry Pi)
        self.assertLess(trainable_params, 50_000_000)


class TestModelSaving(unittest.TestCase):
    """Tests de sauvegarde et chargement du modèle"""
    
    def setUp(self):
        """Créer un modèle pour les tests"""
        from src.model import PlantClassificationModel
        self.model_builder = PlantClassificationModel(num_classes=10)
        self.model = self.model_builder.build()
        self.test_dir = Path('/tmp/test_models')
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Nettoyer les fichiers de test"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_save_keras(self):
        """Test de sauvegarde au format Keras"""
        save_path = self.test_dir / 'test_model.keras'
        self.model.save(save_path)
        self.assertTrue(save_path.exists())
    
    def test_load_keras(self):
        """Test de chargement du format Keras"""
        save_path = self.test_dir / 'test_model.keras'
        self.model.save(save_path)
        
        loaded_model = tf.keras.models.load_model(save_path)
        self.assertIsNotNone(loaded_model)
        
        # Vérifier que les prédictions sont identiques
        input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        orig_pred = self.model.predict(input_data, verbose=0)
        loaded_pred = loaded_model.predict(input_data, verbose=0)
        
        for key in orig_pred:
            np.testing.assert_array_almost_equal(orig_pred[key], loaded_pred[key])


class TestTFLiteConversion(unittest.TestCase):
    """Tests de conversion TensorFlow Lite"""
    
    @classmethod
    def setUpClass(cls):
        """Créer et convertir le modèle"""
        from src.model import PlantClassificationModel
        cls.model_builder = PlantClassificationModel(num_classes=10)
        cls.model = cls.model_builder.build()
        
        # Convertir en TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(cls.model)
        cls.tflite_model = converter.convert()
    
    def test_tflite_conversion(self):
        """Test de la conversion TFLite"""
        self.assertIsNotNone(self.tflite_model)
        self.assertGreater(len(self.tflite_model), 0)
    
    def test_tflite_inference(self):
        """Test de l'inférence TFLite"""
        # Créer l'interpréteur
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        # Obtenir les détails
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Préparer l'entrée
        input_shape = input_details[0]['shape']
        input_data = np.random.rand(*input_shape).astype(np.float32)
        
        # Inférence
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Vérifier les sorties
        for detail in output_details:
            output = interpreter.get_tensor(detail['index'])
            self.assertIsNotNone(output)
    
    def test_tflite_quantized(self):
        """Test de la quantification TFLite"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        quantized_model = converter.convert()
        
        # Le modèle quantifié devrait être plus petit
        self.assertLess(len(quantized_model), len(self.tflite_model))


if __name__ == '__main__':
    unittest.main()