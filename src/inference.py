"""
Module d'inférence optimisé pour temps réel
Supporte TensorFlow, TFLite, ONNX et Coral Edge TPU
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
from queue import Queue
import numpy as np
import cv2
from loguru import logger

# Imports conditionnels
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except ImportError:
    TFLITE_RUNTIME = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .config import MODEL_CONFIG, MODELS_DIR


@dataclass
class InferenceResult:
    """Résultat d'inférence"""
    outputs: Dict[str, np.ndarray]
    inference_time_ms: float
    preprocessing_time_ms: float
    total_time_ms: float
    backend: str


class BaseInferenceEngine(ABC):
    """Classe de base pour les moteurs d'inférence"""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (224, 224)):
        self.model_path = model_path
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    @abstractmethod
    def load_model(self):
        """Charge le modèle"""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Effectue l'inférence"""
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Prétraitement standard"""
        # Redimensionner
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)
        
        # BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normaliser
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Ajouter batch dimension
        return np.expand_dims(image, axis=0)
    
    def run(self, image: np.ndarray) -> InferenceResult:
        """Exécute le pipeline complet"""
        total_start = time.perf_counter()
        
        # Prétraitement
        preprocess_start = time.perf_counter()
        input_data = self.preprocess(image)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # Inférence
        inference_start = time.perf_counter()
        outputs = self.predict(input_data)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        return InferenceResult(
            outputs=outputs,
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocess_time,
            total_time_ms=total_time,
            backend=self.__class__.__name__
        )


class TFLiteEngine(BaseInferenceEngine):
    """Moteur d'inférence TensorFlow Lite"""
    
    def __init__(self, model_path: str, num_threads: int = 4, use_xnnpack: bool = True):
        super().__init__(model_path)
        self.num_threads = num_threads
        self.use_xnnpack = use_xnnpack
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle TFLite"""
        try:
            # Créer l'interpréteur
            if TFLITE_RUNTIME:
                self.interpreter = tflite.Interpreter(
                    model_path=self.model_path,
                    num_threads=self.num_threads
                )
            elif TF_AVAILABLE:
                self.interpreter = tf.lite.Interpreter(
                    model_path=self.model_path,
                    num_threads=self.num_threads
                )
            else:
                raise RuntimeError("Ni tflite_runtime ni tensorflow n'est disponible")
            
            # Allouer les tenseurs
            self.interpreter.allocate_tensors()
            
            # Récupérer les détails
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"TFLite model chargé: {self.model_path}")
            logger.info(f"Threads: {self.num_threads}")
            
        except Exception as e:
            logger.error(f"Erreur chargement TFLite: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Effectue l'inférence TFLite"""
        # Vérifier le type
        input_dtype = self.input_details[0]['dtype']
        if input_data.dtype != input_dtype:
            input_data = input_data.astype(input_dtype)
        
        # Définir l'entrée
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Inférence
        self.interpreter.invoke()
        
        # Récupérer les sorties
        outputs = {}
        for detail in self.output_details:
            name = detail['name'].split('/')[-1]  # Nom simplifié
            tensor = self.interpreter.get_tensor(detail['index'])
            outputs[name] = tensor
        
        return outputs


class CoralEngine(BaseInferenceEngine):
    """Moteur d'inférence pour Coral Edge TPU"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.interpreter = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle sur Edge TPU"""
        try:
            from pycoral.utils.edgetpu import make_interpreter
            from pycoral.adapters import common
            
            self.interpreter = make_interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"Coral Edge TPU model chargé: {self.model_path}")
            
        except ImportError:
            logger.error("pycoral non installé")
            raise
        except Exception as e:
            logger.error(f"Erreur chargement Coral: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Prétraitement pour Edge TPU (INT8)"""
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Pour Edge TPU, utiliser uint8
        return np.expand_dims(image.astype(np.uint8), axis=0)
    
    def predict(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Effectue l'inférence sur Edge TPU"""
        from pycoral.adapters import common
        
        common.set_input(self.interpreter, input_data)
        self.interpreter.invoke()
        
        outputs = {}
        for detail in self.output_details:
            name = detail['name'].split('/')[-1]
            tensor = self.interpreter.get_tensor(detail['index'])
            outputs[name] = tensor
        
        return outputs


class ONNXEngine(BaseInferenceEngine):
    """Moteur d'inférence ONNX Runtime"""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        super().__init__(model_path)
        self.providers = providers or ['CPUExecutionProvider']
        self.session = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle ONNX"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime non disponible")
        
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=self.providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"ONNX model chargé: {self.model_path}")
            logger.info(f"Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Erreur chargement ONNX: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Effectue l'inférence ONNX"""
        results = self.session.run(
            self.output_names,
            {self.input_name: input_data}
        )
        
        outputs = {}
        for name, result in zip(self.output_names, results):
            outputs[name.split('/')[-1]] = result
        
        return outputs


class BatchInferenceEngine:
    """Moteur d'inférence par batch pour traitement parallèle"""
    
    def __init__(self, engine: BaseInferenceEngine, batch_size: int = 8):
        self.engine = engine
        self.batch_size = batch_size
        self.queue = Queue(maxsize=batch_size * 2)
        self.results = {}
        self._lock = threading.Lock()
    
    def predict_batch(self, images: List[np.ndarray]) -> List[InferenceResult]:
        """Traite un batch d'images"""
        results = []
        
        # Traiter par lots
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # Prétraiter le batch
            preprocessed = np.concatenate([
                self.engine.preprocess(img) for img in batch
            ], axis=0)
            
            # Inférence
            start = time.perf_counter()
            outputs = self.engine.predict(preprocessed)
            inference_time = (time.perf_counter() - start) * 1000
            
            # Créer les résultats individuels
            for j in range(len(batch)):
                individual_outputs = {
                    k: v[j:j+1] for k, v in outputs.items()
                }
                results.append(InferenceResult(
                    outputs=individual_outputs,
                    inference_time_ms=inference_time / len(batch),
                    preprocessing_time_ms=0,
                    total_time_ms=inference_time / len(batch),
                    backend=f"{self.engine.__class__.__name__}_batch"
                ))
        
        return results


class AsyncInferenceEngine:
    """Moteur d'inférence asynchrone avec file d'attente"""
    
    def __init__(self, engine: BaseInferenceEngine, max_queue_size: int = 100):
        self.engine = engine
        self.input_queue = Queue(maxsize=max_queue_size)
        self.output_queue = Queue(maxsize=max_queue_size)
        self._running = False
        self._thread = None
    
    def start(self):
        """Démarre le worker d'inférence"""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Async inference worker démarré")
    
    def stop(self):
        """Arrête le worker"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Async inference worker arrêté")
    
    def _worker(self):
        """Worker de traitement"""
        while self._running:
            try:
                item = self.input_queue.get(timeout=0.1)
                if item is None:
                    continue
                
                request_id, image = item
                result = self.engine.run(image)
                self.output_queue.put((request_id, result))
                
            except Exception:
                pass
    
    def submit(self, request_id: str, image: np.ndarray):
        """Soumet une image pour inférence"""
        self.input_queue.put((request_id, image))
    
    def get_result(self, timeout: float = 1.0) -> Optional[Tuple[str, InferenceResult]]:
        """Récupère un résultat"""
        try:
            return self.output_queue.get(timeout=timeout)
        except:
            return None


def create_engine(
    model_path: str,
    backend: str = "auto",
    **kwargs
) -> BaseInferenceEngine:
    """
    Factory pour créer le moteur d'inférence approprié
    
    Args:
        model_path: Chemin vers le modèle
        backend: "tflite", "coral", "onnx", ou "auto"
        **kwargs: Arguments supplémentaires pour le moteur
    
    Returns:
        Instance de BaseInferenceEngine
    """
    
    model_path = str(model_path)
    
    if backend == "auto":
        # Détecter le backend basé sur l'extension
        if model_path.endswith('.tflite'):
            # Vérifier si Coral est disponible
            try:
                from pycoral.utils.edgetpu import list_edge_tpus
                if list_edge_tpus():
                    backend = "coral"
                else:
                    backend = "tflite"
            except ImportError:
                backend = "tflite"
        elif model_path.endswith('.onnx'):
            backend = "onnx"
        else:
            backend = "tflite"
    
    logger.info(f"Création du moteur: {backend}")
    
    if backend == "tflite":
        return TFLiteEngine(model_path, **kwargs)
    elif backend == "coral":
        return CoralEngine(model_path)
    elif backend == "onnx":
        return ONNXEngine(model_path, **kwargs)
    else:
        raise ValueError(f"Backend inconnu: {backend}")


class InferenceBenchmark:
    """Outil de benchmark pour comparer les performances"""
    
    def __init__(self, engine: BaseInferenceEngine):
        self.engine = engine
        self.results = []
    
    def run(self, num_iterations: int = 100, warmup: int = 10) -> Dict:
        """Exécute le benchmark"""
        
        # Image de test
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup
        logger.info(f"Warmup: {warmup} itérations...")
        for _ in range(warmup):
            self.engine.run(test_image)
        
        # Benchmark
        logger.info(f"Benchmark: {num_iterations} itérations...")
        times = []
        
        for i in range(num_iterations):
            result = self.engine.run(test_image)
            times.append(result.total_time_ms)
        
        times = np.array(times)
        
        stats = {
            "backend": self.engine.__class__.__name__,
            "iterations": num_iterations,
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000 / np.mean(times))
        }
        
        logger.info(f"Résultats: {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms ({stats['fps']:.1f} FPS)")
        
        return stats


# === Tests ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        engine = create_engine(model_path)
        
        # Benchmark
        benchmark = InferenceBenchmark(engine)
        stats = benchmark.run(num_iterations=50)
        
        print("\n=== Résultats du benchmark ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("Usage: python inference.py <model_path>")