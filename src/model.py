"""
Architecture du modèle de classification de plantes
Optimisé pour Raspberry Pi avec accélération Coral TPU optionnelle
"""

import os
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    MobileNetV3Small,
    MobileNetV3Large,
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
import tensorflow_model_optimization as tfmot
from loguru import logger

from .config import MODEL_CONFIG, MODELS_DIR


class PlantClassificationModel:
    """
    Modèle de classification de plantes avec architecture multi-tâches:
    1. Détection plante vs non-plante
    2. Identification de l'espèce
    3. Détection de maladie/santé
    4. Estimation du stade de croissance
    """
    
    def __init__(
        self,
        num_classes: int = MODEL_CONFIG["num_classes"],
        input_size: Tuple[int, int] = MODEL_CONFIG["input_size"],
        backbone: str = MODEL_CONFIG["backbone"]
    ):
        self.num_classes = num_classes
        self.input_size = input_size
        self.backbone_name = backbone
        self.model = None
        self.tflite_model = None
    
    def build(self, compile_model: bool = True) -> Model:
        """Construit le modèle avec architecture multi-sorties"""
        
        # Input
        inputs = layers.Input(shape=(*self.input_size, 3), name="input_image")
        
        # Backbone
        backbone = self._get_backbone()
        features = backbone(inputs)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name="gap")(features)
        
        # Couches partagées
        x = layers.Dense(512, activation='relu', name="shared_dense1")(x)
        x = layers.BatchNormalization(name="shared_bn1")(x)
        x = layers.Dropout(0.3, name="shared_dropout1")(x)
        
        x = layers.Dense(256, activation='relu', name="shared_dense2")(x)
        x = layers.BatchNormalization(name="shared_bn2")(x)
        x = layers.Dropout(0.3, name="shared_dropout2")(x)
        
        # === SORTIE 1: Détection plante ===
        is_plant = layers.Dense(64, activation='relu', name="plant_detect_dense")(x)
        is_plant = layers.Dense(1, activation='sigmoid', name="is_plant")(is_plant)
        
        # === SORTIE 2: Classification espèce/maladie ===
        classification = layers.Dense(128, activation='relu', name="class_dense")(x)
        classification = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name="classification"
        )(classification)
        
        # === SORTIE 3: Score de santé (0-1) ===
        health = layers.Dense(64, activation='relu', name="health_dense")(x)
        health = layers.Dense(1, activation='sigmoid', name="health_score")(health)
        
        # === SORTIE 4: Stade de croissance (4 classes) ===
        growth = layers.Dense(64, activation='relu', name="growth_dense")(x)
        growth = layers.Dense(4, activation='softmax', name="growth_stage")(growth)
        
        # Construire le modèle
        self.model = Model(
            inputs=inputs,
            outputs={
                "is_plant": is_plant,
                "classification": classification,
                "health_score": health,
                "growth_stage": growth
            },
            name="PlantAnalyzer"
        )
        
        if compile_model:
            self._compile()
        
        logger.info(f"Modèle construit: {self.model.count_params():,} paramètres")
        return self.model
    
    def _get_backbone(self) -> Model:
        """Retourne le backbone pré-entraîné"""
        
        backbone_configs = {
            "efficientnet_b0": (EfficientNetB0, {"weights": "imagenet", "include_top": False}),
            "efficientnet_b1": (EfficientNetB1, {"weights": "imagenet", "include_top": False}),
            "mobilenet_v3_small": (MobileNetV3Small, {"weights": "imagenet", "include_top": False}),
            "mobilenet_v3_large": (MobileNetV3Large, {"weights": "imagenet", "include_top": False}),
        }
        
        if self.backbone_name not in backbone_configs:
            raise ValueError(f"Backbone inconnu: {self.backbone_name}")
        
        BackboneClass, kwargs = backbone_configs[self.backbone_name]
        backbone = BackboneClass(
            input_shape=(*self.input_size, 3),
            **kwargs
        )
        
        # Geler les premières couches pour transfer learning
        for layer in backbone.layers[:int(len(backbone.layers) * 0.7)]:
            layer.trainable = False
        
        logger.info(f"Backbone: {self.backbone_name}")
        return backbone
    
    def _compile(self):
        """Compile le modèle avec les losses appropriées"""
        
        self.model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=MODEL_CONFIG["learning_rate"],
                weight_decay=1e-5
            ),
            loss={
                "is_plant": "binary_crossentropy",
                "classification": "sparse_categorical_crossentropy",
                "health_score": "mse",
                "growth_stage": "sparse_categorical_crossentropy"
            },
            loss_weights={
                "is_plant": 1.0,
                "classification": 2.0,  # Plus important
                "health_score": 0.5,
                "growth_stage": 1.0
            },
            metrics={
                "is_plant": ["accuracy"],
                "classification": ["accuracy", "top_k_categorical_accuracy"],
                "health_score": ["mae"],
                "growth_stage": ["accuracy"]
            }
        )
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = MODEL_CONFIG["epochs"],
        callbacks: List = None
    ):
        """Entraîne le modèle"""
        
        if self.model is None:
            self.build()
        
        # Callbacks par défaut
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _get_default_callbacks(self) -> List:
        """Retourne les callbacks par défaut"""
        
        return [
            ModelCheckpoint(
                filepath=str(MODELS_DIR / "best_model.keras"),
                monitor="val_classification_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(MODELS_DIR / "logs"),
                histogram_freq=1
            )
        ]
    
    def save(self, path: str = None):
        """Sauvegarde le modèle"""
        if path is None:
            path = MODELS_DIR / "plant_model.keras"
        
        self.model.save(path)
        logger.info(f"Modèle sauvegardé: {path}")
    
    def load(self, path: str = None):
        """Charge le modèle"""
        if path is None:
            path = MODELS_DIR / "plant_model.keras"
        
        self.model = keras.models.load_model(path)
        logger.info(f"Modèle chargé: {path}")
        return self.model
    
    def convert_to_tflite(
        self,
        quantize: bool = True,
        optimize_for_edge_tpu: bool = False
    ) -> bytes:
        """
        Convertit le modèle en TensorFlow Lite pour Raspberry Pi
        """
        
        if self.model is None:
            raise ValueError("Aucun modèle à convertir")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimisations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            # Quantification INT8 pour performance maximale
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32
            
            # Représentant dataset pour calibration
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, *self.input_size, 3).astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_dataset
        
        if optimize_for_edge_tpu:
            # Optimisations spécifiques Edge TPU (Coral)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
        
        self.tflite_model = converter.convert()
        
        # Sauvegarder
        tflite_path = MODELS_DIR / "plant_model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(self.tflite_model)
        
        logger.info(f"Modèle TFLite sauvegardé: {tflite_path}")
        logger.info(f"Taille: {len(self.tflite_model) / 1024 / 1024:.2f} MB")
        
        return self.tflite_model
    
    def apply_pruning(self, target_sparsity: float = 0.5):
        """
        Applique le pruning pour réduire la taille du modèle
        """
        
        if self.model is None:
            raise ValueError("Aucun modèle à optimiser")
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        self.model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            **pruning_params
        )
        
        self._compile()
        logger.info(f"Pruning appliqué: sparsité cible = {target_sparsity}")
        
        return self.model


class SimplifiedModel:
    """
    Version simplifiée pour classification basique
    (si le modèle multi-sorties est trop lourd)
    """
    
    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (224, 224)):
        self.num_classes = num_classes
        self.input_size = input_size
        self.model = None
    
    def build(self) -> Model:
        """Construit un modèle simple et léger"""
        
        base_model = MobileNetV3Small(
            input_shape=(*self.input_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        inputs = keras.Input(shape=(*self.input_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model