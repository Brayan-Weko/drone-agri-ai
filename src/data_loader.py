"""
Module de chargement et préparation des données
"""

import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
from loguru import logger
from tqdm import tqdm

from .config import MODEL_CONFIG, DATA_DIR


class PlantDataLoader:
    """Chargeur de données optimisé pour les images de plantes"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.input_size = MODEL_CONFIG["input_size"]
        self.batch_size = MODEL_CONFIG["batch_size"]
        self.class_names = []
        self.class_to_idx = {}
        
        # Augmentation pour l'entraînement
        self.train_transform = A.Compose([
            A.RandomResizedCrop(
                height=self.input_size[0], 
                width=self.input_size[1], 
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05),
                A.GridDistortion(distort_limit=0.05),
            ], p=0.2),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Transformation pour validation/test
        self.val_transform = A.Compose([
            A.Resize(self.input_size[0], self.input_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def discover_classes(self, split: str = "train") -> List[str]:
        """Découvre automatiquement les classes depuis le dossier"""
        split_dir = self.data_dir / split
        if not split_dir.exists():
            logger.warning(f"Dossier {split_dir} non trouvé")
            return []
        
        self.class_names = sorted([
            d.name for d in split_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        logger.info(f"Découvert {len(self.class_names)} classes")
        return self.class_names
    
    def parse_class_info(self, class_name: str) -> Dict:
        """Parse le nom de classe pour extraire plante et état"""
        # Format attendu: "Plant___Disease" ou "Plant___healthy"
        parts = class_name.split("___")
        if len(parts) == 2:
            return {
                "plant": parts[0].replace("_", " "),
                "condition": parts[1].replace("_", " "),
                "is_healthy": "healthy" in parts[1].lower()
            }
        return {
            "plant": class_name,
            "condition": "unknown",
            "is_healthy": None
        }
    
    def create_tf_dataset(
        self, 
        split: str = "train",
        shuffle: bool = True,
        augment: bool = True
    ) -> tf.data.Dataset:
        """Crée un tf.data.Dataset optimisé"""
        
        split_dir = self.data_dir / split
        
        if not self.class_names:
            self.discover_classes(split)
        
        # Collecter tous les chemins d'images
        image_paths = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(str(img_path))
                    labels.append(self.class_to_idx[class_name])
        
        logger.info(f"Chargé {len(image_paths)} images pour {split}")
        
        # Créer le dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        
        # Fonction de chargement et prétraitement
        def load_and_preprocess(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, self.input_size)
            img = tf.cast(img, tf.float32) / 255.0
            
            # Normalisation ImageNet
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            return img, label
        
        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_sample_images(self, num_samples: int = 5) -> List[Tuple[np.ndarray, str]]:
        """Récupère des images échantillons pour visualisation"""
        samples = []
        split_dir = self.data_dir / "train"
        
        for class_name in self.class_names[:num_samples]:
            class_dir = split_dir / class_name
            if class_dir.exists():
                img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                if img_files:
                    img = cv2.imread(str(img_files[0]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    samples.append((img, class_name))
        
        return samples
    
    def save_class_mapping(self, output_path: str = None):
        """Sauvegarde le mapping des classes"""
        if output_path is None:
            output_path = self.data_dir / "class_mapping.json"
        
        mapping = {
            "class_names": self.class_names,
            "class_to_idx": self.class_to_idx,
            "class_info": {
                name: self.parse_class_info(name) 
                for name in self.class_names
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Mapping sauvegardé: {output_path}")


class ImagePreprocessor:
    """Préprocesseur d'images pour l'inférence"""
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Prétraite une image pour l'inférence"""
        # Redimensionner
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, self.input_size)
        
        # Convertir BGR -> RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image[0, 0, 0] > image[0, 0, 2]:  # Heuristique BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normaliser
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Ajouter dimension batch
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Prétraite un batch d'images"""
        processed = [self.preprocess(img)[0] for img in images]
        return np.stack(processed, axis=0)