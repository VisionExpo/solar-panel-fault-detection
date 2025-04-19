import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from ..utils.logger import logger
from ..config.configuration import Config
import mlflow
import cv2
from typing import Tuple, Dict

class DataPreparation:
    def __init__(self, config: Config):
        self.config = config
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])

    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.config.model.img_size)
            
            # Apply augmentation
            augmented = self.transform(image=image)
            image = augmented['image']
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            return image
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Prepare train, validation and test datasets"""
        images = []
        labels = []
        label_to_index = {}
        
        # Collect all images and labels
        for idx, category in enumerate(sorted(os.listdir(self.config.data.data_dir))):
            label_to_index[category] = idx
            category_path = Path(self.config.data.data_dir) / category
            
            for img_path in category_path.glob("*.[jJ][pP][gG]"):
                processed_img = self.load_and_preprocess_image(str(img_path))
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(idx)
                    
            for img_path in category_path.glob("*.[jJ][pP][eE][gG]"):
                processed_img = self.load_and_preprocess_image(str(img_path))
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(idx)

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.config.data.val_ratio + self.config.data.test_ratio,
            random_state=self.config.data.random_state,
            stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.config.data.test_ratio / (self.config.data.val_ratio + self.config.data.test_ratio),
            random_state=self.config.data.random_state,
            stratify=y_temp
        )

        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(1000)\
            .batch(self.config.model.batch_size)\
            .prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
            .batch(self.config.model.batch_size)\
            .prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))\
            .batch(self.config.model.batch_size)\
            .prefetch(tf.data.AUTOTUNE)

        # Log dataset statistics
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Log to MLflow
        mlflow.log_params({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "num_classes": len(label_to_index)
        })

        return train_ds, val_ds, test_ds, label_to_index