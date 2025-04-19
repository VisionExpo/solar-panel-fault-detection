import os
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
        # Enhanced augmentation pipeline with more aggressive transformations
        # and solar panel specific augmentations
        self.transform = A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),

            # Noise and blur - simulate camera quality issues
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.3),

            # Motion and focus issues
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.Blur(blur_limit=5),
                A.ZoomBlur(max_factor=1.5, p=0.5),
            ], p=0.3),

            # Distortions - simulate lens effects
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1),
                A.GridDistortion(distort_limit=0.3),
                A.ElasticTransform(alpha=1, sigma=50),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
            ], p=0.3),

            # Color adjustments - simulate lighting conditions
            A.OneOf([
                A.CLAHE(clip_limit=4, p=0.7),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7)),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),  # Simulate shadows on panels
                A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1,
                                num_flare_circles_lower=1, num_flare_circles_upper=3,
                                src_radius=100, src_color=(255, 255, 255), p=0.3),  # Simulate sun reflections
            ], p=0.5),

            # Color shifts - simulate different weather conditions
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                A.ChannelShuffle(p=0.2),  # More extreme color distortion
                A.ToSepia(p=0.2),  # Simulate aged/dusty appearance
            ], p=0.5),

            # Weather simulations
            A.OneOf([
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.3),  # Rain effect
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.2),  # Snow effect
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1, p=0.2),  # Fog effect
            ], p=0.3),

            # Compression and quality degradation
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),  # JPEG compression artifacts
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

    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
        """Prepare train, validation and test datasets with balanced sampling"""
        images = []
        labels = []
        label_to_index = {}
        category_counts = {}

        # Collect all images and labels
        for idx, category in enumerate(sorted(os.listdir(self.config.data.data_dir))):
            label_to_index[category] = idx
            category_path = Path(self.config.data.data_dir) / category
            category_images = []

            # Process JPG files
            for img_path in category_path.glob("*.[jJ][pP][gG]"):
                processed_img = self.load_and_preprocess_image(str(img_path))
                if processed_img is not None:
                    category_images.append(processed_img)

            # Process JPEG files
            for img_path in category_path.glob("*.[jJ][pP][eE][gG]"):
                processed_img = self.load_and_preprocess_image(str(img_path))
                if processed_img is not None:
                    category_images.append(processed_img)

            # Store category statistics
            category_counts[category] = len(category_images)

            # Add to main lists
            images.extend(category_images)
            labels.extend([idx] * len(category_images))

            logger.info(f"Category {category}: {len(category_images)} images")

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Split data with stratification to maintain class distribution
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

        # Create additional augmented samples for underrepresented classes in training set
        X_train_balanced, y_train_balanced = self._balance_classes(X_train, y_train)

        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_balanced, y_train_balanced))
        train_ds = train_ds.shuffle(10000)  # Increased shuffle buffer
        train_ds = train_ds.batch(self.config.model.batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(self.config.model.batch_size)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(self.config.model.batch_size)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        # Log dataset statistics
        logger.info(f"Original training set size: {len(X_train)}")
        logger.info(f"Balanced training set size: {len(X_train_balanced)}")
        logger.info(f"Validation set size: {len(X_val)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Category distribution: {category_counts}")

        # Log to MLflow
        mlflow.log_params({
            "original_train_size": len(X_train),
            "balanced_train_size": len(X_train_balanced),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "num_classes": len(label_to_index)
        })

        # Log category counts
        for category, count in category_counts.items():
            mlflow.log_param(f"category_{category}_count", count)

        return train_ds, val_ds, test_ds, label_to_index

    def _balance_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes by augmenting underrepresented classes"""
        # Count samples per class
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_samples = np.max(class_counts)

        logger.info(f"Class counts before balancing: {dict(zip(unique_classes, class_counts))}")

        # Create balanced dataset
        X_balanced = []
        y_balanced = []

        for cls in unique_classes:
            # Get samples for this class
            cls_indices = np.where(y == cls)[0]
            cls_samples = X[cls_indices]

            # If we have enough samples, just use them
            if len(cls_samples) >= max_samples:
                selected_indices = np.random.choice(len(cls_samples), max_samples, replace=False)
                X_balanced.extend(cls_samples[selected_indices])
                y_balanced.extend([cls] * max_samples)
            else:
                # Use all original samples
                X_balanced.extend(cls_samples)
                y_balanced.extend([cls] * len(cls_samples))

                # Generate additional augmented samples
                samples_needed = max_samples - len(cls_samples)

                # Apply more aggressive augmentation to create new samples
                augmented_samples = []
                while len(augmented_samples) < samples_needed:
                    # Randomly select a sample to augment
                    idx = np.random.randint(0, len(cls_samples))
                    img = cls_samples[idx].copy()

                    # Convert to uint8 for augmentation
                    img_uint8 = (img * 255).astype(np.uint8)

                    # Apply augmentation
                    augmented = self.transform(image=img_uint8)
                    aug_img = augmented['image'].astype(np.float32) / 255.0

                    augmented_samples.append(aug_img)

                X_balanced.extend(augmented_samples)
                y_balanced.extend([cls] * len(augmented_samples))

        # Convert to numpy arrays and shuffle
        X_balanced = np.array(X_balanced)
        y_balanced = np.array(y_balanced)

        # Shuffle the balanced dataset
        indices = np.arange(len(X_balanced))
        np.random.shuffle(indices)
        X_balanced = X_balanced[indices]
        y_balanced = y_balanced[indices]

        logger.info(f"Class counts after balancing: {dict(zip(unique_classes, np.unique(y_balanced, return_counts=True)[1]))}")

        return X_balanced, y_balanced