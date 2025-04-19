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
        # Create two different augmentation pipelines: standard and advanced
        # Standard augmentation for general training
        self.transform = A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),

            # Noise and blur - simulate camera quality issues
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ISONoise(p=0.5),
            ], p=0.3),

            # Motion and focus issues
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.Blur(blur_limit=5),
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
                A.RandomShadow(p=0.5),  # Simulate shadows on panels
                A.RandomSunFlare(p=0.3),  # Simulate sun reflections
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
                A.RandomRain(p=0.3),  # Rain effect
                A.RandomSnow(p=0.2),  # Snow effect
                A.RandomFog(p=0.2),  # Fog effect
            ], p=0.3),

            # Compression and quality degradation
            A.ImageCompression(p=0.3),  # JPEG compression artifacts

            # Resize to match model input size
            A.Resize(height=self.config.model.img_size[0], width=self.config.model.img_size[1]),

            # Normalize pixel values
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Advanced augmentation for underrepresented classes
        self.advanced_transform = A.Compose([
            # More aggressive geometric transformations
            A.RandomRotate90(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.Transpose(p=0.7),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.7),
            A.GridDistortion(p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),

            # Cutout and CoarseDropout for robustness
            A.OneOf([
                A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.7),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.7),
            ], p=0.5),

            # Heavy color and lighting adjustments
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.8),
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=50, p=0.8),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.8),
                A.ChannelShuffle(p=0.4),
            ], p=0.7),

            # Weather and environmental effects
            A.OneOf([
                A.RandomRain(p=0.5),
                A.RandomSnow(p=0.5),
                A.RandomFog(p=0.5),
                A.RandomSunFlare(p=0.5),
                A.RandomShadow(p=0.5),
            ], p=0.6),

            # Noise and blur
            A.OneOf([
                A.GaussNoise(p=0.6),
                A.MultiplicativeNoise(p=0.6),
                A.ISONoise(p=0.6),
                A.GaussianBlur(blur_limit=(3, 9), p=0.6),
                A.MotionBlur(blur_limit=7, p=0.6),
            ], p=0.6),

            # Compression artifacts
            A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),

            # Resize to match model input size
            A.Resize(height=self.config.model.img_size[0], width=self.config.model.img_size[1]),

            # Normalize pixel values
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image with ImageNet normalization"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None

            # Convert to RGB (from BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply standard augmentation (includes resize and normalization)
            augmented = self.transform(image=image)
            image = augmented['image']

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
        """Balance classes by augmenting underrepresented classes with advanced augmentation"""
        # Count samples per class
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_samples = np.max(class_counts)

        # Increase the target count for better representation
        target_samples = int(max_samples * 1.2)  # 20% more samples for all classes

        logger.info(f"Class counts before balancing: {dict(zip(unique_classes, class_counts))}")
        logger.info(f"Target samples per class: {target_samples}")

        # Create balanced dataset
        X_balanced = []
        y_balanced = []

        for cls in unique_classes:
            # Get samples for this class
            cls_indices = np.where(y == cls)[0]
            cls_samples = X[cls_indices]
            cls_count = len(cls_samples)

            # Use all original samples
            X_balanced.extend(cls_samples)
            y_balanced.extend([cls] * cls_count)

            # Generate additional augmented samples if needed
            if cls_count < target_samples:
                samples_needed = target_samples - cls_count
                logger.info(f"Generating {samples_needed} additional samples for class {cls}")

                # Apply advanced augmentation for underrepresented classes
                augmented_samples = []
                while len(augmented_samples) < samples_needed:
                    # Randomly select a sample to augment
                    idx = np.random.randint(0, cls_count)
                    img = cls_samples[idx].copy()

                    # Convert to uint8 for augmentation (if not already in that format)
                    if img.max() <= 1.0:
                        img_uint8 = (img * 255).astype(np.uint8)
                    else:
                        img_uint8 = img.astype(np.uint8)

                    # Apply advanced augmentation for more diversity
                    augmented = self.advanced_transform(image=img_uint8)
                    aug_img = augmented['image']

                    # No need to convert back to float32 and divide by 255 since Normalize is applied in the transform
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