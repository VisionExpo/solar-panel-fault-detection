import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetV2S, ResNet50V2, 
    DenseNet121, MobileNetV3Large
)
from ..config.configuration import Config
from ..utils.losses import FocalLoss, weighted_categorical_crossentropy
from ..utils.logger import logger

class ModelFactory:
    """
    Factory class to create different model architectures for ensemble learning.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
    def create_model(self, model_type='efficientnetv2s'):
        """
        Create a model based on the specified type.
        
        Args:
            model_type (str): Type of model to create. Options:
                - 'efficientnetv2s': EfficientNetV2S
                - 'efficientnetb0': EfficientNetB0
                - 'resnet50v2': ResNet50V2
                - 'densenet121': DenseNet121
                - 'mobilenetv3': MobileNetV3Large
                
        Returns:
            tf.keras.Model: Compiled model
        """
        model_type = model_type.lower()
        
        if model_type == 'efficientnetv2s':
            return self._build_efficientnetv2s()
        elif model_type == 'efficientnetb0':
            return self._build_efficientnetb0()
        elif model_type == 'resnet50v2':
            return self._build_resnet50v2()
        elif model_type == 'densenet121':
            return self._build_densenet121()
        elif model_type == 'mobilenetv3':
            return self._build_mobilenetv3()
        else:
            logger.warning(f"Unknown model type: {model_type}. Using EfficientNetV2S as default.")
            return self._build_efficientnetv2s()
    
    def _get_optimizer(self):
        """Get optimizer with learning rate schedule"""
        initial_learning_rate = self.config.model.learning_rate
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-5
        )
        
        return tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    
    def _compile_model(self, model):
        """Compile model with focal loss and metrics"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        model.compile(
            optimizer=self._get_optimizer(),
            loss=focal_loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def _build_efficientnetv2s(self):
        """Build EfficientNetV2S model"""
        base_model = EfficientNetV2S(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.model.img_size[0],
                        self.config.model.img_size[1],
                        self.config.model.num_channels)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        # Build model
        inputs = tf.keras.Input(shape=(self.config.model.img_size[0],
                                      self.config.model.img_size[1],
                                      self.config.model.num_channels))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.config.model.num_classes, activation='softmax',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        
        model = tf.keras.Model(inputs, outputs, name="EfficientNetV2S")
        
        return self._compile_model(model)
    
    def _build_efficientnetb0(self):
        """Build EfficientNetB0 model"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.model.img_size[0],
                        self.config.model.img_size[1],
                        self.config.model.num_channels)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        # Build model
        inputs = tf.keras.Input(shape=(self.config.model.img_size[0],
                                      self.config.model.img_size[1],
                                      self.config.model.num_channels))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.config.model.num_classes, activation='softmax',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        
        model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
        
        return self._compile_model(model)
    
    def _build_resnet50v2(self):
        """Build ResNet50V2 model"""
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.model.img_size[0],
                        self.config.model.img_size[1],
                        self.config.model.num_channels)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        # Build model
        inputs = tf.keras.Input(shape=(self.config.model.img_size[0],
                                      self.config.model.img_size[1],
                                      self.config.model.num_channels))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.config.model.num_classes, activation='softmax',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        
        model = tf.keras.Model(inputs, outputs, name="ResNet50V2")
        
        return self._compile_model(model)
    
    def _build_densenet121(self):
        """Build DenseNet121 model"""
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.model.img_size[0],
                        self.config.model.img_size[1],
                        self.config.model.num_channels)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
            
        # Build model
        inputs = tf.keras.Input(shape=(self.config.model.img_size[0],
                                      self.config.model.img_size[1],
                                      self.config.model.num_channels))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(768, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(384, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.config.model.num_classes, activation='softmax',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        
        model = tf.keras.Model(inputs, outputs, name="DenseNet121")
        
        return self._compile_model(model)
    
    def _build_mobilenetv3(self):
        """Build MobileNetV3Large model"""
        base_model = MobileNetV3Large(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.model.img_size[0],
                        self.config.model.img_size[1],
                        self.config.model.num_channels)
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        # Build model
        inputs = tf.keras.Input(shape=(self.config.model.img_size[0],
                                      self.config.model.img_size[1],
                                      self.config.model.num_channels))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.config.model.num_classes, activation='softmax',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        
        model = tf.keras.Model(inputs, outputs, name="MobileNetV3Large")
        
        return self._compile_model(model)


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models.
    """
    
    def __init__(self, models, weights=None):
        """
        Initialize ensemble model.
        
        Args:
            models (list): List of compiled models
            weights (list, optional): List of weights for each model. If None, equal weights are used.
        """
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        # Ensure weights and models have the same length
        assert len(self.models) == len(self.weights), "Number of models and weights must match"
    
    def predict(self, x):
        """
        Make predictions using the ensemble.
        
        Args:
            x: Input data
            
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(x)
            predictions.append(pred * weight)
            
        # Sum weighted predictions
        ensemble_pred = sum(predictions)
        
        return ensemble_pred
    
    def save(self, path):
        """
        Save all models in the ensemble.
        
        Args:
            path (str): Base path to save models
        """
        for i, model in enumerate(self.models):
            model_path = f"{path}/model_{i}"
            model.save(model_path)
            
        # Save weights
        import json
        with open(f"{path}/ensemble_weights.json", 'w') as f:
            json.dump({"weights": self.weights}, f)
    
    @classmethod
    def load(cls, path, num_models):
        """
        Load ensemble model from saved models.
        
        Args:
            path (str): Base path where models are saved
            num_models (int): Number of models in the ensemble
            
        Returns:
            EnsembleModel: Loaded ensemble model
        """
        models = []
        
        for i in range(num_models):
            model_path = f"{path}/model_{i}"
            model = tf.keras.models.load_model(model_path, compile=False)
            models.append(model)
            
        # Load weights
        import json
        with open(f"{path}/ensemble_weights.json", 'r') as f:
            weights_data = json.load(f)
            weights = weights_data["weights"]
            
        return cls(models, weights)
