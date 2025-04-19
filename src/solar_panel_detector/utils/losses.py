import tensorflow as tf
import tensorflow.keras.backend as K

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss implementation for multi-class classification.
    Focal Loss is designed to address class imbalance by down-weighting easy examples
    and focusing more on hard examples.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (float): Weighting factor for the rare class, typically between 0-1.
            gamma (float): Focusing parameter. Higher values give more weight to hard examples.
            from_logits (bool): Whether the predictions are logits or probabilities.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        """
        Calculate focal loss.
        
        Args:
            y_true: Ground truth labels (sparse, not one-hot encoded)
            y_pred: Model predictions
            
        Returns:
            Focal loss value
        """
        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
        
        # Apply softmax if predictions are logits
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            
        # Clip predictions to avoid numerical instability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * K.log(y_pred)
        
        # Apply the focusing term
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy
        
        # Sum over classes and average over batch
        return K.mean(K.sum(loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


def weighted_categorical_crossentropy(class_weights):
    """
    Weighted categorical crossentropy for imbalanced datasets.
    
    Args:
        class_weights: Dictionary mapping class indices to weights
        
    Returns:
        Weighted loss function
    """
    def loss(y_true, y_pred):
        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
        
        # Apply softmax if predictions are logits
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate weighted cross entropy
        loss = y_true * K.log(y_pred) * class_weights
        loss = -K.sum(loss, axis=-1)
        return K.mean(loss)
    
    return loss
