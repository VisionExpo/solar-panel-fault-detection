import tensorflow as tf
from typing import List, Optional


def focal_loss(
    gamma: float = 2.0,
    alpha: Optional[List[float]] = None,
):
    """
    Focal loss for multi-class classification.

    Args:
        gamma: focusing parameter
        alpha: optional class weights

    Returns:
        Callable loss function
    """

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)

        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            cross_entropy *= alpha_tensor

        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    return loss_fn


def weighted_categorical_crossentropy(class_weights: List[float]):
    """
    Weighted categorical cross-entropy loss.

    Args:
        class_weights: list of weights per class

    Returns:
        Callable loss function
    """

    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss = -y_true * tf.math.log(y_pred) * class_weights_tensor
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    return loss_fn
