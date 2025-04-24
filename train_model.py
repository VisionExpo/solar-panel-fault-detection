import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Using GPU: {physical_devices}")
else:
    print("No GPU found, using CPU")

def create_model(input_shape=(384, 384, 3), num_classes=6):
    """
    Create a model based on EfficientNetB3 with custom top layers.
    
    Args:
        input_shape: Input shape of the images
        num_classes: Number of classes to predict
        
    Returns:
        A compiled Keras model
    """
    # Create base model
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def create_data_generators(train_dir, val_dir, test_dir, batch_size=32, img_size=(384, 384)):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        test_dir: Directory containing test data
        batch_size: Batch size for training
        img_size: Size to resize images to
        
    Returns:
        Training, validation, and test data generators
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def unfreeze_model(model, base_model, num_layers_to_unfreeze=30):
    """
    Unfreeze the last few layers of the base model for fine-tuning.
    
    Args:
        model: The full model
        base_model: The base model (EfficientNetB3)
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
        
    Returns:
        The model with unfrozen layers
    """
    # Unfreeze the last few layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_training_plots(history, output_dir):
    """
    Save training history plots.
    
    Args:
        history: Training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Learning rate plot if available
    if 'lr' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
        plt.close()
    
    # Save history as JSON
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history.history, f)
    
    # Save training summary
    best_val_acc_epoch = np.argmax(history.history['val_accuracy'])
    best_val_loss_epoch = np.argmin(history.history['val_loss'])
    
    summary = {
        'epochs': len(history.history['accuracy']),
        'best_val_acc': float(history.history['val_accuracy'][best_val_acc_epoch]),
        'best_val_acc_epoch': int(best_val_acc_epoch),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
        'best_val_loss': float(history.history['val_loss'][best_val_loss_epoch]),
        'best_val_loss_epoch': int(best_val_loss_epoch),
        'final_val_loss': float(history.history['val_loss'][-1])
    }
    
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f)

def main():
    # Directories
    data_dir = Path("data/solar_panels/organized")
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    test_dir = data_dir / "test"
    
    # Check if directories exist
    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        print("Dataset directories not found. Please run download_dataset.py first.")
        return 1
    
    # Output directories
    output_dir = Path("src/models")
    vis_dir = Path("visualization/static")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Training parameters
    batch_size = 32
    img_size = (384, 384)
    initial_epochs = 20
    fine_tuning_epochs = 10
    
    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator, test_generator = create_data_generators(
        train_dir, val_dir, test_dir, batch_size, img_size
    )
    
    # Save class indices
    class_indices = train_generator.class_indices
    label_mapping = {v: k for k, v in class_indices.items()}
    
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f)
    
    print(f"Class indices: {class_indices}")
    
    # Create model
    print("Creating model...")
    model, base_model = create_model(input_shape=img_size + (3,), num_classes=len(class_indices))
    model.summary()
    
    # Create callbacks
    checkpoint_path = output_dir / "best_model.h5"
    tensorboard_dir = output_dir / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir=str(tensorboard_dir))
    ]
    
    # Initial training with frozen base model
    print("Starting initial training...")
    history1 = model.fit(
        train_generator,
        epochs=initial_epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Fine-tuning
    print("Fine-tuning the model...")
    model = unfreeze_model(model, base_model)
    
    history2 = model.fit(
        train_generator,
        epochs=fine_tuning_epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        initial_epoch=len(history1.history['accuracy'])
    )
    
    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]
    
    class CombinedHistory:
        def __init__(self, history):
            self.history = history
    
    combined_history_obj = CombinedHistory(combined_history)
    
    # Save training plots
    print("Saving training plots...")
    save_training_plots(combined_history_obj, vis_dir)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save the final model
    print("Saving model...")
    model.save(output_dir / "final_model")
    
    # Save model summary
    with open(output_dir / "model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Save test results
    with open(vis_dir / "test_results.json", "w") as f:
        json.dump({
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss)
        }, f)
    
    print("Training completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
