import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import cv2
import os
import pathlib
import logging
from datetime import datetime
from typing import Optional, Union

logger = logging.getLogger("HybridClassifier")


class HybridCropClassifier:
    """
    A hybrid CNN-VNet model for crop classification.
    
    Attributes:
        img_size (int): The size of input images (assumed square).
        num_classes (int): The number of crop classes to classify.
        class_names (list): List of crop class names.
        model: The compiled TensorFlow/Keras model.
    """

    def __init__(self, img_size: int = 224, num_classes: int = 6):
        """
        Initialize the HybridCropClassifier.
        
        Parameters:
            img_size (int): The size of input images (default: 224).
            num_classes (int): The number of classes (default: 6).
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = ['bajra', 'castor', 'cotton', 'paddy', 'sugarcane', 'wheat']
        self.model = self._build_model()
        logger.info("HybridCropClassifier initialized with img_size=%d and num_classes=%d", img_size, num_classes)

    def _build_model(self) -> models.Model:
        """
        Build the hybrid CNN-VNet model architecture.
        
        Returns:
            models.Model: Compiled Keras model.
        """
        try:
            inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
            x = layers.Rescaling(1.0 / 255)(inputs)

            # CNN Branch
            cnn = self._create_cnn_branch(x)
            cnn_output = layers.GlobalAveragePooling2D()(cnn)

            # VNet Branch
            vnet = self._create_vnet_branch(x)
            vnet_output = layers.GlobalAveragePooling2D()(vnet)

            # Feature Fusion
            merged = layers.Concatenate()([cnn_output, vnet_output])
            x = layers.Dense(
                512,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            )(merged)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(
                256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            )(x)
            x = layers.BatchNormalization()(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)

            model = models.Model(inputs, outputs)
            logger.info("Model successfully built.")
            return model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    @staticmethod
    def _create_cnn_branch(input_layer: layers.Layer) -> layers.Layer:
        """
        Create the CNN branch of the hybrid model.
        
        Parameters:
            input_layer (layers.Layer): Input tensor for the CNN branch.
        
        Returns:
            layers.Layer: Output tensor of the CNN branch.
        """
        cnn = layers.Conv2D(64, 3, activation='relu', padding='same')(input_layer)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.MaxPooling2D(2)(cnn)

        cnn = layers.Conv2D(128, 3, activation='relu', padding='same')(cnn)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.MaxPooling2D(2)(cnn)

        cnn = layers.Conv2D(256, 3, activation='relu', padding='same')(cnn)
        cnn = layers.BatchNormalization()(cnn)
        return cnn

    @staticmethod
    def _create_vnet_branch(input_layer: layers.Layer) -> layers.Layer:
        """
        Create the VNet branch of the hybrid model.
        
        Parameters:
            input_layer (layers.Layer): Input tensor for the VNet branch.
        
        Returns:
            layers.Layer: Output tensor of the VNet branch.
        """
        vnet = layers.Conv2D(64, 3, padding='same', activation='relu')(input_layer)
        vnet = layers.BatchNormalization()(vnet)
        vnet = layers.MaxPooling2D(2)(vnet)

        vnet = layers.Conv2D(128, 3, padding='same', activation='relu')(vnet)
        vnet = layers.BatchNormalization()(vnet)
        vnet = layers.MaxPooling2D(2)(vnet)

        vnet = layers.Conv2D(256, 3, padding='same', activation='relu')(vnet)
        vnet = layers.BatchNormalization()(vnet)
        return vnet

    def train(self, train_dir: str, epochs: int = 50, batch_size: int = 32) -> Optional[dict]:
        """
        Train the model using the provided dataset.
        
        Parameters:
            train_dir (str): Path to the directory containing training data.
            epochs (int): Number of training epochs (default: 50).
            batch_size (int): Batch size for training (default: 32).
        
        Returns:
            dict: Training history if successful, None otherwise.
        """
        try:
            logger.info("Loading training and validation datasets...")
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(self.img_size, self.img_size),
                batch_size=batch_size
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(self.img_size, self.img_size),
                batch_size=batch_size
            )

            # Optimize dataset performance
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            checkpoint_path = os.path.join("models", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.TensorBoard(log_dir=f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            ]

            logger.info("Starting model training...")
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks
            )

            logger.info("Training completed successfully.")
            return history.history
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return None

    def predict(self, image: Union[np.ndarray, str, bytes]) -> np.ndarray:
        """
        Predict the class probabilities for a given image.
        
        Parameters:
            image (Union[np.ndarray, str, bytes]): Input image as a NumPy array, file path, or bytes object.
        
        Returns:
            np.ndarray: Array of class probabilities.
        """
        try:
            if isinstance(image, str):  # File path
                image = cv2.imread(image)
            elif isinstance(image, bytes):  # Bytes object
                image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            elif not isinstance(image, np.ndarray):
                raise ValueError("Input image must be a NumPy array, file path, or bytes object.")

            img = cv2.resize(image, (self.img_size, self.img_size))
            img = np.expand_dims(img / 255.0, axis=0)
            predictions = self.model.predict(img, verbose=0)[0]
            logger.info("Prediction completed.")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def classify(self, image: Union[np.ndarray, str, bytes]) -> dict:
        """
        Classify an image and return the predicted class and confidence scores.
        
        Parameters:
            image (Union[np.ndarray, str, bytes]): Input image as a NumPy array, file path, or bytes object.
        
        Returns:
            dict: Dictionary containing the predicted class and confidence scores.
        """
        predictions = self.predict(image)
        class_idx = np.argmax(predictions)
        predicted_class = self.class_names[class_idx]
        confidence = predictions[class_idx]
        logger.info(f"Classified as {predicted_class} with confidence {confidence:.2f}")
        return {
            "class": predicted_class,
            "confidence": float(confidence),
            "scores": {cls: float(score) for cls, score in zip(self.class_names, predictions)}
        }

    @classmethod
    def load_latest_model(cls) -> "HybridCropClassifier":
        """
        Load the latest trained model from the models directory.
        
        Returns:
            HybridCropClassifier: Instance of the classifier with the loaded model.
        """
        try:
            model_files = list(pathlib.Path("models").glob("*.keras"))
            if not model_files:
                raise FileNotFoundError("No trained models found in the models directory.")

            latest_model = max(model_files, key=os.path.getctime)
            logger.info(f"Loading latest model: {latest_model}")
            classifier = cls()
            classifier.model = tf.keras.models.load_model(latest_model)
            return classifier
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def model_confidence(self) -> float:
        """
        Calculate the confidence score of the model based on validation accuracy (placeholder logic).
        
        Returns:
            float: Confidence score between 0 and 1.
        """
        # Placeholder for actual confidence calculation (e.g., based on validation performance)
        return 0.95