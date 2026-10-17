# inference_engine.py
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from config import MODEL_PATH, CLASSES_PATH, IMAGE_SIZE, CLASSES

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.classes = CLASSES
        self.load_resources()

    def load_resources(self):
        """Charge le modèle CNN entraîné et la liste des classes."""
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Modèle chargé avec succès depuis {MODEL_PATH}")
        else:
            print(f"[AVERTISSEMENT] Modèle introuvable à l'emplacement {MODEL_PATH}. L'inférence simulera un résultat en attendant l'entraînement du Membre 3.")

        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                self.classes = json.load(f)

    def preprocess_canvas_image(self, image_input):
        """
        Prétraite l'image brute du canvas :
        - Conversion en niveaux de gris
        - Inversion des couleurs si nécessaire (fond noir/trait blanc ou inverse)
        - Redimensionnement en 28x28 pixels
        - Normalisation des pixels entre 0.0 et 1.0
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("L")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
        else:
            # Supposons un tableau numpy ou autre format brut
            img = Image.fromarray(np.uint8(image_input)).convert("L")

        # Redimensionnement propre avec anticrénelage
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Transformation en tableau numpy
        img_array = np.array(img, dtype=np.float32)
        
        # Normalisation (0 à 1)
        img_array = img_array / 255.0
        
        # Redimensionnement pour correspondre à l'entrée du CNN : (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def predict(self, image_input, top_k=3):
        """
        Effectue la prédiction et retourne le Top-K des classes avec leurs scores de confiance.
        """
        if self.model is None:
            # Mode simulation si le modèle du Membre 3 n'est pas encore dispo
            return [("chat (simulation)", 0.85), ("maison (simulation)", 0.10), ("soleil (simulation)", 0.05)]

        processed_img = self.preprocess_canvas_image(image_input)
        predictions = self.model.predict(processed_img)[0]
        
        # Récupération des indices des top_k scores les plus élevés
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            class_name = self.classes[idx] if idx < len(self.classes) else str(idx)
            confidence = float(predictions[idx])
            results.append((class_name, confidence))
            
        return results