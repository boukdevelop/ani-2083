# inference_engine.py
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
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
        1. Conversion en niveaux de gris
        2. Inversion (fond noir, traits blancs) conforme au dataset Quick, Draw!
        3. Recadrage centré via Bounding Box avec marge de sécurité (padding)
        4. Redimensionnement en 28x28 pixels avec cv2.INTER_AREA
        5. Normalisation des pixels entre 0.0 et 1.0 au format (1, 28, 28, 1)
        """
        # Conversion d'entrée vers un tableau numpy 2D (niveaux de gris)
        if isinstance(image_input, np.ndarray):
            img_np = image_input
            if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
                gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
            elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
                gray = img_np[:, :, 0]
            else:
                gray = img_np.copy()
        elif isinstance(image_input, (str, Image.Image)):
            if isinstance(image_input, str):
                img_pil = Image.open(image_input).convert("L")
            else:
                img_pil = image_input.convert("L")
            gray = np.array(img_pil, dtype=np.uint8)
        else:
            img_pil = Image.fromarray(np.uint8(image_input)).convert("L")
            gray = np.array(img_pil, dtype=np.uint8)

        # 1. Inversion de polarité :
        # Le canvas de dessin est blanc (255) avec trait noir (0).
        # Quick, Draw! a un fond noir (0) avec trait blanc (255).
        if np.mean(gray) > 127:
            inv = 255 - gray
        else:
            inv = gray

        # 2. Gestion du canvas vide (aucun trait dessiné)
        coords = cv2.findNonZero(inv)
        if coords is None:
            return np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=np.float32)

        # 3. Extraction de la boîte englobante (Bounding Box)
        x, y, w, h = cv2.boundingRect(coords)
        crop = inv[y:y+h, x:x+w]

        # 4. Rognage carré avec padding proportionnel (~20%) pour garder les ratios
        max_side = max(w, h)
        padding = int(max_side * 0.20)
        square_side = max_side + 2 * padding

        square = np.zeros((square_side, square_side), dtype=np.uint8)
        offset_x = padding + (max_side - w) // 2
        offset_y = padding + (max_side - h) // 2
        square[offset_y:offset_y+h, offset_x:offset_x+w] = crop

        # 5. Redimensionnement en (28, 28) avec cv2.INTER_AREA (anti-crénelage pour contours fins)
        resized = cv2.resize(square, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # 6. Normalisation [0.0, 1.0] et reshape (1, 28, 28, 1) pour le CNN Keras
        img_array = (resized.astype(np.float32) / 255.0).reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

        return img_array

    def predict(self, image_input, confidence_threshold=0.55):
        """
        Effectue la prédiction et retourne le résultat principal avec son score de confiance.
        Si la confiance est inférieure au seuil, retourne ('Inconnu / Pas sûr', score).
        """
        if self.model is None:
            return ("chat (simulation)", 0.85)

        processed_img = self.preprocess_canvas_image(image_input)

        if np.max(processed_img) == 0.0:
            return ("Canvas vide", 0.0)

        predictions = self.model.predict(processed_img, verbose=0)[0]

        # Récupération de la meilleure prédiction (1er résultat)
        best_idx = int(np.argmax(predictions))
        best_score = float(predictions[best_idx])
        best_class = self.classes[best_idx] if best_idx < len(self.classes) else str(best_idx)

        # Si la confiance est inférieure au seuil
        if best_score < confidence_threshold:
            return ("Inconnu / Pas sûr", best_score)

        return (best_class, best_score)