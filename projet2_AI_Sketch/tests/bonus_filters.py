import cv2
import numpy as np
import os
from datetime import datetime

class BonusFilters:
    @staticmethod
    def apply_sketch_effect(image):
        """Applique un effet esquisse au crayon (Bonus OpenCV)."""
        if image is None:
            raise ValueError("L'image fournie est invalide.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256.0)
        return sketch

    @staticmethod
    def apply_canny_edges(image, threshold1=100, threshold2=200):
        """Applique un filtre de contours Canny."""
        if image is None:
            raise ValueError("L'image fournie est invalide.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges

    @staticmethod
    def apply_cartoon_effect(image):
        """Applique un effet cartoon / peinture stylisée."""
        if image is None:
            raise ValueError("L'image fournie est invalide.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    @staticmethod
    def save_to_gallery(image, output_dir="static/gallery/"):
        """Enregistre le dessin réalisé dans la galerie d'images."""
        if image is None:
            raise ValueError("Impossible d'enregistrer une image vide.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = datetime.now().strftime("sketch_%Y%m%d_%H%M%S.png")
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath