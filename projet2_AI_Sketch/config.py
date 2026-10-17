# config.py
import os

# Chemins des dossiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Paramètres du modèle et des images
IMAGE_SIZE = (28, 28)
MODEL_PATH = os.path.join(MODEL_DIR, "sketch_cnn_model.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")

# Liste des 8 catégories choisies pour le projet
CLASSES = [
    "cat", 
    "house", 
    "car", 
    "tree", 
    "sun", 
    "bicycle", 
    "apple", 
    "star"
]