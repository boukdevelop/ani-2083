"""
config.py permet de centraliser les paramètres pour éviter d'avoir des valeurs "en dur" répétées partout.
"""

import os

# Nombre d'échantillons par classe pour garantir l'équilibrage
SAMPLES_PER_CLASS = 5000 

# Dossiers de stockage
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

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

# URL de base de Google pour Quick, Draw!
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"