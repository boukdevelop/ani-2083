"""
config.py permet de centraliser les paramètres pour éviter d'avoir des valeurs "en dur" répétées partout.
"""

import os

# Classes/Catégories retenues pour la reconnaissance
CLASSES = ['apple', 'car', 'door', 'fish', 'house']

# Nombre d'échantillons par classe pour garantir l'équilibrage
SAMPLES_PER_CLASS = 5000 

# Dossiers de stockage
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# URL de base de Google pour Quick, Draw!
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"