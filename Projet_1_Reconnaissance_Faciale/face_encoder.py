# ====================== DOCSTRING ===========================
# * Ce module permet de :
# * Analyser les différentes images;
# * Vectoriser les differentes images du dataset;
# * Générer un fichier .pkl contenant les vecteurs d'images.
# ============================================================

import os, pickle, time, datetime, pathlib
import face_recognition
import numpy

# Les constantes à réutiliser plus bas dans le module
ENCODING_DIM = 128 # la dimensions
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

DEFAULT_MODEL = "small" # Model d'encodage par défaut

DEFAULT_NUM_JITTERS = 1 # Nombre de jitter par défaut

DEFAULT_OUTPUT = "encoded_faces.pkl"

def build_encodings(dataset_path, output_path=DEFAULT_OUTPUT, model=DEFAULT_MODEL, num_jitters=DEFAULT_NUM_JITTERS):
    pass

def load_encodings(pkl_path=DEFAULT_OUTPUT):
    pass

def _iter_person_images(dataset_path):
    pass

def _encode_image(image_path, model, num_jitters):
    pass

if __name__ == "__main__":
    # Les appels de test ici
    print(f"Dimension d'encodage : {ENCODING_DIM}")