"""
ce fichier joue trop role:
-Téléchargement des données brutes .npy.
-Extraction & Équilibrage (5000 images par classe).
-Normalisation (conversion des valeurs [0, 255] vers [0.0, 1.0]).
-Découpage Stratifié (70% Train, 15% Validation, 15% Test) et sauvegarde.
"""

import os
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split
from config import CLASSES, SAMPLES_PER_CLASS, RAW_DATA_DIR, PROCESSED_DATA_DIR, BASE_URL

def create_directories():
    """Crée les dossiers data/raw et data/processed s'ils n'existent pas."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def download_data():
    """Télécharge les fichiers .npy depuis Google Cloud Storage."""
    print("1. --- Téléchargement des fichiers .npy bruts ---")
    for category in CLASSES:
        file_path = os.path.join(RAW_DATA_DIR, f"{category}.npy")
        if not os.path.exists(file_path):
            url = f"{BASE_URL}{category}.npy"
            print(f"  Téléchargement de {category}...")
            urllib.request.urlretrieve(url, file_path)
        else:
            print(f"  {category}.npy déjà présent.")

def process_and_split_data():
    """Prétraite les données (équilibrage, normalisation) et les découpe."""
    print("\n2. --- Traitement et Équilibrage des données ---")
    
    x_data = []
    y_data = []

    for label_idx, category in enumerate(CLASSES):
        file_path = os.path.join(RAW_DATA_DIR, f"{category}.npy")
        data = np.load(file_path)
        
        # Sélection aléatoire de SAMPLES_PER_CLASS (ex: 5000 images)
        indices = np.random.choice(data.shape[0], SAMPLES_PER_CLASS, replace=False)
        selected_data = data[indices]
        
        # Labels sous forme d'entiers (0, 1, 2...)
        labels = np.full(SAMPLES_PER_CLASS, label_idx)
        
        x_data.append(selected_data)
        y_data.append(labels)
        print(f"  Class '{category}' (label {label_idx}) : {SAMPLES_PER_CLASS} images sélectionnées.")

    # Fusion des données
    X = np.concatenate(x_data, axis=0)
    y = np.concatenate(y_data, axis=0)

    # Reshape en (N, 28, 28, 1) pour l'entrée d'un CNN Keras / TensorFlow
    X = X.reshape(-1, 28, 28, 1)

    # Normalisation des pixels entre 0.0 et 1.0
    X = X.astype('float32') / 255.0

    print("\n3. --- Découpage Stratifié (Train / Val / Test) ---")
    # Premier découpage : 70% Train, 30% Reste (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Second découpage du reste : 15% Validation, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"  Taille Train      : {X_train.shape[0]} images ({y_train.shape[0]} labels)")
    print(f"  Taille Validation : {X_val.shape[0]} images ({y_val.shape[0]} labels)")
    print(f"  Taille Test       : {X_test.shape[0]} images ({y_test.shape[0]} labels)")

    print("\n4. --- Sauvegarde dans data/processed/ ---")
    output_path = os.path.join(PROCESSED_DATA_DIR, 'dataset.npz')
    np.savez_compressed(
        output_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print(f"✅ Jeu de données exporté avec succès vers : {output_path}")

if __name__ == '__main__':
    create_directories()
    download_data()
    process_and_split_data()