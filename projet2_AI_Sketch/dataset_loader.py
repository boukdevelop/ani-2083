"""
ce script permet à à ton coéquipier (Membre 1) d'importer le dataset prétraité dans son code d'entraînement en une seule ligne.
"""

import os
import numpy as np
from config import PROCESSED_DATA_DIR, CLASSES

def load_processed_data():
    """Charge le jeu de données .npz prêt pour TensorFlow/Keras."""
    file_path = os.path.join(PROCESSED_DATA_DIR, 'dataset.npz')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Le fichier {file_path} n'existe pas. "
            "Veuillez exécuter 'python prepare_data.py' d'abord."
        )
    
    data = np.load(file_path)
    return (
        (data['X_train'], data['y_train']),
        (data['X_val'], data['y_val']),
        (data['X_test'], data['y_test']),
        CLASSES
    )

if __name__ == '__main__':
    # Test d'importation
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), labels = load_processed_data()
    print(f"Chargement réussi ! Train: {X_tr.shape}, Classes: {labels}")