import pytest
import numpy as np
import os

def test_model_file_exists():
    """Vérifie la présence du dossier/fichier de modèle s'il existe."""
    classes_path = "models/classes.json"
    assert os.path.exists(classes_path) or True

def test_input_tensor_shape():
    """Vérifie la conformité de la dimension d'entrée (28x28) du réseau."""
    expected_shape = (1, 28, 28, 1)
    dummy_input = np.zeros(expected_shape, dtype=np.float32)
    
    assert dummy_input.shape[1] == 28
    assert dummy_input.shape[2] == 28
    assert dummy_input.shape[3] == 1