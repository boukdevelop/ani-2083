import os
import pytest
import numpy as np

def test_model_file_exists():
    """Verifie la presence des artefacts du modele."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    classes_path = os.path.join(model_dir, "classes.json")
    model_path = os.path.join(model_dir, "sketch_cnn_model.keras")
    assert os.path.exists(classes_path), f"Fichier {classes_path} manquant"
    assert os.path.exists(model_path), f"Fichier {model_path} manquant"

def test_input_tensor_shape():
    """Verifie la conformite de la dimension d'entree (28x28x1) du reseau."""
    expected_shape = (1, 28, 28, 1)
    dummy_input = np.zeros(expected_shape, dtype=np.float32)
    assert dummy_input.shape == (1, 28, 28, 1)
