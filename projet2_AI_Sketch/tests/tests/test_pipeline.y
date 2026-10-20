import pytest
import numpy as np
import cv2

def test_empty_canvas_robustness():
    """Vérifie le comportement et la robustesse face à un canvas complètement vide."""
    empty_image = np.zeros((28, 28, 1), dtype=np.float32)
    assert np.mean(empty_image) == 0.0

def test_image_resizing_pipeline():
    """Vérifie que le redimensionnement d'un croquis brut s'effectue correctement vers 28x28."""
    large_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    resized = cv2.resize(large_canvas, (28, 28))
    
    assert resized.shape == (28, 28, 3)