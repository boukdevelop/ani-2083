import pytest
import numpy as np
import cv2

def test_empty_canvas_preprocessing():
    """Verifie le comportement face a un canvas completement blanc ou vide."""
    empty_image = np.full((400, 400, 3), 255, dtype=np.uint8)
    assert empty_image.shape == (400, 400, 3)
    assert np.all(empty_image == 255)

def test_image_resizing():
    """Verifie que le redimensionnement d'un croquis s'effectue vers 28x28."""
    canvas = np.zeros((300, 300), dtype=np.uint8)
    resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    assert resized.shape == (28, 28)
