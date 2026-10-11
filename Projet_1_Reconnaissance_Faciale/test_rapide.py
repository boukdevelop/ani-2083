"""
test_rapide.py — Valide FaceMatcher SANS encoded_faces.pkl
"""
import pickle
import numpy as np
from pathlib import Path

# 1. Créer un faux encoded_faces.pkl avec 2 personnes
fake_data = {
    "encodings": [
        np.zeros(128),           # Alice (vecteur de zéros)
        np.ones(128) * 5,        # Bob (vecteur de 5)
    ],
    "names": ["Alice", "Bob"],
}

with open("fake_faces.pkl", "wb") as f:
    pickle.dump(fake_data, f)

print("✅ fake_faces.pkl créé")

# 2. Tester FaceMatcher avec ce faux pickle
from face_matcher import FaceMatcher

matcher = FaceMatcher("fake_faces.pkl", tolerance=0.50)
print("Stats :", matcher.stats())

# 3. Test 1 : encodage proche d'Alice
query_alice = np.zeros(128) + 0.05
nom, conf = matcher.match(query_alice)
print(f"Test 1 (proche Alice) → {nom} ({conf:.1f}%)")
# Attendu : Alice (95%)

# 4. Test 2 : encodage très loin de tout
query_unknown = np.ones(128) * 50
nom, conf = matcher.match(query_unknown)
print(f"Test 2 (inconnu)      → {nom} ({conf:.1f}%)")
# Attendu : Unknown / Inconnu (0.0%)