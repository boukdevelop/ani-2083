"""
face_matcher.py — Module de matching facial (Étudiant 3)
Projet 1 : Reconnaissance Faciale — ANI-2083

Rôle :
    Charger la base d'encodages (encoded_faces.pkl) et comparer un nouvel
    encodage 128-D à cette base. Retourner le nom le plus probable + un score
    de confiance, ou "Unknown / Inconnu" si la distance dépasse TOLERANCE.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import face_recognition


DEFAULT_TOLERANCE = 0.50   # seuil par défaut (ajustable lors des tests J5)
UNKNOWN_LABEL = "Unknown / Inconnu"


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------
class FaceMatcher:
    """
    Compare un encodage facial à une base d'encodages connus.

    Attributs
    ---------
    tolerance : float
        Distance euclidienne maximale pour considérer un match valide.
        En dessous → reconnu. Au-dessus → inconnu.
    known_encodings : np.ndarray (N, 128)
        Matrice des encodages de la base.
    known_names : list[str]
        Noms associés (même ordre que known_encodings).
    """

    def __init__(self, pickle_path: str | Path,
                 tolerance: float = DEFAULT_TOLERANCE,
                 verbose: bool = True):
        self.pickle_path = Path(pickle_path)
        self.tolerance = float(tolerance)
        self.verbose = verbose

        self.known_encodings: np.ndarray | None = None
        self.known_names: list[str] = []

        self._load()

    # ------------------------------------------------------------------ #
    # Chargement / rechargement de la base
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Charge encoded_faces.pkl produit par face_encoder.py (Étudiant 2)."""
        if not self.pickle_path.exists():
            raise FileNotFoundError(
                f"Base d'encodages introuvable : {self.pickle_path}\n"
                f"→ Lancez d'abord face_encoder.py pour la générer."
            )

        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)

        # Tolérance sur le format du pickle (doit contenir 'encodings' et 'names')
        if isinstance(data, dict):
            encodings = data.get("encodings")
            names = data.get("names")
        else:
            # Ancien format : tuple (encodings, names)
            encodings, names = data

        if encodings is None or names is None:
            raise ValueError(
                "Format de pickle invalide : clés 'encodings' et 'names' attendues."
            )

        self.known_encodings = np.asarray(encodings, dtype=np.float64)
        self.known_names = list(names)

        if self.verbose:
            print(f"[FaceMatcher] {len(self.known_names)} encodages chargés "
                  f"({len(set(self.known_names))} personnes distinctes). "
                  f"TOLERANCE = {self.tolerance}")

    def reload(self) -> None:
        """Recharge la base (utile si l'Étudiant 2 régénère le pickle)."""
        self._load()

    # ------------------------------------------------------------------ #
    # Matching
    # ------------------------------------------------------------------ #
    def match(self, face_encoding: np.ndarray) -> tuple[str, float]:
        """
        Compare UN encodage à la base.

        Paramètres
        ----------
        face_encoding : np.ndarray de forme (128,)

        Retour
        ------
        (nom, confiance) où :
            - nom       : str, nom reconnu ou UNKNOWN_LABEL
            - confiance : float entre 0.0 et 100.0 (pourcentage)
        """
        if self.known_encodings is None or len(self.known_encodings) == 0:
            return UNKNOWN_LABEL, 0.0

        face_encoding = np.asarray(face_encoding, dtype=np.float64)
        if face_encoding.shape != (128,):
            raise ValueError(
                f"Encodage attendu de forme (128,), reçu {face_encoding.shape}"
            )

        # Distance euclidienne à chaque encodage connu
        distances = face_recognition.face_distance(
            self.known_encodings, face_encoding
        )

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        # Score de confiance : 100 % si distance = 0, 0 % si distance >= 1
        confidence = max(0.0, min(100.0, (1.0 - best_distance) * 100.0))

        if best_distance < self.tolerance:
            return self.known_names[best_idx], confidence
        return UNKNOWN_LABEL, confidence

    def match_many(self, face_encodings: Iterable[np.ndarray]
                   ) -> list[tuple[str, float]]:
        """Applique match() à une liste d'encodages (utile pour les images)."""
        return [self.match(enc) for enc in face_encodings]

    # ------------------------------------------------------------------ #
    # Utilitaires
    # ------------------------------------------------------------------ #
    def set_tolerance(self, value: float) -> None:
        """Permet d'ajuster le seuil à chaud (tests J5)."""
        self.tolerance = float(value)
        if self.verbose:
            print(f"[FaceMatcher] TOLERANCE = {self.tolerance}")

    def stats(self) -> dict:
        """Renvoie quelques stats sur la base chargée."""
        if self.known_encodings is None:
            return {"n_encodings": 0, "n_persons": 0, "tolerance": self.tolerance}
        return {
            "n_encodings": len(self.known_names),
            "n_persons": len(set(self.known_names)),
            "tolerance": self.tolerance,
        }


# ---------------------------------------------------------------------------
# Test rapide en ligne de commande
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    pickle_file = sys.argv[1] if len(sys.argv) > 1 else "encoded_faces.pkl"
    matcher = FaceMatcher(pickle_file, tolerance=0.50)
    print("Stats base :", matcher.stats())