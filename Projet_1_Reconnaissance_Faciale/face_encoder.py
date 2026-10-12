# =============================================================================
# face_encoder.py
# -----------------------------------------------------------------------------
# Module d'encodage facial (KDs Encoder).
#
# Rôle :
#   1. Parcourir le dataset `data/known_faces/<nom>/*.jpg,*png,*webp` et ignorer les fichiers invalides
#   2. Calculer un embedding 128-D pour chaque image contenant un visage
#   3. Sérialiser le tout (encodages + noms + métadonnées) dans `encoded_faces.pkl`
#   4. Exposer `load_encodings()` pour que E3 (face_matcher) consomme la base.
#
# Contrat public :
#   - build_encodings(dataset_path, output_path, model, num_jitters) -> dict (rapport)
#   - load_encodings(pkl_path) -> (encodings, names, metadata)
#   - ENCODING_DIM = 128
# =============================================================================

# --- Bibliothèque standard -----------------------------------------------------
import os
import pickle
import time
import logging
from datetime import datetime
from pathlib import Path

# --- Bibliothèques tierces ----------------------------------------------------
import face_recognition
import numpy as np

# --- Logger du module (ne PAS configurer basicConfig ici, c'est le rôle de main.py)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES
# =============================================================================
ENCODING_DIM = 128
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_MODEL = "small"          # "small" = 128-D rapide ; "large" = 128-D + précis
DEFAULT_NUM_JITTERS = 1          # 1 = rapide ; 10 = +précis mais 10x plus lent
DEFAULT_OUTPUT = "encoded_faces.pkl"
DETECTION_MODEL = "hog"          # "hog" = CPU rapide ; "cnn" = GPU + précis


# =============================================================================
# FONCTIONS PRIVÉES (préfixe "_")
# =============================================================================
def _iter_person_images(dataset_path):
    """
    Parcourt le dataset et yield (nom_personne, chemin_image) pour chaque image valide.

    Le tri est important : il garantit que deux builds successifs produisent
    exactement le même .pkl (reproductibilité).

    Args:
        dataset_path (str | Path): racine du dataset (ex. "data/known_faces").

    Yields:
        tuple[str, str]: (nom de la personne, chemin complet de l'image)

    Raises:
        FileNotFoundError: si dataset_path n'existe pas.
    """
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Le dataset '{dataset_path}' n'existe pas.")
    if not root.is_dir():
        raise NotADirectoryError(f"'{dataset_path}' n'est pas un dossier.")

    # sorted() => ordre stable des personnes
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue  # on ignore les fichiers à la racine
        if person_dir.name.startswith("."):
            continue  # on ignore les dossiers cachés (.DS_Store, etc.)

        # sorted() => ordre stable des images dans chaque personne
        for image_file in sorted(person_dir.iterdir()):
            if image_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield person_dir.name, str(image_file)


def _encode_image(image_path, model=DEFAULT_MODEL, num_jitters=DEFAULT_NUM_JITTERS):
    """
    Calcule l'embedding 128-D d'une image.

    Politique :
      - 0 visage détecté  -> ("no_face", None)
      - 1 visage détecté  -> ("ok", vecteur 128-D)
      - 2+ visages        -> on garde le PLUS GRAND (heuristique : le sujet principal)
      - image illisible   -> ("corrupted", None)

    Args:
        image_path (str): chemin de l'image.
        model (str): "small" ou "large".
        num_jitters (int): nombre de ré-échantillonnages.

    Returns:
        tuple[str, np.ndarray | None]: (raison, encodage ou None)
    """
    # 1) Chargement en RGB (face_recognition le fait pour nous, contrairement à cv2)
    try:
        image = face_recognition.load_image_file(image_path)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Image illisible '{image_path}' : {e}")
        return "corrupted", None
    except Exception as e:
        logger.error(f"Erreur inattendue sur '{image_path}' : {e}")
        return "corrupted", None

    # 2) Détection des visages — un seul appel, on réutilise le résultat
    try:
        boxes = face_recognition.face_locations(image, model=DETECTION_MODEL)
    except Exception as e:
        logger.error(f"Échec de détection sur '{image_path}' : {e}")
        return "corrupted", None

    # 3) Application de la politique
    if len(boxes) == 0:
        logger.info(f"Aucun visage dans '{image_path}' — ignorée.")
        return "no_face", None

    if len(boxes) > 1:
        # Heuristique : on prend le visage avec la plus grande aire (sujet principal)
        # box = (top, right, bottom, left)
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
        logger.debug(f"'{image_path}' : {len(boxes)} visages, on garde le plus grand.")

    chosen_box = boxes[0]

    # 4) Encodage — on passe chosen_box pour NE PAS re-détecter
    try:
        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=[chosen_box],
            model=model,
            num_jitters=num_jitters,
        )
    except Exception as e:
        logger.error(f"Échec d'encodage sur '{image_path}' : {e}")
        return "corrupted", None

    if not encodings:
        # Cas rare : la détection a trouvé un visage mais l'encodage échoue
        return "no_face", None

    return "ok", encodings[0]


# =============================================================================
# API PUBLIQUE
# =============================================================================
def build_encodings(
    dataset_path,
    output_path=DEFAULT_OUTPUT,
    model=DEFAULT_MODEL,
    num_jitters=DEFAULT_NUM_JITTERS,
):
    """
    Génère la base d'encodages à partir du dataset et la sauvegarde en .pkl.

    Args:
        dataset_path (str): racine du dataset.
        output_path (str): chemin du .pkl de sortie.
        model (str): "small" ou "large".
        num_jitters (int): nombre de ré-échantillonnages.

    Returns:
        dict: rapport de build (compteurs, durées, personnes).

    Raises:
        RuntimeError: si aucun encodage valide n'a été produit.
    """
    t_start = time.time()

    # --- Compteurs ---
    images_traitees = 0
    encodages_valides = 0
    echecs_sans_visage = 0
    echecs_corrompus = 0
    personnes = {}  # nom -> nombre d'encodages valides

    # --- Accumulateurs ---
    encodings = []
    names = []

    logger.info(f"Démarrage du build sur '{dataset_path}'...")

    # --- Boucle principale ---
    for nom, image_path in _iter_person_images(dataset_path):
        images_traitees += 1
        reason, encoding = _encode_image(image_path, model=model, num_jitters=num_jitters)

        if reason == "ok":
            encodings.append(encoding)
            names.append(nom)
            encodages_valides += 1
            personnes[nom] = personnes.get(nom, 0) + 1

        elif reason == "no_face":
            echecs_sans_visage += 1

        elif reason == "corrupted":
            echecs_corrompus += 1

    # --- Garde-fou : base vide = inutile ---
    if encodages_valides == 0:
        raise RuntimeError(
            "Aucun encodage valide produit. Vérifie le dataset et le format des images."
        )

    duree = time.time() - t_start

    # --- Métadonnées (contrat avec E3) ---
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "num_jitters": num_jitters,
        "detection_model": DETECTION_MODEL,
        "dim": ENCODING_DIM,
        "n_images": images_traitees,
        "n_valid": encodages_valides,
        "n_no_face": echecs_sans_visage,
        "n_corrupted": echecs_corrompus,
        "personnes": personnes,
        "duree_secondes": round(duree, 2),
    }

    # --- Sérialisation ATOMIQUE (temp -> rename) ---
    payload = {"encodings": encodings, "names": names, "metadata": metadata}
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, output_path)  # rename atomique

    logger.info(
        f"Build terminé : {encodages_valides}/{images_traitees} encodages valides "
        f"en {metadata['duree_secondes']}s -> '{output_path}'"
    )

    return metadata


def load_encodings(pkl_path=DEFAULT_OUTPUT):
    """
    Charge la base d'encodages sérialisée.

    Args:
        pkl_path (str): chemin du .pkl.

    Returns:
        tuple[list, list, dict]: (encodings, names, metadata)

    Raises:
        FileNotFoundError: si le .pkl n'existe pas.
        ValueError: si la structure est invalide.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Base d'encodages introuvable : '{pkl_path}'. "
            "Lance d'abord build_encodings()."
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # --- Validation du contrat ---
    for key in ("encodings", "names", "metadata"):
        if key not in data:
            raise ValueError(f"Structure invalide : clé '{key}' manquante dans {pkl_path}.")

    if len(data["encodings"]) != len(data["names"]):
        raise ValueError("Incohérence : len(encodings) != len(names).")

    if data["metadata"].get("dim") != ENCODING_DIM:
        raise ValueError(
            f"Dimension inattendue : {data['metadata'].get('dim')} != {ENCODING_DIM}."
        )

    return data["encodings"], data["names"], data["metadata"]


# =============================================================================
# TEST RAPIDE (exécution directe du module)
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"Dimension d'encodage : {ENCODING_DIM}")

    # Décommente quand ton dataset est prêt :
    # rapport = build_encodings("data/known_faces")
    # print(rapport)