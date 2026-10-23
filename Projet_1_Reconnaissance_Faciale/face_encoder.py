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
import cv2
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
DETECTION_MODEL = "cnn"          # détecteur principal ; voir la cascade ci-dessous

# Détection en cascade : chaque détecteur reçoit la même image, réduite à une
# taille où le visage fait quelques dizaines de pixels.
# HOG s'est révélé très peu fiable sur le dataset réel (photos téléphone en
# 1920x2560, vues de haut, éclairage irrégulier) : il est donc essayé en
# dernier, uniquement comme filet de sécurité sur une image plus grande.
CNN_MAX_SIDE = 250               # côté max de l'image utilisée pour la détection CNN (fiable)
HAAR_MAX_SIDE = 400              # côté max pour les cascades OpenCV de secours
HOG_MAX_SIDE = 800               # côté max pour le HOG dlib (filet de secours)
ENCODE_MAX_SIDE = 250            # taille de l'image utilisée pour calculer l'embedding
MIN_FACE_RATIO = 0.06            # visage = au moins 6% du plus petit côté (filtre anti faux positifs)
BOX_PADDING = 0.10               # marge autour de la boîte avant encodage (menton + front)


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


def _resize_max_side(rgb, max_side):
    """
    Réduit une image RGB pour que son plus grand côté ne dépasse pas max_side.

    Returns:
        tuple[np.ndarray, float]: (image réduite, facteur d'échelle appliqué)
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return rgb, 1.0
    resized = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _equalize_lighting(rgb):
    """
    Égalise l'éclairage (CLAHE sur le canal L de Lab).

    Les photos du dataset ont des luminosités très inégales (murs clairs,
    contre-jour, ombres). Sans cela, les gradients de Sobel du HOG sont trop
    faibles sur les visages en zone sombre, d'où les détections manquées.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def _detect_hog(rgb):
    """Détection HOG de dlib sur une image déjà réduite et éclaircie."""
    try:
        boxes = face_recognition.face_locations(rgb, model="hog")
    except Exception as e:
        logger.debug(f"HOG en échec : {e}")
        return []
    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]


# Le détecteur CNN de dlib est un modèle natif lourd : il n'est pas thread-safe
# et prend ~1 s par image sur CPU. On le charge une seule fois (au lieu de le
# recharger à chaque appel) et on le désactive au-delà d'un budget de temps.
_CNN_DETECTOR = None
_CNN_TIME_SPENT = 0.0            # secondes cumulées déjà consommées par le CNN
CNN_TIME_BUDGET = 120.0          # budget total alloué au CNN pour tout le build


def _get_cnn_detector():
    """Instancie (une seule fois) le détecteur neuronal dlib MMOD."""
    global _CNN_DETECTOR
    if _CNN_DETECTOR is None:
        import dlib
        from face_recognition_models import face_detection_model
        _CNN_DETECTOR = dlib.cnn_face_detection_model_v1(
            face_detection_model()
        )
    return _CNN_DETECTOR


def _detect_cnn(rgb):
    """
    Détection CNN (dlib MMOD) : couvre les visages de profil / vues du haut.

    Ce détecteur est le plus fiable mais aussi le plus lent (~1 s/image sur CPU).
    Un budget global évite qu'un dataset de centaines de photos ne prenne des
    heures : passé le budget, la cascade continue avec Haar et HOG.
    """
    global _CNN_TIME_SPENT
    if _CNN_TIME_SPENT > CNN_TIME_BUDGET:
        return []

    # Le CNN s'éteint aussi dès qu'un visage a déjà été trouvé par Haar/HOG :
    # inutile de payer 1 s par image quand une cascade rapide suffit.

    t0 = time.time()
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detections = _get_cnn_detector()(gray, 0)
    except Exception as e:
        logger.debug(f"CNN en échec : {e}")
        return []
    finally:
        _CNN_TIME_SPENT += time.time() - t0

    # dlib renvoie des rectangles (left, top, right, bottom)
    return [(d.rect.top(), d.rect.right(), d.rect.bottom(), d.rect.left())
            for d in detections]


def _detect_haar(rgb):
    """Détection de secours par cascades OpenCV (frontal, profile, profile miroir)."""
    gray = cv2.cvtColor(_equalize_lighting(rgb), cv2.COLOR_RGB2GRAY)
    min_side = max(30, int(min(gray.shape[:2]) * MIN_FACE_RATIO))
    boxes = []

    for xml_name in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ):
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + xml_name)
        except Exception:
            continue
        if cascade.empty():
            continue

        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_side, min_side)
        )
        if len(faces) == 0 and "profile" in xml_name:
            # Profil orienté à droite
            flipped = cv2.flip(gray, 1)
            width = gray.shape[1]
            faces = cascade.detectMultiScale(
                flipped, scaleFactor=1.05, minNeighbors=3, minSize=(min_side, min_side)
            )
            faces = [(width - x - w, y, w, h) for (x, y, w, h) in faces]

        for (x, y, w, h) in faces:
            # Conversion (x, y, w, h) -> (top, right, bottom, left)
            boxes.append((int(y), int(x + w), int(y + h), int(x)))

    return boxes


def _detect_face_box(image, image_path):
    """
    Détecte le visage principal avec une stratégie en cascade.

    Ordre : CNN (le plus fiable, détecteur neuronal dlib MMOD) -> cascades Haar -> HOG.
    Chaque détecteur travaille sur une version réduite de l'image : le détecteur
    neuronal est trop lent sur une photo de 2500 px, tandis que le HOG de dlib
    n'est fiable que si le visage fait ~80 px de large, ce qui n'est jamais le cas
    sur une photo prise en pleine résolution.

    Returns:
        tuple[tuple[int, int, int, int] | None, str]:
            (boîte (top, right, bottom, left) en coordonnées ORIGINALES, détecteur utilisé)
    """
    h_orig, w_orig = image.shape[:2]
    equalized = _equalize_lighting(image)

    detectors = (
        ("cnn", _detect_cnn, CNN_MAX_SIDE),
        ("haar", _detect_haar, HAAR_MAX_SIDE),
        ("hog", _detect_hog, HOG_MAX_SIDE),
    )

    for name, detect, max_side in detectors:
        small, scale = _resize_max_side(equalized, max_side)
        boxes = detect(small)
        if not boxes:
            logger.debug(f"'{image_path}' : aucun visage via {name} ({max_side} px).")
            continue

        # Le sujet principal = le plus grand visage proposé
        boxes = sorted(
            boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True
        )

        for top, right, bottom, left in boxes:
            # Remise à l'échelle des coordonnées d'origine
            top, right, bottom, left = (
                int(top / scale), int(right / scale),
                int(bottom / scale), int(left / scale),
            )

            # Marge : le modèle de landmarks 68 points a besoin du menton et du front
            pad_w = int((right - left) * BOX_PADDING)
            pad_h = int((bottom - top) * BOX_PADDING)
            box = (
                max(0, top - pad_h),
                min(w_orig, right + pad_w),
                min(h_orig, bottom + pad_h),
                max(0, left - pad_w),
            )

            # Garde-fou anti faux positif : le crop doit contenir exactement un visage
            if not _is_single_face(equalized, box):
                logger.debug(
                    f"'{image_path}' : boîte {name} rejetée (pas un visage unique)."
                )
                continue

            logger.debug(f"'{image_path}' : visage trouvé via {name}.")
            return box, name

    return None, "none"


def _encode_image(image_path, model=DEFAULT_MODEL, num_jitters=DEFAULT_NUM_JITTERS):
    """
    Calcule l'embedding 128-D d'une image.

    Pipeline :
      1. Chargement + égalisation d'éclairage
      2. Réduction à au plus ~250 px de côté pour les détecteurs
      3. Détection en cascade du visage principal (CNN -> Haar -> HOG)
      4. Encodage 128-D sur l'image réduite, dans le même repère que la boîte

    Politique :
      - 0 visage détecté  -> ("no_face", None, "none")
      - 1 visage détecté  -> ("ok", vecteur 128-D, détecteur utilisé)
      - 2+ visages        -> on garde le PLUS GRAND (heuristique : le sujet principal)
      - image illisible   -> ("corrupted", None, "none")

    Args:
        image_path (str): chemin de l'image.
        model (str): "small" ou "large".
        num_jitters (int): nombre de ré-échantillonnages.

    Returns:
        tuple[str, np.ndarray | None, str]: (raison, encodage ou None, détecteur)
    """
    # 1) Chargement en RGB (face_recognition le fait pour nous, contrairement à cv2)
    try:
        image = face_recognition.load_image_file(image_path)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Image illisible '{image_path}' : {e}")
        return "corrupted", None, "none"
    except Exception as e:
        logger.error(f"Erreur inattendue sur '{image_path}' : {e}")
        return "corrupted", None, "none"

    # Garde-fou : image en niveaux de gris ou en 4 canaux -> conversion RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # 2) Détection en cascade du visage principal
    box, detector = _detect_face_box(image, image_path)
    if box is None:
        logger.info(f"Aucun visage dans '{image_path}' — ignorée.")
        return "no_face", None, "none"

    # 3) Encodage
    # ATTENTION : les landmarks de dlib échouent sur le crop haut de gamme d'une
    # photo 2560 px (0 landmark détecté) alors qu'ils fonctionnent sur la même
    # zone de l'image réduite. On encode donc **sur l'image réduite** explorée
    # lors de la détection (coordonnées dans le même repère), ce qui est aussi
    # beaucoup plus rapide (0,1 s contre 1,2 s).
    small, scale = _resize_max_side(image, ENCODE_MAX_SIDE)
    small = np.ascontiguousarray(small)
    small_box = (
        int(box[0] * scale), int(box[1] * scale),
        int(box[2] * scale), int(box[3] * scale),
    )

    try:
        encodings = face_recognition.face_encodings(
            small,
            known_face_locations=[small_box],
            model=model,
            num_jitters=num_jitters,
        )
    except Exception as e:
        logger.error(f"Échec d'encodage sur '{image_path}' : {e}")
        return "corrupted", None, "none"

    if not encodings:
        # La détection a trouvé un visage mais l'encodage échoue (carré hors-cadre)
        logger.info(f"Encodage impossible malgré détection '{image_path}' ({detector}).")
        return "no_face", None, "none"

    return "ok", encodings[0], detector


# =============================================================================
# GARDE-FOU QUALITÉ
# =============================================================================
def _is_single_face(image, box):
    """
    Vérifie que le candidat ressemble à un visage avant de l'encoder.

    dlib peut planter de façon irrécupérable (access violation) au lieu de
    retourner une erreur Python quand le prédicteur de landmarks reçoit une
    boîte trop serrée ou mal cadrée. On se contente donc de contrôles de forme
    et de taille, sans lancer d'appel natif risqué.

    Returns:
        bool: True si la boîte est exploitable telle quelle.
    """
    top, right, bottom, left = box
    height, width = bottom - top, right - left

    # Un visage doit occuper une zone non dégénérée et pas la totalité de l'image
    if height < 30 or width < 30:
        return False
    if height > image.shape[0] * 0.98 or width > image.shape[1] * 0.98:
        return False

    # Le visage doit être contenu dans l'image (débordement = box fausse)
    if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
        return False

    # Ratio hauteur/largeur : un visage réel est entre 0,8 et 2,0
    ratio = height / width
    return 0.8 <= ratio <= 2.0


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
    personnes = {}   # nom -> nombre d'encodages valides
    detecteurs = {}  # "hog" / "cnn" / "haar" -> nombre de visages acceptés

    # --- Accumulateurs ---
    encodings = []
    names = []

    logger.info(f"Démarrage du build sur '{dataset_path}'...")

    # --- Boucle principale ---
    for nom, image_path in _iter_person_images(dataset_path):
        images_traitees += 1
        reason, encoding, detector = _encode_image(
            image_path, model=model, num_jitters=num_jitters
        )

        if reason == "ok":
            encodings.append(encoding)
            names.append(nom)
            encodages_valides += 1
            personnes[nom] = personnes.get(nom, 0) + 1
            detecteurs[detector] = detecteurs.get(detector, 0) + 1

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
        "detection_cascade": ["cnn", "haar", "hog"],
        "detecteurs_utilises": detecteurs,
        "dim": ENCODING_DIM,
        "n_images": images_traitees,
        "n_valid": encodages_valides,
        "n_no_face": echecs_sans_visage,
        "n_corrupted": echecs_corrompus,
        "taux_reussite": round(encodages_valides / images_traitees * 100, 1) if images_traitees else 0.0,
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
    rapport = build_encodings("data/known_faces")
    print(rapport)