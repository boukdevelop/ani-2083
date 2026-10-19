# 🏗️ Architecture Logicielle & Contrats d'Interface — AI Sketch Recognizer

> Ce document définit l'architecture logicielle modulaire du projet, l'arborescence des fichiers, les flux de données de bout en bout et les **contrats d'interface (signatures Python formelles)** entre chacun des 5 modules.
> Il garantit que chaque membre de l'équipe peut développer et tester son code de manière totalement isolée et asynchrone sans bloquer ses coéquipiers.

---

## 1. Vue d'Ensemble du Flux de Données

Le système fonctionne selon deux phases distinctes :

1. **Phase Hors-Ligne (Pipeline ML)** : Préparation du jeu de données $\rightarrow$ Entraînement du CNN $\rightarrow$ Évaluation & Exportation du modèle.
2. **Phase Temps Réel (Application Interactive)** : Tracé utilisateur sur Canvas $\rightarrow$ Capture de l'image $\rightarrow$ Pipeline de centrage & redimensionnement OpenCV $\rightarrow$ Inférence TensorFlow $\rightarrow$ Affichage du Top-3 des prédictions.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PHASE HORS-LIGNE                                       │
│                                                                                        │
│   Google Cloud Storage               prepare_data.py [M2]         train_model.py [M3]  │
│   ┌────────────────────┐            ┌────────────────────┐       ┌────────────────────┐│
│   │ 8x fichiers .npy   │ ─────────► │ - Filtrage (10k/cl)│ ────► │ - Modèle CNN       ││
│   │ (Quick, Draw!)     │            │ - Normalisation    │       │ - Entraînement     ││
│   └────────────────────┘            │ - Split Train/Test │       │ - Évaluation       ││
│                                     └─────────┬──────────┘       └─────────┬──────────┘│
│                                               │                            │           │
│                                               ▼                            ▼           │
│                                     data/processed/              models/sketch_cnn     │
│                                     dataset_sketch.npz           .keras + classes.json │
└───────────────────────────────────────────────┼────────────────────────────┼───────────┘
                                                │                            │
                                                ▼                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PHASE TEMPS RÉEL (APPLICATION)                          │
│                                                                                        │
│   ┌──────────────────────┐          ┌────────────────────┐       ┌────────────────────┐│
│   │    ui_app.py [M4]    │          │inference_engine [M1│       │ bonus_filters [M5] ││
│   │  Canvas PyQt6 / Web  │ ───────► │ - Centrage & Crop  │ ────► │ - Filtres Canny    ││
│   │  Tracé de l'utilisateur│ (Image)  │ - Resize 28x28     │       │ - Effet Néon       ││
│   │  Boutons & Affichage │          │ - Inférence CNN    │       │ - Export Galerie   ││
│   └──────────▲───────────┘          └─────────┬──────────┘       └────────────────────┘│
│              │                                │                                        │
│              └────────────────────────────────┘                                        │
│                        Résultats (Top-3 classes + scores)                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Arborescence du Projet

```text
projet2_AI_Sketch/
│
├── config.py                     # [M1/Tous] Constantes, chemins et hyperparamètres partagés
├── main.py                       # [M1] Point d'entrée principal de l'application
├── inference_engine.py           # [M1] Moteur de prédiction et pipeline de prétraitement
├── prepare_data.py               # [M2] Script ETL de téléchargement et préparation du dataset
├── dataset_loader.py             # [M2] Utilitaires de chargement et découpage des matrices
├── model_builder.py              # [M3] Architecture neuronale Keras (CNN)
├── train_model.py                # [M3] Script d'entraînement, callbacks et sauvegarde
├── evaluate.py                   # [M3] Métriques de performance, matrice de confusion
├── ui_app.py                     # [M4] Interface graphique (PyQt6 ou alternative Web Flask)
├── bonus_filters.py              # [M5] Effets visuels créatifs OpenCV et exports
│
├── data/
│   ├── raw/                      # Fichiers bruts Quick, Draw! (.npy téléchargés)
│   └── processed/                # Dataset pré-traité et assemblé (dataset_sketch.npz)
│
├── models/
│   ├── sketch_cnn_model.keras    # Modèle CNN entraîné sauvegardé
│   └── classes.json              # Dictionnaire des labels de classes
│
├── tests/                        # [M5] Tests unitaires et d'intégration automatisés
│   ├── test_data.py
│   ├── test_inference.py
│   └── test_model.py
│
├── exports/                      # Dossier de sauvegarde des croquis créés par l'utilisateur
├── requirements.txt              # Dépendances pip du projet
├── roadmap.md                    # Planning et organisation de la semaine
├── dependance.md                 # Documentation d'installation et analyse technique
├── architecture.md               # Architecture logicielle et contrats d'API (ce fichier)
└── pre_requis.md                 # Fondations théoriques du projet
```

---

## 3. Contrats d'Interface Formels (API Python entre Modules)

Pour que chaque membre travaille sans friction, les signatures des fonctions et classes ci-dessous sont **figées et normées**.

---

### 🔹 Module 1 : `config.py` — Configuration Globale (Membre 1)

Ce module centralise tous les paramètres clés du projet afin d'éviter les valeurs codées en dur ("magic numbers").

```python
"""
config.py — Configuration centrale du projet AI Sketch Recognizer
Responsable : Membre 1 (Chef de Projet)
"""
from pathlib import Path
from typing import List

# Répertoires racines
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
EXPORTS_DIR: Path = BASE_DIR / "exports"

# Paramètres de l'image Quick, Draw!
IMAGE_SIZE: int = 28
IMAGE_CHANNELS: int = 1
INPUT_SHAPE: tuple = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

# Catégories sélectionnées par défaut (8 classes)
CATEGORIES: List[str] = [
    "cat",
    "house",
    "car",
    "tree",
    "sun",
    "bicycle",
    "apple",
    "star",
]

# Hyperparamètres d'entraînement
SAMPLES_PER_CLASS: int = 10000   # Nombre d'exemples par classe (équilibrage)
BATCH_SIZE: int = 128
EPOCHS: int = 20
LEARNING_RATE: float = 0.001
TEST_SPLIT: float = 0.15
VAL_SPLIT: float = 0.15

# Chemins des artéfacts
MODEL_FILE_PATH: Path = MODELS_DIR / "sketch_cnn_model.keras"
CLASSES_FILE_PATH: Path = MODELS_DIR / "classes.json"
DATASET_FILE_PATH: Path = PROCESSED_DATA_DIR / "dataset_sketch.npz"
```

---

### 🔹 Module 2 : `prepare_data.py` & `dataset_loader.py` — Ingestion & Traitement des Données (Membre 2)

#### Contrat de `dataset_loader.py` :

```python
"""
dataset_loader.py — Fonctions de chargement et découpage des données
Responsable : Membre 2 (Data Engineer)
"""
from typing import Tuple, List
import numpy as np

def download_quickdraw_category(category_name: str, dest_dir: str) -> str:
    """
    Télécharge le fichier bitmap .npy officiel depuis Google Cloud Storage.
    URL Source : https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category_name}.npy

    Args:
        category_name: Nom de la catégorie (ex: 'cat')
        dest_dir: Répertoire de destination

    Returns:
        Chemin absolu du fichier .npy téléchargé
    """
    pass


def prepare_and_split_dataset(
    categories: List[str],
    samples_per_category: int = 10000,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge les fichiers npy, filtre, normalise dans [0.0, 1.0], applique un encodage One-Hot
    et sépare en jeux d'entraînement, validation et test.

    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test)
        - X_train: forme (N_train, 28, 28, 1), type float32, valeurs dans [0.0, 1.0]
        - y_train: forme (N_train, num_classes), type float32 (One-Hot)
    """
    pass


def save_processed_dataset(output_path: str, **arrays) -> None:
    """
    Sauvegarde les tenseurs préparés sous forme de fichier compressé npz.
    """
    pass
```

---

### 🔹 Module 3 : `model_builder.py` & `train_model.py` — Intelligence Artificielle CNN (Membre 3)

#### Contrat de `model_builder.py` :

```python
"""
model_builder.py — Définition de l'architecture du réseau de neurones CNN
Responsable : Membre 3 (AI Engineer)
"""
import tensorflow as tf

def build_sketch_cnn(
    input_shape: tuple = (28, 28, 1),
    num_classes: int = 8,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Construit et compile le modèle CNN pour la reconnaissance de croquis.

    Architecture recommandée :
    1. Conv2D(32, (3, 3), activation='relu', padding='same') -> BatchNormalization()
    2. Conv2D(32, (3, 3), activation='relu') -> MaxPooling2D((2, 2)) -> Dropout(0.25)
    3. Conv2D(64, (3, 3), activation='relu', padding='same') -> BatchNormalization()
    4. Conv2D(64, (3, 3), activation='relu') -> MaxPooling2D((2, 2)) -> Dropout(0.25)
    5. Flatten()
    6. Dense(256, activation='relu') -> BatchNormalization() -> Dropout(dropout_rate)
    7. Dense(num_classes, activation='softmax')

    Returns:
        tf.keras.Model compilé avec l'optimiseur Adam et la loss categorical_crossentropy.
    """
    pass
```

#### Contrat de `train_model.py` :

```python
"""
train_model.py — Entraînement et sérialisation du modèle
Responsable : Membre 3 (AI Engineer)
"""
from typing import Dict, Any

def train_and_export_model(
    dataset_path: str,
    model_export_path: str,
    classes_export_path: str,
    epochs: int = 20,
    batch_size: int = 128
) -> Dict[str, Any]:
    """
    Exécute le cycle complet d'entraînement avec callbacks (EarlyStopping, ReduceLROnPlateau),
    sauvegarde le fichier .keras et exporte le dictionnaire de classes au format JSON.

    Returns:
        Dictionnaire contenant l'historique d'entraînement et les scores sur le jeu de test.
    """
    pass
```

---

### 🔹 Module 4 : `inference_engine.py` — Pipeline d'Inférence & Prétraitement (Membre 1)

Ce module est le cœur d'assemblage : il reçoit une image provenant de l'UI (quelle que soit sa résolution) et effectue la conversion conforme au format d'entrée du modèle.

```python
"""
inference_engine.py — Pipeline d'inférence en temps réel
Responsable : Membre 1 (Chef de Projet)
"""
from typing import List, Tuple, Union
import numpy as np
import tensorflow as tf
from PIL import Image

class SketchClassifier:
    """
    Moteur de classification de croquis en temps réel.
    """
    def __init__(self, model_path: str, classes_path: str):
        """
        Charge le modèle TensorFlow et la table de correspondance des labels.
        """
        self.model = None
        self.classes: List[str] = []
        self.load(model_path, classes_path)

    def load(self, model_path: str, classes_path: str) -> None:
        """Charge les fichiers modèles et classes."""
        pass

    def preprocess_canvas_image(self, image_input: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Prétraite l'image brute provenant du canvas :
        1. Conversion en niveaux de gris (grayscale).
        2. Inversion si nécessaire (le dataset Quick, Draw! a des traits blancs [255] sur fond noir [0]).
        3. Extraction de la boîte englobante (Bounding Box) pour rogner les marges vides.
        4. Ajout d'une marge de padding proportionnelle pour préserver le ratio d'aspect.
        5. Redimensionnement en 28x28 pixels avec cv2.INTER_AREA.
        6. Normalisation [0.0, 1.0] et reshape en (1, 28, 28, 1).

        Returns:
            np.ndarray de dimension (1, 28, 28, 1) prêt pour model.predict()
        """
        pass

    def predict(self, image_input: Union[np.ndarray, Image.Image], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Réalise la prédiction et retourne le classement des catégories les plus probables.

        Args:
            image_input: Image PIL ou matrice NumPy issue du Canvas UI
            top_k: Nombre de résultats à retourner

        Returns:
            Liste de tuples [(classe_1, probabilite_1), (classe_2, probabilite_2), ...]
            Exemple : [('cat', 0.942), ('apple', 0.035), ('car', 0.011)]
        """
        pass
```

---

### 🔹 Module 5 : `ui_app.py` — Interface Utilisateur Canvas (Membre 4)

L'interface graphique implémente le composant visuel de dessin et consomme directement le `SketchClassifier`.

```python
"""
ui_app.py — Interface Graphique Interactive (PyQt6)
Responsable : Membre 4 (Frontend / UI Developer)
"""
from PyQt6.QtWidgets import QMainWindow, QWidget
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QImage

class DrawingCanvas(QWidget):
    """
    Zone interactive de dessin vectoriel à la souris.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pen_width = 8
        self.pen_color = Qt.GlobalColor.black
        self.image = QImage(400, 400, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.drawing = False
        self.last_point = QPoint()

    def clear(self) -> None:
        """Réinitialise le canvas à blanc."""
        pass

    def get_image_as_numpy(self) -> np.ndarray:
        """Exporte le dessin sous forme de matrice NumPy RGB pour le moteur d'inférence."""
        pass


class MainWindow(QMainWindow):
    """
    Fenêtre principale regroupant le Canvas, les boutons d'action et le panneau de scores.
    """
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        # Initialisation de l'agencement UI
        # Connexion des signaux :
        # - Bouton 'Effacer' -> canvas.clear()
        # - Bouton 'Reconnaître' -> self.on_predict_clicked()
        # - Événement fin de trait souris -> self.on_auto_predict()

    def update_results_display(self, predictions: List[Tuple[str, float]]) -> None:
        """Met à jour l'affichage dynamique des barres de confiance."""
        pass
```

---

### 🔹 Module 6 : `bonus_filters.py` — Filtres Artistiques OpenCV & Galerie (Membre 5)

```python
"""
bonus_filters.py — Effets visuels créatifs et fonctions bonus
Responsable : Membre 5 (QA & Bonus Engineer)
"""
import numpy as np

def apply_canny_edge_filter(image: np.ndarray) -> np.ndarray:
    """Applique un filtre de détection de contours Canny pour un effet croquis technique."""
    pass

def apply_neon_glow_filter(image: np.ndarray, color_bgr: tuple = (255, 0, 128)) -> np.ndarray:
    """Transforme le croquis en dessin néon lumineux sur fond noir stylisé."""
    pass

def apply_watercolor_effect(image: np.ndarray) -> np.ndarray:
    """Applique un lissage bilatéral simulant une aquarelle artistique."""
    pass

def save_drawing_to_gallery(image: np.ndarray, predicted_label: str, output_folder: str) -> str:
    """
    Sauvegarde le croquis dans le dossier exports/ avec horodatage et label prédit.
    Returns:
        Chemin du fichier PNG généré
    """
    pass
```

---

## 4. Stratégie de Résolution des Défis Techniques Majeurs

| Défi Technique           | Problème Potentiel                                                                                                                                               | Solution Architecturale Retenue                                                                                                                             |
| :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Inversion Blanc/Noir** | L'utilisateur dessine en noir sur blanc, alors que Quick, Draw! stocke des pixels blancs ($255$) sur fond noir ($0$).                                            | Le moteur d'inférence vérifie la couleur moyenne du fond et inverse automatiquement les intensités : `255 - image`.                                         |
| **Centrage du Croquis**  | Si l'utilisateur dessine un petit chat tout en bas à gauche du canvas, un simple resize $28 \times 28$ déforme le dessin et produit une mauvaise classification. | Extraction des contours avec OpenCV (`cv2.boundingRect`), découpage strict de la zone utile, ajout d'un padding proportionnel carré puis redimensionnement. |
| **Épaisseur du Trait**   | Un trait trop fin (1 px sur canvas 400x400) devient presque invisible une fois réduit à 28x28.                                                                   | Le pinceau UI est calibré par défaut à $\ge 8$ px avec morphologie mathématique (dilatation `cv2.dilate`) si nécessaire.                                    |
| **Canvas Vide**          | Si l'utilisateur clique sur prédire sans avoir rien dessiné, l'IA produit une prédiction aléatoire.                                                              | Détection de variance minimale de pixels : si moins de 10 pixels actifs sont détectés, renvoi immédiat de l'état `"Aucun dessin détecté"`.                  |
