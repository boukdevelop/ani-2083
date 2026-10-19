# 📦 Dépendances & Environnement Technique — AI Sketch Recognizer

> Ce document liste de manière exhaustive l'ensemble des bibliothèques nécessaires au fonctionnement du projet **AI Sketch Recognizer**, leurs versions recommandées, la méthode d'installation pas à pas dans un environnement virtuel et les solutions aux problèmes fréquents sous Windows.

---

## 1. Vue d'Ensemble de l'Environnement

Le projet repose sur l'écosystème Python orienté **Deep Learning (Vision par Ordinateur)** et interface utilisateur moderne.

| Composant                | Version Préconisée           | Rôle dans l'Architecture                                                                        |
| :----------------------- | :--------------------------- | :---------------------------------------------------------------------------------------------- |
| **Python**               | **3.10 à 3.12** (x64)        | Interpréteur Python stable compatible TensorFlow 2.15/2.16                                      |
| **TensorFlow**           | `2.15.0` (ou $\ge 2.15.0$)   | Framework de Deep Learning (construction, entraînement et inférence du CNN)                     |
| **NumPy**                | `1.26.4`                     | Manipulation matricielle des tenseurs de pixels $(28 \times 28)$                                |
| **OpenCV**               | `4.9.0.80` (`opencv-python`) | Prétraitement d'image (redimensionnement, binarisation, seuillage d'Otsu) & filtres artistiques |
| **Pillow (PIL)**         | `10.2.0`                     | Conversion entre formats graphiques UI et tableaux matriciels                                   |
| **PyQt6** _(ou Flask)_   | `6.6.1` _(ou 3.0.2)_         | Interface graphique interactive (Canvas de dessin à la souris)                                  |
| **Scikit-Learn**         | `1.4.1`                      | Split stratifié train/test, matrice de confusion et métriques d'évaluation                      |
| **Matplotlib / Seaborn** | `3.8.3`                      | Visualisation des courbes de perte/précision et distribution des données                        |
| **Pytest**               | `8.0.2`                      | Cadre de tests unitaires et d'intégration                                                       |

---

## 2. Contenu du Fichier `requirements.txt`

Le fichier `requirements.txt` prêt à l'emploi pour le projet comprend deux options d'interface (Desktop avec PyQt6 par défaut, ou Web avec Flask) :

```plaintext
# === Core Machine Learning & Calcul Numérique ===
tensorflow>=2.15.0,<2.17.0
numpy>=1.26.0,<2.0.0
scikit-learn>=1.4.0

# === Traitement d'Images & Vision par Ordinateur ===
opencv-python>=4.9.0.80
Pillow>=10.2.0

# === Interface Graphique (Choix par défaut : Desktop PyQt6) ===
PyQt6>=6.6.1

# === Alternative Web (Si option Web choisie) ===
Flask>=3.0.2
flask-cors>=4.0.0

# === Visualisation & Évaluation ===
matplotlib>=3.8.0
seaborn>=0.13.0

# === Qualité de Code, Tests & Utilitaires ===
pytest>=8.0.0
requests>=2.31.0
tqdm>=4.66.0
```

---

## 3. Procédure d'Installation Pas à Pas (Windows / PowerShell)

Suivez scrupuleusement les commandes ci-dessous pour initialiser un environnement propre et isolé :

### Étape 1 : Créer l'environnement virtuel dédié

Ouvrez un terminal PowerShell dans le répertoire racine du projet :

```powershell
# Création du venv avec Python 3.12 (ou 3.11)
python -m venv .venv
```

### Étape 2 : Activer l'environnement virtuel

- **Sous PowerShell** :

  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

  _(Si vous rencontrez une restriction d'exécution de script, lancez `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)_

- **Sous Command Prompt (CMD)** :

  ```cmd
  .venv\Scripts\activate.bat
  ```

- **Sous Linux / macOS / Git Bash** :
  ```bash
  source .venv/bin/activate
  ```

### Étape 3 : Mettre à jour les outils d'installation de base

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Étape 4 : Installer les dépendances du projet

```powershell
pip install -r requirements.txt
```

### Étape 5 : Valider l'intégrité de l'installation

Exécutez ce one-liner pour valider que tous les modules critiques s'importent correctement :

```powershell
python -c "import tensorflow as tf; import cv2; import numpy as np; import PIL; print('✅ Tous les packages critiques sont opérationnels ! TensorFlow version:', tf.__version__)"
```

---

## 4. Détail et Justification de Chaque Dépendance

### 🔹 TensorFlow (`tensorflow`)

- **Pourquoi ?** C'est le standard industriel pour l'apprentissage profond en vision par ordinateur. Il inclut l'API `tf.keras` de haut niveau qui permet de déclarer des réseaux de neurones convolutionnels de façon concise.
- **Rôle dans le projet** :
  - Définition du modèle `Sequential` ou fonctionnel (`Conv2D`, `MaxPooling2D`, `Dense`).
  - Entraînement optimisé avec suivi des époques.
  - Sauvegarde et chargement instantané du format standard `.keras`.

### 🔹 OpenCV (`opencv-python`)

- **Pourquoi ?** OpenCV (Open Source Computer Vision Library) est la référence absolue pour le traitement matriciel d'images à très haute vitesse en C++ avec bindings Python.
- **Rôle dans le projet** :
  - **Prétraitement d'inférence** : Extraction de la boîte englobante (bounding box) du dessin grâce à `cv2.findContours` pour centrer automatiquement le croquis.
  - **Redimensionnement intelligent** : `cv2.resize` avec interpolation d'aire (`cv2.INTER_AREA`) pour passer de la haute résolution du canvas vers la matrice $28 \times 28$.
  - **Filtres artistiques bonus** : Détecteur de bords de Canny (`cv2.Canny`), lissage bilatéral (`cv2.bilateralFilter`), inversions de palettes de couleurs.

### 🔹 NumPy (`numpy`)

- **Pourquoi ?** Quick, Draw! distribue ses fichiers au format binaire compressé `.npy`, qui stocke des tableaux de vecteurs 1D de 784 éléments ($28 \times 28$).
- **Rôle dans le projet** :
  - Chargement mémoire direct via `np.load()`.
  - Reshaping de matrices `(N, 784)` vers `(N, 28, 28, 1)`.
  - Normalisation vectorielle rapide : `X = X.astype('float32') / 255.0`.

### 🔹 PyQt6 (`PyQt6`)

- **Pourquoi ?** Interface native multiplateforme ultra-réactive. Contrairement aux solutions web qui peuvent subir des latences de transfert réseau ou de rendu, PyQt6 offre un composant `QWidget` avec redéfinition directe de `paintEvent` à 60 FPS sans lag lors du tracé souris.
- **Rôle dans le projet** :
  - Fournir la zone de dessin interactive (Canvas avec tracé fluide).
  - Gestion des boutons (Effacer, Reconnaître, Choisir couleur/taille).
  - Rendu des graphiques de confiance en direct.

### 🔹 Flask & Flask-CORS _(Option alternative Web)_

- **Pourquoi ?** Si l'équipe retient l'option Web, Flask permet de créer un micro-service REST très léger avec deux routes :
  - `GET /` : Sert la page HTML/Canvas.
  - `POST /predict` : Reçoit le croquis encodé en Base64, l'envoie au moteur d'inférence et renvoie le JSON de résultat.

### 🔹 Pillow (`Pillow`)

- **Pourquoi ?** Pont idéal entre les données graphiques issues de PyQt6/Canvas (flux d'octets PNG/BMP) et les tableaux bruts NumPy.
- **Rôle dans le projet** :
  - Conversion d'un tampon mémoire d'image (`BytesIO`) en tableau NumPy.
  - Sauvegarde facile des exports de dessins dans la galerie utilisateur.

### 🔹 Scikit-Learn (`scikit-learn`)

- **Pourquoi ?** Fournit des fonctions robustes et fiables pour l'ingénierie des données et la métrologie.
- **Rôle dans le projet** :
  - `train_test_split(..., stratify=y)` pour garantir des proportions de classes strictement identiques entre train, validation et test.
  - `classification_report` et `confusion_matrix` pour l'analyse des performances.

---

## 5. Dépannage & Bonnes Pratiques Windows

### ⚠️ Avertissement sur la version de Python

- **Conseil** : Privilégiez **Python 3.11 ou 3.12**.
- Si vous utilisez Python 3.13 (très récent), certains packages compilés en C (comme certaines versions de TensorFlow ou OpenCV) peuvent ne pas disposer de wheel officiel au moment de l'installation. Dans ce cas, lancez la création du venv avec un lanceur ciblé : `py -3.11 -m venv .venv` ou `py -3.12 -m venv .venv`.

### ⚠️ Erreur `Execution of scripts is disabled on this system` (PowerShell)

Lors de l'activation du venv sous Windows, PowerShell bloque parfois par défaut l'exécution des scripts non signés.
**Solution** :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### ⚠️ Erreur d'allocation mémoire avec TensorFlow

Sur des machines ne disposant pas de GPU dédié Nvidia avec CUDA configuré, TensorFlow affichera un message d'information indiquant qu'il bascule sur le CPU (`OneDNN custom operations are on`). C'est **tout à fait normal et attendu** : un CNN sur des images $28 \times 28$ s'entraîne en moins de 3 à 5 minutes sur un CPU moderne !
