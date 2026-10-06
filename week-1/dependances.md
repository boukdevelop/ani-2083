# 📦 Dépendances & Bibliothèques — Projet Reconnaissance Faciale

> Liste complète des dépendances du projet, leur rôle, la procédure d'installation et le dépannage.
> Aligné sur `week-1/explainHIm.md`, `week-1/pré-requis.md`, `roadmap.md` et `architecture.md`.

---

## 1. Prérequis système

| Prérequis                     | Version recommandée | Rôle                                                                           |
| :---------------------------- | :------------------ | :----------------------------------------------------------------------------- |
| **Python**                    | 3.10 – 3.12         | Langage du projet (éviter 3.13, compatibilité dlib incertaine)                 |
| **pip**                       | à jour              | Gestionnaire de paquets                                                        |
| **Visual Studio Build Tools** | 2019 / 2022         | Compilation de **dlib** sous Windows (option « Développement Desktop en C++ ») |
| **Webcam**                    | —                   | Capture vidéo temps réel                                                       |
| **Git**                       | récent              | Gestion de version du code                                                     |

> ⚠️ **Point critique Windows** : `face_recognition` dépend de **dlib**, qui doit souvent être
> **compilé**. Sans les outils C++ de Visual Studio, l'installation échoue. **À vérifier le Jour 1
> par toute l'équipe** pour éviter un blocage.

---

## 2. Installation pas à pas

### Étape 1 — Créer et activer un environnement virtuel

```powershell
# Depuis la racine du projet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Sous **CMD** : `.venv\Scripts\activate.bat`
> Sous **Git Bash** : `source .venv/Scripts/activate`

### Étape 2 — Mettre à jour pip et les outils de build

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Étape 3 — Installer CMake (requis pour compiler dlib si aucun wheel binaire n'est dispo)

```powershell
pip install cmake
```

### Étape 4 — Installer les dépendances principales

```powershell
pip install face_recognition
pip install customtkinter
pip install opencv-python
pip install Pillow
pip install numpy
```

### Étape 5 — Tout installer d'un coup (recommandé)

```powershell
pip install -r requirements.txt
```

### Étape 6 — Vérifier l'installation

```powershell
python -c "import face_recognition, cv2, customtkinter, PIL, numpy; print('OK')"
```

Si `face_recognition` s'importe sans erreur, **dlib est correctement installé**.

---

## 3. Liste détaillée des dépendances

| Bibliothèque            | Version conseillée | Rôle dans le projet                                                                                                                       | Module(s) concerné(s)                              |
| :---------------------- | :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------- |
| **face_recognition**    | `>=1.3.0`          | Détection (`face_locations`), encodage 128-D (`face_encodings`), distance (`face_distance`), comparaison (`compare_faces`). Cœur de l'IA. | `face_encoder.py`, `face_matcher.py`, `main.py`    |
| **dlib**                | `>=19.24`          | Moteur sous-jacent de `face_recognition` (HOG + SVM + ResNet). Installé automatiquement comme dépendance.                                 | (transitive)                                       |
| **customtkinter**       | `>=5.2.0`          | Interface graphique moderne (thème sombre), boutons, labels image, panneaux.                                                              | `gui.py`, `main.py`                                |
| **opencv-python**       | `>=4.8.0`          | Capture webcam (`VideoCapture`), conversion BGR↔RGB, redimensionnement, dessin des bounding boxes.                                        | `camera_stream.py`, `capture_dataset.py`, `gui.py` |
| **Pillow**              | `>=10.0.0`         | Conversion des frames numpy → images affichables (`CTkImage`/`ImageTk`), chargement d'images statiques.                                   | `gui.py`                                           |
| **numpy**               | `>=1.24.0`         | Manipulation des tableaux de pixels et des vecteurs d'encodage (128-D).                                                                   | Tous les modules IA                                |
| **cmake**               | `>=3.27`           | Outil de build requis pour compiler dlib si aucun _wheel_ binaire n'est disponible.                                                       | (installation)                                     |
| **setuptools / wheel**  | à jour             | Outils de build Python pour l'installation des paquets natifs.                                                                            | (installation)                                     |
| **tkinter**             | inclus avec Python | Backend de Tkinter utilisé par CustomTkinter. Généralement fourni avec Python.                                                            | `gui.py`                                           |
| **pickle**              | stdlib             | Sérialisation des encodages dans `encoded_faces.pkl`.                                                                                     | `face_encoder.py`, `face_matcher.py`               |
| **os / pathlib / json** | stdlib             | Parcours des dossiers, gestion des chemins, métadonnées.                                                                                  | `dataset_manager.py`                               |

---

## 4. Fichier `requirements.txt`

```txt
# --- Cœur IA / reconnaissance faciale ---
face_recognition>=1.3.0
numpy>=1.24.0

# --- Traitement d'image / vidéo ---
opencv-python>=4.8.0
Pillow>=10.0.0

# --- Interface graphique ---
customtkinter>=5.2.0

# --- Outils de build (compilation dlib) ---
cmake>=3.27
```

> **Note** : `dlib` n'est pas listé explicitement car il est installé automatiquement comme
> dépendance de `face_recognition`. On peut toutefois le figer si nécessaire :
> `dlib>=19.24`.

---

## 5. Rôle détaillé des bibliothèques clés

### 🧠 `face_recognition`

Bibliothèque de haut niveau basée sur **dlib**. Fournit une API simple et adaptée à un petit
dataset pédagogique (5 personnes, 100-200 images). Fonctionne sur **CPU**.

Fonctions utilisées dans le projet :

- `face_locations(image)` → liste de bounding boxes `(top, right, bottom, left)`.
- `face_encodings(image, locations)` → liste de vecteurs **128 dimensions**.
- `face_distance(known_encodings, face_encoding)` → distances euclidiennes.
- `compare_faces(known_encodings, face_encoding, tolerance)` → booléens.

### 🎥 `opencv-python` (cv2)

Gère le matériel et le traitement d'image :

- `cv2.VideoCapture(0)` → ouverture de la webcam.
- `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` → conversion **BGR (OpenCV) → RGB (Pillow/Tk)**.
- `cv2.resize(...)` → réduction à **25 %** pour accélérer l'analyse.
- `cv2.rectangle` / `cv2.putText` → tracé des bounding boxes et du texte (nom + score).

### 🖼️ `customtkinter`

Surcouche moderne de Tkinter :

- Fenêtre principale, **thème sombre** (`set_appearance_mode("dark")`).
- Boutons « Démarrer Webcam », « Charger une Image », « Arrêter ».
- `CTkLabel` pour la zone vidéo et `CTkImage` pour l'affichage des frames.
- Boucle d'actualisation non bloquante via `after(15, callback)`.

### 🎨 `Pillow` (PIL)

- Convertit les tableaux numpy (frames RGB) en `ImageTk.PhotoImage` / `CTkImage`.
- Charge les images statiques pour la fonction « Charger une Image ».
- Redimensionne proprement les images pour l'affichage dans la GUI.

### 🔢 `numpy`

- Stocke les frames sous forme de tableaux `ndarray`.
- Représente les **vecteurs d'encodage 128-D**.
- Calcule les opérations matricielles de distance.

---

## 6. Dépannage (Troubleshooting)

### ❌ `ERROR: Could not build wheels for dlib`

**Cause** : CMake ou les outils de compilation C++ sont manquants.

**Solution** :

1. Installer **Visual Studio Build Tools** avec l'option « **Développement Desktop en C++** »
   (composant « MSVC v143 - VS 2022 C++ x64/x86 build tools » + « Windows 10/11 SDK »).
2. Installer CMake : `pip install cmake`.
3. Réinstaller : `pip install dlib` puis `pip install face_recognition`.

**Alternative rapide** : utiliser un **wheel précompilé** de dlib pour votre version de Python
(ex. via le dépôt `sachinprasadhs/dlib-windows` ou en cherchant un `.whl` correspondant à votre
version de Python et à votre architecture x64), puis :

```powershell
pip install dlib-19.xx.x-cpXX-win_amd64.whl
```

### ❌ `No module named 'tkinter'`

**Solution** : réinstaller Python en cochant « **tcl/tk and IDLE** » dans l'installeur Windows.

### ❌ La caméra ne s'ouvre pas (`VideoCapture` renvoie `False`)

**Solution** :

- Vérifier qu'aucune autre application n'utilise la webcam (Teams, Zoom…).
- Essayer un autre index : `cv2.VideoCapture(1)`.
- Vérifier les autorisations caméra dans Windows (Paramètres → Confidentialité → Caméra).

### ❌ L'interface est lente / saccadée

**Solution** :

- Vérifier que la détection s'effectue bien sur la frame réduite à **25 %**.
- Augmenter l'intervalle de la boucle `after()` (ex. `after(30, ...)`).
- Réduire le nombre d'encodages si le dataset est trop grand.

### ❌ `encoded_faces.pkl` introuvable

**Solution** : exécuter d'abord le script d'encodage :

```powershell
python face_encoder.py
```

Il génère `encoded_faces.pkl` à partir de `data/known_faces/`.

### ❌ Tous les visages sont « Unknown »

**Solution** :

- Vérifier que `encoded_faces.pkl` correspond bien aux personnes filmées.
- Ajuster le seuil `TOLERANCE` dans `face_matcher.py` (essayer `0.55`).
- Vérifier la qualité/luminosité de l'éclairage pendant la capture.

---

## 7. Configuration recommandée de l'environnement

| Élément                           | Valeur                                                                        |
| :-------------------------------- | :---------------------------------------------------------------------------- |
| Version Python                    | 3.10 – 3.12 (x64)                                                             |
| Environnement                     | virtuel (`.venv/`)                                                            |
| IDE conseillé                     | VS Code + extension Python, ou PyCharm                                        |
| Notebook                          | Jupyter (`pip install notebook`) pour le livrable `exploration.ipynb`         |
| Fichiers à ignorer (`.gitignore`) | `.venv/`, `__pycache__/`, `*.pkl`, `data/known_faces/`, `data/unknown_tests/` |

> ⚠️ **RGPD** : `encoded_faces.pkl` et les images de `data/` contiennent des **données
> biométriques**. Ne jamais les publier ni les versionner sur un dépôt public.

---

## 8. Installation du notebook (livrable)

Le projet exige un **notebook d'entraînement / expérimentation** :

```powershell
pip install notebook
jupyter notebook
```

Puis créer `notebooks/exploration.ipynb` pour explorer les encodages, les distances et le choix
du seuil `TOLERANCE`.
