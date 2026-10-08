# 📦 Dépendances & Bibliothèques — Projet Reconnaissance Faciale

> Liste complète des dépendances du projet, leur rôle, la procédure d'installation et le dépannage.
> Aligné sur `week-1/explainHIm.md`, `week-1/pré-requis.md`, `roadmap.md` et `architecture.md`.

---

## 1. Prérequis système

| Prérequis        | Version recommandée  | Rôle                                                          |
| :--------------- | :------------------- | :------------------------------------------------------------ |
| **Python**       | **3.12 exactement**  | Langage du projet (⚠️ voir avertissement ci-dessous)          |
| **pip**          | à jour               | Gestionnaire de paquets                                       |
| **Webcam**       | —                    | Capture vidéo temps réel                                      |
| **Git**          | récent               | Gestion de version du code                                    |
| **VS Build Tools** | *Optionnel*        | Uniquement si vous choisissez de compiler dlib (voir §6.1)    |

> ⚠️ **AVERTISSEMENT CRITIQUE — Python 3.12 uniquement**
>
> Le projet a été **testé et validé sur Python 3.12.3**. Utilisez **exactement 3.12** :
>
> - **Python 3.13 / 3.14** : `face_recognition` n'a pas de wheel dlib fiable → échec probable.
> - **Python 3.10 / 3.11** : fonctionne, mais les wheels testés par l'équipe sont pour cp312.
>
> Pour vérifier vos versions installées : `py --list`

> ⚠️ **Point critique Windows** : `face_recognition` dépend de **dlib**. Contrairement à la
> croyance courante, **dlib n'a AUCUN wheel précompilé pour Windows sur PyPI officiel** — un simple
> `pip install dlib` tente donc de **compiler depuis les sources** et échoue sans Visual C++.
> La solution retenue par l'équipe est d'installer un **wheel communautaire** (voir §2, étape 3).
> **À vérifier le Jour 1 par toute l'équipe** pour éviter tout blocage.

---

## 2. Installation pas à pas (procédure validée ✅)

> Cette procédure a été **testée avec succès** sur Windows 11 x64. Suivez les étapes **dans l'ordre**.

### Étape 1 — Créer et activer un environnement virtuel avec **Python 3.12**

```powershell
# Depuis la racine du projet
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Sous **CMD** : `.venv\Scripts\activate.bat`
> Sous **Git Bash** : `source .venv/Scripts/activate`

> ⚠️ Utilisez impérativement `py -3.12` et **non** `python` (qui pointe souvent vers 3.13/3.14).

### Étape 2 — Mettre à jour pip et installer `setuptools<81`

```powershell
python -m pip install --upgrade pip wheel
python -m pip install "setuptools<81"
```

> 💡 **Pourquoi `setuptools<81` ?** `face_recognition_models` importe `pkg_resources`, qui a été
> retiré des setuptools ≥ 81. Sans cette étape, l'import échoue avec `ModuleNotFoundError`.

### Étape 3 — Installer **dlib via le wheel communautaire** (étape indispensable)

Téléchargez le wheel précompilé pour Python 3.12 :

```powershell
# Téléchargement direct du wheel
Invoke-WebRequest `
  -Uri "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl" `
  -OutFile "dlib-19.24.99-cp312-cp312-win_amd64.whl"

# Installation du wheel (aucune compilation !)
python -m pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
```

> ✅ Cette méthode **évite totalement** l'installation de Visual Studio C++ (~7 Go).
> 📚 Source du wheel : [z-mahmud22/Dlib_Windows_Python3.x](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)

### Étape 4 — Installer le reste des dépendances

```powershell
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
```

> ⚠️ **Si vous utilisez un miroir pip** (ex. Tsinghua, Aliyun) : il ne fournit pas toujours les
> wheels Windows. Forcez PyPI officiel avec `--index-url https://pypi.org/simple`.

### Étape 5 — Vérifier l'installation

```powershell
python -c "import numpy, dlib, cv2, PIL, customtkinter, face_recognition as fr; print('Python:', __import__('sys').version.split()[0]); print('numpy:', numpy.__version__); print('dlib:', dlib.__version__); print('cv2:', cv2.__version__); print('=== OK ===')"
```

**Résultat attendu :**

```
Python: 3.12.3
numpy: 1.26.4
dlib: 19.24.99
cv2: 4.10.0
=== OK ===
```

> ⚠️ Un avertissement `UserWarning: pkg_resources is deprecated` peut apparaître : il est
> **inoffensif**, l'import reste fonctionnel.

---

## 3. Liste détaillée des dépendances

> ✅ Versions **exactes testées et validées** par l'équipe. Ne pas les modifier sans raison :
> les contraintes croisées (dlib ↔ numpy ↔ opencv) sont fragiles.

| Bibliothèque                | Version validée    | Rôle dans le projet                                                                                                                       | Module(s) concerné(s)                              |
| :-------------------------- | :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------- |
| **face_recognition**        | `1.3.0`            | Détection (`face_locations`), encodage 128-D (`face_encodings`), distance (`face_distance`), comparaison (`compare_faces`). Cœur de l'IA. | `face_encoder.py`, `face_matcher.py`, `main.py`    |
| **face-recognition-models** | `0.3.0`            | Modèles pré-entraînés (ResNet 128-D, prédicteurs de points, détecteur CNN). ~100 Mo.                                                      | (transitive de `face_recognition`)                 |
| **dlib**                    | `19.24.99`         | Moteur sous-jacent de `face_recognition` (HOG + SVM + ResNet). **Via wheel communautaire Windows.**                                       | (transitive)                                       |
| **numpy**                   | `1.26.4`           | Manipulation des tableaux de pixels et des vecteurs d'encodage (128-D). **Doit rester en 1.x** (voir §6.3).                                | Tous les modules IA                                |
| **opencv-python**           | `4.10.0.84`        | Capture webcam (`VideoCapture`), conversion BGR↔RGB, redimensionnement, dessin des bounding boxes. **Dernière version compatible numpy 1.x.** | `camera_stream.py`, `capture_dataset.py`, `gui.py` |
| **Pillow**                  | `12.3.0`           | Conversion des frames numpy → images affichables (`CTkImage`/`ImageTk`), chargement d'images statiques.                                   | `gui.py`                                           |
| **customtkinter**           | `6.0.0`            | Interface graphique moderne (thème sombre), boutons, labels image, panneaux.                                                              | `gui.py`, `main.py`                                |
| **setuptools**              | `<81`              | Fournit `pkg_resources`, requis par `face_recognition_models`.                                                                            | (installation)                                     |
| **click**                   | `>=6.0`            | Dépendance CLI de `face_recognition`.                                                                                                     | (transitive)                                       |
| **darkdetect**              | dernière           | Détection du thème système, utilisée par CustomTkinter.                                                                                   | (transitive)                                       |
| **tkinter**                 | inclus avec Python | Backend de Tkinter utilisé par CustomTkinter. Généralement fourni avec Python.                                                            | `gui.py`                                           |
| **pickle**                  | stdlib             | Sérialisation des encodages dans `encoded_faces.pkl`.                                                                                     | `face_encoder.py`, `face_matcher.py`               |
| **os / pathlib / json**     | stdlib             | Parcours des dossiers, gestion des chemins, métadonnées.                                                                                  | `dataset_manager.py`                               |

---

## 4. Fichier `requirements.txt`

Contenu réel du fichier livré (versions épinglées pour garantir la reproductibilité) :

```txt
# =============================================================================
#  requirements.txt — Projet Reconnaissance Faciale
#  Versions TESTÉES ET VALIDÉES sur Windows 11 x64 / Python 3.12.3
# =============================================================================
#
#  ⚠️ INSTALLER dlib AVANT ce fichier (voir §2, étape 3) :
#     python -m pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
#
# =============================================================================

# --- Cœur IA / reconnaissance faciale ---
face_recognition==1.3.0
face-recognition-models==0.3.0
dlib==19.24.99

# --- Traitement d'image / vidéo ---
numpy==1.26.4                 # ⚠️ rester en 1.x (incompatible numpy 2.x avec dlib)
opencv-python==4.10.0.84      # ⚠️ dernière version compatible numpy 1.x
Pillow==12.3.0

# --- Interface graphique ---
customtkinter==6.0.0

# --- Dépendances internes requises ---
setuptools<81                 # fournit pkg_resources (requis par face_recognition_models)
click>=6.0
darkdetect
```

> ⚠️ **Contraintes de versions interdépendantes** (ne pas modifier isolément) :
>
> - `dlib 19.24.99` **exige** `numpy 1.x` → ne jamais passer à numpy 2.x.
> - `opencv-python ≥ 4.14` **exige** `numpy 2.x` → d'où l'épinglage à `4.10.0.84`.
> - `setuptools ≥ 81` **supprime** `pkg_resources` → d'où l'épinglage à `setuptools<81`.

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

> Les 5 problèmes ci-dessous ont été **réellement rencontrés et résolus** lors de la mise en place
> du projet. Ils sont classés par ordre d'apparition.

### 6.1 ❌ `ERROR: Could not build wheels for dlib` (le blocage principal)

**Message typique :**

```
Building wheel for dlib (pyproject.toml) ... error
CMake Error at CMakeLists.txt:5 (message):
  You must use Visual Studio to build a python extension on windows.
  ...You need to install Visual Studio for C++.
ERROR: Failed building wheel for dlib
```

**Cause réelle (3 facteurs cumulés) :**

1. **dlib n'a aucun wheel Windows sur PyPI officiel** — pip tente donc de le **compiler depuis les
   sources** (fichier `.tar.gz`).
2. La compilation nécessite **Visual C++** (Visual Studio Build Tools), absent de la machine.
3. Il peut s'ajouter un **miroir pip** (ex. Tsinghua) qui ne sert pas les wheels Windows.

**✅ Solution retenue — wheel communautaire précompilé (recommandée) :**

```powershell
# 1. Télécharger le wheel pour Python 3.12 (cp312)
Invoke-WebRequest `
  -Uri "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl" `
  -OutFile "dlib-19.24.99-cp312-cp312-win_amd64.whl"

# 2. Installer le wheel (aucune compilation)
python -m pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
```

> 📚 Source : [z-mahmud22/Dlib_Windows_Python3.x](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
> (wheels disponibles pour Python 3.7 → 3.14).

**🔧 Solution alternative — compiler avec Visual Studio :**

1. Installer **Visual Studio Build Tools 2022** avec la charge de travail
   « **Développement Desktop en C++** » (composants *MSVC v143* + *Windows 10/11 SDK*). ~7 Go.
2. `python -m pip install dlib --index-url https://pypi.org/simple`

> ⚠️ Cette méthode est lourde et **échoue fréquemment sur Python 3.13+**. Préférez le wheel.

### 6.2 ❌ `No module named 'pkg_resources'`

**Message typique :**

```
File "...face_recognition_models\__init__.py", line 7, in <module>
    from pkg_resources import resource_filename
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause** : `pkg_resources` faisait partie de `setuptools`, mais a été **retiré des versions
récentes** (`setuptools ≥ 81`). `face_recognition_models` en dépend encore.

**✅ Solution :**

```powershell
python -m pip install "setuptools<81"
```

> 💡 Un avertissement `UserWarning: pkg_resources is deprecated` subsistera : il est **normal et
> inoffensif** tant que `setuptools<81` est installé.

### 6.3 ❌ `RuntimeError: Unsupported image type, must be 8bit gray or RGB image`

**Cause** : **numpy 2.x est incompatible avec dlib 19.24.99**. Le tableau numpy n'est plus reconnu
comme une image RGB valide par dlib, même avec un `dtype=uint8` correct.

**✅ Solution — rétrograder numpy en 1.x :**

```powershell
python -m pip install "numpy==1.26.4" --index-url https://pypi.org/simple
```

> ⚠️ **Effet domino** : `opencv-python ≥ 4.14` exige `numpy ≥ 2`. Après avoir rétrogradé numpy,
> il faut donc **aussi** épingler opencv :
>
> ```powershell
> python -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84" --index-url https://pypi.org/simple
> ```

### 6.4 ❌ Le miroir pip ne trouve pas les paquets

**Symptôme** : pip télécharge systématiquement des `.tar.gz` (sources) au lieu de `.whl`
(binaires), ou retourne `No matching distribution found`.

**Cause** : un miroir pip non officiel est configuré (ex. `pypi.tuna.tsinghua.edu.cn`).

**Vérifier la configuration :**

```powershell
python -m pip config list
```

**✅ Solution** — forcer PyPI officiel ponctuellement :

```powershell
python -m pip install <paquet> --index-url https://pypi.org/simple
```

### 6.5 ❌ Mauvaise version de Python (3.13 / 3.14)

**Symptôme** : l'erreur dlib apparaît malgré tout, ou des wheels incompatibles sont téléchargés.

**Vérifier :**

```powershell
python --version
py --list
```

**✅ Solution** — recréer le venv explicitement en **3.12** :

```powershell
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

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

| Élément                           | Valeur                                                                         |
| :-------------------------------- | :----------------------------------------------------------------------------- |
| Version Python                    | **3.12.x (x64) — exactement**                                                  |
| Environnement                     | virtuel (`.venv/`) créé avec `py -3.12 -m venv .venv`                           |
| IDE conseillé                     | VS Code + extension Python, ou PyCharm                                          |
| Notebook                          | Jupyter (`pip install notebook`) pour le livrable `exploration.ipynb`           |
| Index pip                         | PyPI officiel (`--index-url https://pypi.org/simple` si miroir configuré)       |
| Fichiers à ignorer (`.gitignore`) | `.venv/`, `__pycache__/`, `*.pkl`, `data/`, `wheels/`, `*.whl`, `*.zip`        |

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
