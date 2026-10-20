# AI Sketch Recognizer

Application interactive de reconnaissance de croquis en temps reel basee sur un reseau de neurones convolutif (CNN) entraine sur un sous-ensemble du jeu de donnees Google Quick, Draw!.

## 1. Description du projet

L'application permet a un utilisateur de dessiner a la main levee sur un canvas interactif. L'image tracee est pretraitee (inversion, centrage par boite englobante, redimensionnement 28x28) puis classee en temps reel par un modele CNN parmi 10 categories d'objets.

Categories reconnues :

- cat (chat)
- house (maison)
- car (voiture)
- tree (arbre)
- sun (soleil)
- umbrella (parapluie)
- apple (pomme)
- airplane (avion)
- fish (poisson)
- star (etoile)

## 2. Structure du projet

- `config.py` : Parametres globaux, chemins et categories.
- `prepare_data.py` : Telechargement des donnees brutes .npy, equilibrage, normalisation et split Train/Val/Test.
- `dataset_loader.py` : Utilitaire de chargement du dataset traite (dataset.npz).
- `model_builder.py` : Architecture du modele CNN TensorFlow/Keras.
- `train_model.py` : Entrainement, callbacks et export du modele .keras et de classes.json.
- `evaluate.py` : Evaluation du modele (matrice de confusion, rapport de classification).
- `inference_engine.py` : Pipeline de pretraitement du dessin et prediction Top-K.
- `ui_app.py` : Interface graphique utilisateur (PyQt6).
- `bonus_filters.py` : Filtres artistiques OpenCV et export dans la galerie.
- `main.py` : Point d'entree de l'application.

## 3. Installation

### Pre-requis

- Python 3.10 a 3.12 recommande
- pip

### Installation des dependances

```bash
pip install -r requirements.txt
```

Principales bibliotheques requises :

- tensorflow
- PyQt6
- opencv-python
- numpy
- pillow
- scikit-learn
- matplotlib

## 4. Utilisation

### Lancer l'application

```bash
python main.py
```

### Re-preparer les donnees (optionnel)

```bash
python prepare_data.py
```

### Re-entrainer le modele (optionnel)

```bash
python train_model.py
```

### Evaluer le modele

```bash
python evaluate.py
```

## 5. Fonctionnalites de l'interface

- Zone de dessin interactive a la souris avec gestion de l'epaisseur et choix des couleurs.
- Reconnaissance instantanee avec affichage des 3 meilleures predictions et de leurs scores de confiance.
- Bouton d'effacement complet du canvas.
- Sauvegarde du dessin dans le dossier static/gallery/.
- Effet artistique esquisse via OpenCV.
