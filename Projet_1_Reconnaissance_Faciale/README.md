# Systeme de Reconnaissance Faciale Temps Reel

Projet academique (UE ANI-2083 — ENSPY Yaounde) implementant un pipeline complet de capture, encodage et reconnaissance faciale en temps reel sur flux video webcam.

## 1. Description du projet

Ce systeme permet d'identifier des personnes connues a partir du flux video de la webcam en comparant les descripteurs faciaux (embeddings 128-D) a une base locale prealablement encodee. Si le visage correspond a une personne enregistree avec une distance inferieure au seuil de tolerance (par defaut 0.50), son nom et son indice de confiance s'affichent dans un cadre vert. Dans le cas contraire, le visage est etiquete "Inconnu" dans un cadre rouge.

L'application dispose d'une interface graphique moderne realisee avec CustomTkinter, affichant la video traitee et permettant d'effectuer des captures d'ecran.

## 2. Structure du projet

- `main.py` : Point d'entree principal reliant le flux camera, le matching facial et l'interface graphique.
- `camera_stream.py` : Gestion du flux webcam OpenCV (effet miroir, reduction d'echelle pour optimisation des performances, conversion BGR vers RGB).
- `capture_dataset.py` : Script interactif guidant la prise de 20 poses et expressions pour enregistrer une nouvelle personne.
- `dataset_manager.py` : Audit et validation de qualite du dataset (verification de la presence d'exactement 1 visage par photo).
- `face_encoder.py` : Parcours du dossier d'images, extraction des embeddings 128-D via dlib / face_recognition et serialisation dans `encoded_faces.pkl`.
- `face_matcher.py` : Moteur de comparaison vectorielle (calcul des distances euclidiennes et determination du score de confiance).
- `gui.py` : Interface utilisateur graphique basee sur CustomTkinter.
- `consentement.md` : Formulaire d'autorisation et de respect de la vie privee signe par les participants.
- `data/known_faces/` : Repertoire contenant les photographies classees par sous-dossier de personne.
- `encoded_faces.pkl` : Fichier binaire contenant les encodages et les noms associes.

## 3. Installation

### Pre-requis

- Python 3.10 a 3.12 (sur Windows, Python 3.12 est recommande)
- Webcam fonctionnelle

### Installation de dlib sous Windows

`dlib` ne disposant pas de wheel precompile officiel sur PyPI pour les versions recentes de Python sur Windows, il convient d'installer le wheel communautaire avant les autres paquets :

1. Telecharger ou installer directement le wheel pour Python 3.12 :

```bash
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.99-cp312-cp312-win_amd64.whl
```

2. Installer ensuite les dependances du projet :

```bash
pip install -r requirements.txt
```

Principales bibliotheques requises :

- face_recognition
- dlib
- opencv-python
- numpy (version 1.26.4 recommandee pour compatibilite dlib)
- customtkinter
- pillow

## 4. Utilisation

### Etape 1 : Capturer de nouvelles photos (optionnel)

Pour enregistrer une nouvelle personne avec le guide interactif de 20 poses :

```bash
python capture_dataset.py
```

### Etape 2 : Verifier la qualite du dataset

Pour auditer les photos et s'assurer qu'un seul visage est visible par cliché :

```bash
python dataset_manager.py
```

### Etape 3 : Generer la base d'encodages

Pour extraire les descripteurs faciaux et produire `encoded_faces.pkl` :

```bash
python face_encoder.py
```

### Etape 4 : Lancer l'application en temps reel

```bash
python main.py
```

## 5. Fonctionnalites cles

- Detection et tracking de visages a haute frequence (saut d'images et redimensionnement optimise).
- Comparaison vectorielle robuste en espace 128-D (distance euclidienne).
- Cadre vert et pourcentage de certitude pour les visages reconnus, cadre rouge pour les inconnus.
- Interface graphique CustomTkinter avec fonction de capture d'image instantanee.
- Gestion ethique et securisee des donnees photographiques (conforme a consentement.md).
