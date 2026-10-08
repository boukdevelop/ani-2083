# 🏗️ Architecture du Projet — Reconnaissance Faciale

> Ce document décrit l'architecture logicielle du projet, les modules, leurs **contrats
> (interfaces)** et le flux de données. Il est aligné sur `week-1/`[explainHIm.md](explainHIm.md),
> `week-1/`[pré-requis.md](pré-requis.md) et `week-1/`[roadmap.md](roadmap.md).

---

## 1. Vue d'ensemble

Le système est un **pipeline séquentiel en 4 étapes** (détection → analyse → embedding → matching),
piloté par une interface graphique. Chaque étape correspond à un module Python indépendant, ce qui
permet à 5 étudiants de travailler en parallèle sans se bloquer.

```
┌──────────────────────────────────────────────────────────────────────┐
│                            main.py (Assemblage)                       │
│          Boucle d'actualisation GUI  ──►  after(15, update)           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
┌───────────────┐      ┌───────────────┐        ┌────────────────┐
│  gui.py [E5]  │◄────►│ camera_stream │        │ face_matcher   │
│ CustomTkinter │      │   .py  [E4]   │        │   .py   [E3]   │
│  (affichage)  │      │  (OpenCV)     │        │ (décision)     │
└───────┬───────┘      └───────┬───────┘        └───────┬────────┘
        │                      │                        │
        │              frame RGB + frame réduite         │
        │                      │                        │
        │                      ▼                        ▼
        │              ┌───────────────┐        ┌────────────────┐
        └─────────────►│  Détection +  │───────►│ encoded_faces  │
                       │  Encodage     │        │     .pkl       │
                       │ (face_recog.) │        │  [E2] produit  │
                       └───────┬───────┘        └───────┬────────┘
                               │                        │
                               ▼                        │
                       ┌───────────────┐                │
                       │ face_encoder  │◄───────────────┘
                       │    .py  [E2]  │   (génère la base hors-ligne)
                       └───────┬───────┘
                               ▲
                               │
                       ┌───────────────┐      ┌──────────────────┐
                       │ dataset_manager│◄────│ capture_dataset  │
                       │    .py  [E1]   │      │   .py   [E1]     │
                       └───────┬───────┘      └────────┬─────────┘
                               │                        │
                               ▼                        ▼
                       data/known_faces/  ◄────  webcam (capture)
```

---

## 2. Découpage en modules et contrats (interfaces)

> Chaque module expose une **API publique stable**. Les autres modules ne doivent dépendre
> que de ces signatures, jamais des détails internes. Cela permet le travail en parallèle.

### 2.1 `camera_stream.py` — Flux vidéo `[Étudiant 4]`

**Rôle** : wrapper OpenCV autour du matériel caméra et de la conversion d'images.

**Détails clés** :

- Réduction à **25 %** (`scale = 0.25`) pour accélérer la détection.
- Le **facteur d'échelle** (`1/scale = 4`) est renvoyé pour **re-projeter les coordonnées**
  des bounding boxes sur l'image pleine résolution.
- Conversion **BGR → RGB** via `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.

---

### 2.2 `dataset_manager.py` — Gestion & validation du dataset `[Étudiant 1]`

**Rôle** : garantir que le dataset est exploitable avant l'encodage.

**Détails clés** :

- Formats acceptés : `.jpg`, `.jpeg`, `.png`.
- Vérifie la **résolution minimale** et la **présence d'au moins 20 images**.
- Retourne un **rapport structuré** (dict) exploitable par la GUI ou les scripts.

---

### 2.3 `capture_dataset.py` — Script de capture webcam `[Étudiant 1]`

**Rôle** : application utilitaire pour constituer le dataset.

**Comportement** :

- Ouvre la webcam, affiche un aperçu live avec le compteur d'images.
- Touche `ESPACE` → capture 1 image ; touche `C` → capture automatique de 20-40 images.
- Touche `Q` → quitter.
- Crée automatiquement `data/known_faces/<nom_etudiant>/` et nomme `img_001.jpg`, `img_002.jpg`, …
- Vérifie à la volée qu'un visage est détecté avant de sauvegarder.


---

### 2.4 `face_encoder.py` — Moteur d'embedding `[Étudiant 2]`

**Rôle** : transformer le dataset en base d'empreintes numériques (hors-ligne).


**Détails clés** :

- Utilise `face_recognition.face_encodings()` → vecteurs de **128 dimensions**.
- Sérialisation via `pickle` dans `encoded_faces.pkl` (encodages + noms + métadonnées de build).
- Ignore proprement les images sans visage et les fichiers corrompus (log + compteur).
- Génère un résumé : images traitées / encodages valides / échecs.

---

### 2.5 `face_matcher.py` — Moteur de reconnaissance `[Étudiant 3]`

**Rôle** : décider à qui appartient un visage détecté, ou rejeter (« Unknown »).

**Détails clés** :

- Charge `encoded_faces.pkl` **une seule fois** au démarrage (pas de ré-encodage).
- `face_recognition.face_distance()` → distance euclidienne au plus proche voisin.
- Règle de décision :
  - `distance < TOLERANCE` → nom trouvé, `confiance = (1 - distance) * 100`.
  - `distance ≥ TOLERANCE` → `"Unknown / Inconnu"`, score faible.
- `TOLERANCE` est une **constante unique** facilement ajustable (J5).

---

### 2.6 `gui.py` — Interface CustomTkinter & annotation `[Étudiant 5]`

**Rôle** : interface utilisateur, dessin des bounding boxes, mise à jour dynamique.


**Détails clés** :

- Thème sombre (`ctk.set_appearance_mode("dark")`).
- Boutons : **Démarrer Webcam**, **Charger une Image**, **Arrêter**.
- Zone vidéo : `CTkLabel` alimenté par une image `CTkImage`/`ImageTk` (Pillow).
- Panneau latéral : nom + score + statut.
- **Annotation** : box verte si reconnu, rouge si Unknown, texte « Nom — 87 % » au-dessus.
- **Reprojection des coordonnées** : les boxes sont calculées sur la frame réduite (25 %),
  il faut multiplier par `1/scale` (= 4) avant de dessiner sur la frame pleine résolution.

---

### 2.7 `main.py` — Point d'entrée & assemblage `[Étudiant 5]`

**Rôle** : instancier les composants, lancer la boucle principale.

**Boucle d'actualisation** (cœur de l'intégration) :

```
1. frame, factor = camera.read_small()            # 25 % pour l'analyse
2. face_locations = face_recognition.face_locations(frame)
3. face_encodings = face_recognition.face_encodings(frame, face_locations)
4. résultats = matcher.match_many(face_encodings)
5. re-projeter les boxes (× 1/factor) sur la frame pleine résolution
6. annoter (draw_annotation) et afficher dans la GUI
7. self.after(15, self._update_loop)              # ~60 FPS max, non bloquant
```

---

## 3. Flux de données (pipeline)

### 3.1 Phase hors-ligne (préparation, J1 → J2)

```
Webcam ──► capture_dataset.py ──► data/known_faces/<nom>/*.jpg
                                        │
                                        ▼
                              dataset_manager.validate_dataset()
                                        │  (OK)
                                        ▼
                              face_encoder.build_encodings()
                                        │
                                        ▼
                              encoded_faces.pkl  (128-D + noms)
```

### 3.2 Phase temps réel (exécution, J4 → J5)

```
CameraStream.read_small()  ──►  frame 25 % (RGB)
        │
        ├─► face_recognition.face_locations()   ──► bounding boxes
        ├─► face_recognition.face_encodings()   ──► vecteurs 128-D
        │
        ▼
FaceMatcher.match_many()  ──►  [(nom, score), ...]
        │
        ▼
draw_annotation() + re-projection ×4  ──►  frame annotée
        │
        ▼
GUI (CTkLabel) + panneau latéral  ──►  affichage utilisateur
```

---

## 4. Structure des fichiers

```
Projet_1_Reconnaissance_Faciale/
├── data/
│   ├── known_faces/              # 20-40 photos par étudiant (base d'entraînement)
│   │   ├── etudiant_1/
│   │   ├── etudiant_2/
│   │   └── ...
│   └── unknown_tests/            # Images de test (personnes externes)
├── notebooks/
│   └── exploration.ipynb         # Notebook d'expérimentation (livrable)
├── dataset_manager.py            # [E1] Validation du dataset
├── capture_dataset.py            # [E1] Capture webcam
├── face_encoder.py               # [E2] Embeddings + sérialisation
├── face_matcher.py               # [E3] Matching + seuil + Unknown
├── camera_stream.py              # [E4] Flux vidéo OpenCV
├── gui.py                        # [E5] Interface CustomTkinter + annotation
├── main.py                       # [E5] Assemblage / point d'entrée
├── encoded_faces.pkl             # Généré par face_encoder.py
├── requirements.txt              # Dépendances
├── README.md                     # Installation & utilisation
└── Rapport_Projet_1.pdf          # Rapport (3-5 pages)
```

---

## 5. Choix d'architecture : justification

| Choix                                        | Justification                                                 |
| :------------------------------------------- | :------------------------------------------------------------ |
| **Découpage en 6 modules**                   | Un module = un étudiant → travail parallèle sans conflit Git  |
| **Contrats/signatures stables**              | Intégration possible en J4 sans réécrire les modules          |
| **Base d'encodages pré-calculée (`pickle`)** | Évite de ré-encoder 100-200 images à chaque démarrage         |
| **Traitement sur frame réduite (25 %)**      | Accélère la détection, garde l'affichage fluide               |
| **Reprojection des coordonnées**             | Box précises sur l'image pleine résolution sans ralentir l'IA |
| **Seuil `TOLERANCE` unique**                 | Calibrage simple et centralisé (J5)                           |
| **Séparation UI / logique**                  | La GUI ne connaît que `FaceMatcher.match()` et `CameraStream` |
| **Boucle `after()` non bloquante**           | L'UI reste réactive pendant le traitement                     |

---

## 6. Contraintes & limites connues

- **Précision** : `face_recognition` est limité sur les profils et conditions difficiles
  (voir `pré-requis.md`). Pas de détection de vivacité (anti-spoofing).
- **Performance** : tourne sur CPU ; pas de GPU requis pour ce volume de données.
- **Éthique / RGPD** : consentement écrit obligatoire, usage pédagogique, minimisation,
  suppression des données en fin d'année scolaire.
- **Sécurité** : `encoded_faces.pkl` contient des **données biométriques** → à ne pas
  versionner publiquement (à ajouter dans `.gitignore`).
