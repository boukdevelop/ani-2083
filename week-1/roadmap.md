# 🗺️ Roadmap — Projet 1 : Reconnaissance Faciale (Semaine 1)

> **Objectif** : livrer une application Python (webcam + image) qui détecte un visage, l'identifie
> parmi les étudiants enregistrés et retourne « Unknown / Inconnu » si la confiance est insuffisante.
> **Durée** : 6 jours de travail (06 → 12 octobre 2026, J7 = tampon / démo).
> **Base conceptuelle** : voir `week-1/explainHIm.md` et `week-1/pré-requis.md`.

---

## 👥 1. Répartition sommaire des tâches

| Membre                          | Module principal        | Fichier(s)                                 | Responsabilité fonctionnelle                                                        | Livrable hors-code                                     |
| :------------------------------ | :---------------------- | :----------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Étudiant 1** — Chef de projet | Dataset & Admin         | `capture_dataset.py`, `dataset_manager.py` | Capture webcam (20-40 photos/personne), validation du dataset                       | Rapport (3-5 p.), coordination, note RGPD/consentement |
| **Étudiant 2**                  | Encodage / IA           | `face_encoder.py`                          | Extraction des embeddings 128-D, sérialisation `encoded_faces.pkl`                  | Documentation technique du moteur                      |
| **Étudiant 3**                  | Matching / Unknown      | `face_matcher.py`                          | Comparaison temps réel, distance euclidienne, seuil `TOLERANCE`, score de confiance | Protocole de tests (faux positifs, seuil)              |
| **Étudiant 4**                  | Flux vidéo              | `camera_stream.py`                         | Wrapper OpenCV (`CameraStream`), redimensionnement 25 %, BGR→RGB                    | Fiche technique caméra                                 |
| **Étudiant 5**                  | Interface / Intégration | `gui.py`, `main.py`                        | UI CustomTkinter (thème sombre), bounding box + nom + score, boucle `after()`       | Capture d'écran / vidéo démo                           |

**Principe de collaboration** : chaque étudiant écrit du **code Python** dans son module, avec une
**interface (contrat) claire** entre les modules (voir `architecture.md`). Aucun module ne doit
dépendre de l'implémentation interne d'un autre — seulement de ses signatures publiques.

---

## 📋 2. Liste ordonnée des tâches (J1 → J6)

> Convention : `[Ex]` = étudiant responsable · `⏱` = durée estimée · `🔗` = dépendance.

### 🔹 J1 — 06 Oct : Configuration & Collecte `[Étudiant 1 & 4]`

**But** : avoir un environnement qui tourne et un dataset brut collecté.

- [ ] `[Tous]` Créer l'environnement virtuel : `python -m venv .venv` puis activation.
- [ ] `[Tous]` Installer les dépendances (voir `dependances.md`) : `pip install -r requirements.txt`.
- [ ] `[Tous]` **Vérifier dlib/face_recognition** : tester `import face_recognition`. Si échec → installer
      Visual Studio C++ Build Tools (voir `dependances.md` §Dépannage).
- [ ] `[E1]` Rédiger la **note d'information + autorisation d'usage d'image** (consentement RGPD explicite).
- [ ] `[E1]` Créer la structure de dossiers : `data/known_faces/`, `data/unknown_tests/`.
- [ ] `[E1]` Coder `capture_dataset.py` :
  - [ ] Ouvrir la webcam (`cv2.VideoCapture(0)`).
  - [ ] Touche `ESPACE` = capturer 1 image ; touche `C` = capture auto de 20-40 images.
  - [ ] Créer automatiquement `data/known_faces/<nom_etudiant>/`.
  - [ ] Sauver au format `.jpg`, nommage `img_001.jpg`, `img_002.jpg`, …
  - [ ] Afficher un compteur d'images capturées à l'écran.
- [ ] `[E1]` Coder la **validation du dataset** dans `dataset_manager.py` :
  - [ ] Vérifier le format (`.jpg` / `.png` uniquement).
  - [ ] Vérifier la résolution minimale (ex. ≥ 200×200 px).
  - [ ] Vérifier la présence d'**au moins 20 images** par dossier.
  - [ ] Retourner un rapport clair (OK / avertissements / erreurs).
- [ ] `[E4]` Coder la **classe `CameraStream`** (squelette) dans `camera_stream.py` :
  - [ ] `open()`, `read_frame()`, `release()`.
  - [ ] Redimensionnement à **25 %** pour l'analyse (factor `0.25`).
  - [ ] Conversion **BGR → RGB** (`cv2.cvtColor`).
- [ ] `[E1+E4]` **Collecter 100-200 images** (5 étudiants × 20-40) avec variabilité contrôlée
      (orientations, expressions, distances, éclairages).
- [ ] `[E1]` Capturer les **images de test d'une personne absente** dans `data/unknown_tests/`.

**✅ Fin de J1** : dataset collecté + validé, environnement opérationnel, `camera_stream.py` fonctionnel.

---

### 🔹 J2 — 07 Oct : Moteur d'Encodage & Cœur IA `[Étudiant 2 & 3]`

**But** : transformer les images en empreintes numériques et définir la logique de décision.

- [ ] `[E2]` Coder `face_encoder.py` :
  - [ ] Parcourir `data/known_faces/` (un sous-dossier = une personne).
  - [ ] Charger chaque image, la convertir en RGB.
  - [ ] Calculer `face_recognition.face_encodings()` → vecteur de **128 dimensions**.
  - [ ] Associer chaque encodage au **nom** de la personne.
  - [ ] Sérialiser dans `encoded_faces.pkl` avec `pickle` (encodages + noms + métadonnées).
  - [ ] Gérer les cas d'erreur : image sans visage détecté, image corrompue.
  - [ ] Afficher un résumé : nombre d'images traitées, nombre d'encodages valides.
- [ ] `[E3]` Coder `face_matcher.py` :
  - [ ] Charger `encoded_faces.pkl` au démarrage (éviter de ré-encoder à chaque lancement).
  - [ ] Fonction `match(face_encoding)` :
    - [ ] Calculer `face_recognition.face_distance()` (distance euclidienne).
    - [ ] Définir `TOLERANCE = 0.50` (constante ajustable).
    - [ ] Si `distance < TOLERANCE` → retourner `(nom, score_confiance)`.
    - [ ] Sinon → retourner `("Unknown / Inconnu", score_faible)`.
  - [ ] Formule du score : `confiance = (1 - distance) * 100` (borné à 0-100 %).
- [ ] `[E2+E3]` **Test unitaire** de la logique distance/seuil :
  - [ ] Comparer 2 encodages d'une même personne → distance faible → identifié.
  - [ ] Comparer 2 encodages de personnes différentes → distance élevée → rejeté.
  - [ ] Tester plusieurs valeurs de `TOLERANCE` (0.45 / 0.50 / 0.55) et noter l'impact.

**✅ Fin de J2** : `encoded_faces.pkl` généré + logique de matching testée en isolation.

---

### 🔹 J3 — 08 Oct : Interface CustomTkinter `[Étudiant 5]`

**But** : disposer d'une UI fonctionnelle avant de brancher la reconnaissance.

- [ ] `[E5]` Coder `gui.py` — squelette de l'application :
  - [ ] Fenêtre principale CustomTkinter, **thème sombre** (`ctk.set_appearance_mode("dark")`).
  - [ ] Bouton **« Démarrer Webcam »**, **« Charger une Image »**, **« Arrêter »**.
  - [ ] Zone d'affichage vidéo via `CTkLabel` (image `CTkImage`/`ImageTk`).
  - [ ] **Panneau latéral** : résultats (nom, score de confiance, statut Unknown).
- [ ] `[E5]` Coder les **fonctions d'annotation graphique** :
  - [ ] Tracer la **bounding box** autour du visage (`cv2.rectangle` ou dessin Pillow).
  - [ ] Afficher **nom + score** au-dessus du visage (`cv2.putText`).
  - [ ] Couleur différente pour un visage reconnu (vert) vs Unknown (rouge/orange).
- [ ] `[E5]` Coder le **chargement d'image statique** (`CTkImage` depuis un fichier).
- [ ] `[E5]` Préparer la **boucle d'actualisation** `after()` (squelette, non branchée à la caméra).

**✅ Fin de J3** : UI complète visible et navigable, sans logique de reconnaissance.

---

### 🔹 J4 — 09 Oct : Intégration du Pipeline Complet `[Tous]`

**But** : connecter flux vidéo + encodage + matching + UI dans `main.py`.

- [ ] `[E5+E4]` Brancher `CameraStream` sur le `CTkLabel` de la GUI.
- [ ] `[E2+E3]` Charger `encoded_faces.pkl` au démarrage et exposer l'API de matching à la GUI.
- [ ] `[E5]` Coder `main.py` : assemblage complet.
  - [ ] Boucle `after(15, update)` : lire frame → détecter → matcher → annoter → afficher.
  - [ ] Redimensionner la frame à 25 % pour la détection, puis **remettre les coordonnées à l'échelle**
        (×4) pour tracer la box sur l'image pleine résolution.
  - [ ] Mettre à jour le panneau latéral dynamiquement (nom + score).
  - [ ] Bouton « Arrêter » → `release()` propre de la caméra.
- [ ] `[Tous]` **Premier test bout-en-bout** devant la webcam avec un étudiant de la base.

**✅ Fin de J4** : application fonctionnelle de bout en bout (webcam → nom affiché).

---

### 🔹 J5 — 10 Oct : Batterie de Tests & Intrus `[Étudiant 3 & 5]`

**But** : valider la robustesse et calibrer le seuil.

- [ ] `[E3]` **Test personne absente** : passer les images de `data/unknown_tests/` → doit afficher
      « Unknown / Inconnu ». Mesurer le **taux de faux positifs**.
- [ ] `[E3]` **Ajustement du seuil `TOLERANCE`** : tester 0.45 / 0.50 / 0.55, choisir la meilleure
      valeur (compromis faux positifs / faux négatifs).
- [ ] `[E3]` **Tests de luminosité** : faible éclairage, contre-jour, lumière forte → noter les échecs.
- [ ] `[E3]` **Tests de pose** : face, trois-quarts, profil → noter la dégradation.
- [ ] `[E5]` **Tests d'interface** : charger une image statique, vérifier l'affichage du score et des boxes.
- [ ] `[E3+E5]` Calculer les **métriques** (Précision, Rappel, F1-score, AUC) sur un jeu de test annoté.
- [ ] `[Tous]` Corriger les bugs identifiés et documenter les limites connues.

**✅ Fin de J5** : système testé, seuil calibré, limites documentées.

---

### 🔹 J6 — 11 Oct : Finalisation, Fichiers Annexes & Packaging `[Étudiant 1 & 2]`

**But** : livrer un projet propre, documenté et empaqueté.

- [ ] `[E1]` **Nettoyage du code** : commentaires, suppression du code mort, noms cohérents.
- [ ] `[E2]` Créer le **`requirements.txt`** final (versions épinglées).
- [ ] `[E2]` Créer le **notebook d'entraînement / expérimentation** (`notebooks/exploration.ipynb`).
- [ ] `[E1]` Rédiger le **`README.md`** : installation, utilisation, structure, limites.
- [ ] `[E1]` Rédiger le **rapport (3-5 pages)** : contexte, méthode, résultats, éthique/RGPD, limites.
- [ ] `[E1+E2]` **Packaging** : créer le `.zip` du projet (code + dataset + rapport).
- [ ] `[Tous]` Enregistrer la **vidéo démo** (webcam + reconnaissance + cas Unknown).
- [ ] `[Tous]` **Revue finale** : vérifier que tous les livrables de `explainHIm.md` sont présents.

**✅ Fin de J6** : projet complet, documenté, testé et livré.

---

## 🎯 3. Critères de réussite (Definition of Done)

| Critère                 | Vérification                                                         |
| :---------------------- | :------------------------------------------------------------------- |
| Webcam ou image chargée | L'app affiche un flux vidéo ou une image                             |
| Détection de visage     | Bounding box visible sur chaque visage                               |
| Identification correcte | Nom affiché pour un étudiant de la base                              |
| Score de confiance      | Pourcentage affiché à côté du nom                                    |
| Cas Unknown             | Personne absente → « Unknown / Inconnu »                             |
| Dataset organisé        | `data/known_faces/<nom>/` avec ≥ 20 images validées                  |
| Livrables               | Code, dataset, notebook, `requirements.txt`, `README`, rapport, démo |

---

## 📦 4. Rappel des livrables finaux (issus de `explainHIm.md`)

1. Code Python complet (6 modules + `main.py`)
2. Dataset organisé (5 étudiants × 20-40 images)
3. Notebook d'entraînement / expérimentation
4. `requirements.txt`
5. `README.md` (installation + utilisation)
6. Rapport court de 3 à 5 pages
7. Captures d'écran ou courte démonstration vidéo
