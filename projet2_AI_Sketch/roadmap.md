# 🗺️ Roadmap Détaillée — Projet 2 : AI Sketch Recognizer (Quick, Draw! CNN)

> **Objectif du Projet** : Concevoir, entraîner et déployer une application interactive d'intelligence artificielle capable de reconnaître en temps réel des croquis dessinés à main levée par un utilisateur, à partir d'un sous-ensemble du jeu de données Google Quick, Draw! (5 à 10 catégories d'objets, ex. : _chat, maison, voiture, arbre, soleil, bicyclette, pomme, poisson, étoile, nuage_).
>
> **Durée** : 7 jours de sprint structuré. Date butoir : **Mardi 20 octobre 2026**.
>
> **Équipe** : 5 développeurs (100% Code, chaque membre implémente un module autonome avec un contrat d'interface documenté).

---

## 👥 1. Matrice RACI & Répartition Détaillée des Rôles (5 Membres)

| Rôle & Membre                                            | Module & Fichiers                                                                      | Responsabilités Techniques Principales                                                                                                                                                                                                                                                                                                                                                                                       | Livrables Attendus                                                                                  |
| :------------------------------------------------------- | :------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
| **Membre 1 (Toi)**<br>👑 **Chef de Projet & Core Logic** | `inference_engine.py`<br>`main.py`<br>`config.py`                                      | • Orchestration générale du projet et git merge.<br>• Pipeline d'inférence (prétraitement canvas → image $28 \times 28$ niveaux de gris normalisée).<br>• Chargement du modèle CNN (`.keras` / `.h5`) et prédiction avec scores Top-K et indice de confiance.<br>• Coordination inter-modules et gestion des configurations globales.                                                                                        | Code du moteur d'inférence validé, gestion de release Git, rapport de synthèse technique.           |
| **Membre 2**<br>📊 **Data Engineer**                     | `prepare_data.py`<br>`dataset_loader.py`<br>`data/`                                    | • Téléchargement automatisé des fichiers Quick, Draw! au format `.npy` (bitmap).<br>• Filtrage, équilibrage (ex. 5 000 à 10 000 échantillons par classe).<br>• Normalisation $[0, 1]$, encodage One-Hot des labels.<br>• Découpage stratifié Train / Validation / Test (70% / 15% / 15%).<br>• Sauvegarde des jeux de données prêts pour l'entraînement.                                                                     | Script ETL propre, jeu de données compact `data/processed/`, métriques de distribution des données. |
| **Membre 3**<br>🧠 **AI Engineer**                       | `model_builder.py`<br>`train_model.py`<br>`evaluate.py`                                | • Conception de l'architecture CNN (Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dropout, Dense, Softmax).<br>• Pipeline d'entraînement TensorFlow/Keras (Adam, categorical_crossentropy, EarlyStopping, ModelCheckpoint).<br>• Courbes d'apprentissage (Loss/Accuracy) et matrice de confusion scikit-learn.<br>• Export du modèle optimisé `models/sketch_cnn_model.keras` + dictionnaire des classes (`classes.json`). | Modèle entraîné (précision test > 90%), graphiques d'évaluation, script d'export des poids.         |
| **Membre 4**<br>🎨 **Frontend / UI Developer**           | `ui_app.py`<br>_(Option Desktop PyQt6 ou Web Flask)_<br>`templates/` / `static/`       | • Implémentation du Canvas interactif de dessin à la souris/stylet.<br>• Gestion des événements souris (MouseDown, MouseMove, MouseUp).<br>• Outils interactifs : Effacer le canvas, changer d'épaisseur de trait.<br>• Envoi de l'image au backend (requête locale ou API Flask JSON/Base64).<br>• Affichage visuel dynamique des prédictions (barres de probabilité Top-3 / Top-5).                                        | Interface fluide à haute réactivité (< 50ms latence d'affichage), canvas responsive.                |
| **Membre 5**<br>🛡️ **QA, Bonus & Features Engineer**     | `bonus_filters.py`<br>`tests/test_pipeline.py`<br>`tests/test_model.py`<br>`README.md` | • Implémentation des fonctionnalités bonus créativité : filtres artistiques OpenCV (effet esquisse, contours Canny, néon/cartoon).<br>• Galerie d'exportation d'images (PNG/JPEG) et historique des croquis.<br>• Suite de tests automatisés (vérification des dimensions d'entrée $28 \times 28$, robustesse aux entrées vides).<br>• Rédaction du guide d'installation et aide au rapport final.                           | Suite de tests `pytest`, modules bonus OpenCV, documentation utilisateur et de déploiement.         |

---

## 📅 2. Planning Opérationnel Jour par Jour (Semaine du Projet)

```
Lundi (J1)       Mardi (J2)        Mercredi (J3)       Jeudi (J4)          Vendredi (J5)       Samedi/Dim (J6-J7)  Mardi 20 Oct
┌──────────────┐ ┌───────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐
│ Cadrage &    │ │ Data & UI     │ │ Entraînement    │ │ Intégration     │ │ Bonus OpenCV    │ │ Documentation & │ │ Livrable &   │
│ Setup Git    │ │ Skeleton      │ │ CNN & Inférence │ │ Globale         │ │ Tests Finaux    │ │ Démo Vidéo      │ │ Clôture      │
└──────────────┘ └───────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘ └──────────────┘
```

### 🔹 Jour 1 (Lundi) : Cadrage Stratégique, Architecture & Initialisation

- **Objectif** : Aligner l'équipe sur les standards, figer les spécifications techniques et préparer l'environnement de développement collaboratif.
- **Tâches Membre 1 (Toi - Lead)** :
  - [ ] Définir la structure globale du dépôt GitHub (`projet2_AI_Sketch/`) avec les branches de travail (`main`, `develop`, `feature/*`).
  - [ ] Valider avec l'équipe le choix des **8 catégories Quick, Draw! retenues** : `cat`, `house`, `car`, `tree`, `sun`, `bicycle`, `apple`, `star`.
  - [ ] Établir le fichier de configuration central `config.py` (tailles d'images, noms de classes, hyperparamètres par défaut).
  - [ ] Rédiger les règles de Git Workflow (Pull Requests obligatoires, revues de code, conventions de nommage PEP 8).
- **Tâches Membre 2 (Data)** :
  - [ ] Vérifier la source de téléchargement Google Cloud Storage des fichiers `.npy` Quick, Draw!.
  - [ ] Mettre en place le script squelette `prepare_data.py` et le dossier de stockage local `data/raw/`.
- **Tâches Membre 3 (AI)** :
  - [ ] Établir le benchmark d'architecture CNN cible (dimension d'entrée `(28, 28, 1)`, nombre de couches, régularisation Dropout).
  - [ ] Configurer l'environnement TensorFlow / Keras (compatibilité GPU / CPU).
- **Tâches Membre 4 (UI)** :
  - [ ] Définir le framework d'interface (PyQt6 pour application Desktop réactive native OU Flask + HTML5 Canvas).
  - [ ] Maquetter le wireframe de l'écran principal : zone de dessin (canvas 280x280 ou 400x400), panneau de boutons, panneau de résultats.
- **Tâches Membre 5 (QA & Bonus)** :
  - [ ] Mettre en place le framework de test unitaire `pytest`.
  - [ ] Initialiser le gabarit du `README.md` avec badges et instructions d'installation.
- **Critère de validation J1** : Tous les membres clonent le dépôt, installent le `requirements.txt` sans erreur et s'accordent sur le contrat d'interface.

---

### 🔹 Jour 2 (Mardi) : Pipeline Données & Maquette Fonctionnelle de l'UI

- **Objectif** : Avoir les données prêtes pour l'entraînement et une interface capable de capturer un tracé souris.
- **Tâches Membre 1 (Toi - Lead)** :
  - [ ] Concevoir le module de prétraitement mathématique d'une image dessinée : extraction du canal alpha/niveaux de gris, bounding box du dessin pour centrage par centre de masse, redimensionnement bi-cubique en $28 \times 28$.
  - [ ] Examiner les premières fusions de code de M2 et M4.
- **Tâches Membre 2 (Data)** :
  - [ ] Coder la fonction de téléchargement automatique des 8 fichiers `.npy` depuis Google Storage.
  - [ ] Implémenter le filtrage à $10\,000$ dessins par classe (total $80\,000$ images).
  - [ ] Normaliser les intensités de pixels de $[0, 255]$ vers $[0.0, 1.0]$.
  - [ ] Réaliser le split `X_train, X_val, X_test, y_train, y_val, y_test` et sauvegarder au format compressé `.npz`.
- **Tâches Membre 3 (AI)** :
  - [ ] Implémenter la structure de la classe `SketchCNN` dans `model_builder.py`.
  - [ ] Vérifier le `model.summary()` : dimensionnement des tenseurs, nombre de paramètres entraînables (~500k paramètres).
- **Tâches Membre 4 (UI)** :
  - [ ] Développer le composant de dessin interactif : capture des coordonnées souris, tracé continu à trait épais noir sur fond blanc (ou fond transparent).
  - [ ] Ajouter les boutons d'action : `Effacer`, `Prédire`.
  - [ ] Implémenter la méthode d'export du dessin actuel en objet `QImage` / `PIL.Image` ou Base64.
- **Tâches Membre 5 (QA & Bonus)** :
  - [ ] Rédiger les tests unitaires de `prepare_data.py` (vérification de la forme matricielle `(N, 28, 28, 1)` et de l'intervalle $[0, 1]$).
  - [ ] Prototyper les premiers filtres artistiques OpenCV sur des croquis statiques.
- **Critère de validation J2** : Le fichier `data/processed/dataset_sketch.npz` est généré et exploitable. Le canvas permet de dessiner et de vider l'écran.

---

### 🔹 Jour 3 (Mercredi) : Entraînement de l'IA & Moteur d'Inférence Backend

- **Objectif** : Obtenir un modèle CNN performant et brancher la chaîne de prédiction.
- **Tâches Membre 1 (Toi - Lead)** :
  - [ ] Implémenter la classe `SketchClassifier` dans `inference_engine.py` :
    - Méthode `load_model(model_path, classes_path)`.
    - Méthode `preprocess(image_input) -> np.ndarray`.
    - Méthode `predict(image_input) -> List[Tuple[str, float]]`.
  - [ ] Mettre en place le tri décroissant des scores de probabilité (Softmax) et le calcul du taux de certitude.
  - [ ] Gérer les cas limites : canvas entièrement vide (retourner statut "Aucun dessin détecté").
- **Tâches Membre 2 (Data)** :
  - [ ] Développer un script de visualisation `visualize_samples.py` affichant une grille de 25 croquis aléatoires par classe pour contrôle qualité visuel.
  - [ ] Documenter le format des données et la distribution des classes.
- **Tâches Membre 3 (AI)** :
  - [ ] Lancer l'entraînement complet sur 15 à 25 époques avec batch size de 128.
  - [ ] Appliquer des callbacks : `EarlyStopping(patience=3)` et `ModelCheckpoint(save_best_only=True)`.
  - [ ] Évaluer la performance sur le jeu de test (précision globale, Precision, Recall, F1-Score).
  - [ ] Sauvegarder le modèle final `models/sketch_cnn_model.keras` et la liste ordonnée des classes `models/classes.json`.
- **Tâches Membre 4 (UI)** :
  - [ ] Concevoir le panneau d'affichage des résultats :
    - Nom de la classe prédite principale en grand format avec emoji d'illustration.
    - Top 3 des catégories avec barres de progression horizontales colorées.
- **Tâches Membre 5 (QA & Bonus)** :
  - [ ] Implémenter l'ajustement dynamique de l'épaisseur du pinceau (curseur slider de 2px à 20px).
  - [ ] Intégrer un bouton de sélection de couleur de trait.
- **Critère de validation J3** : Modèle exporté avec une précision > 90% sur le jeu de test. Le moteur d'inférence charge le modèle et classe un croquis simulé en moins de 50 ms.

---

### 🔹 Jour 4 (Jeudi) : Intégration Complète du Système

- **Objectif** : Réunir l'interface graphique et l'IA en une application fluide de bout en bout.
- **Tâches Membre 1 (Toi - Lead)** :
  - [ ] Intégrer le moteur d'inférence au sein de l'application principale `main.py`.
  - [ ] Connecter le déclencheur de prédiction : soit lors du clic sur le bouton "Reconnaître", soit en mode temps réel (inférence différée 300ms après la fin du trait `MouseRelease`).
  - [ ] Résoudre les conflits de merge Git et valider la stabilité de la branche `develop`.
- **Tâches Membre 2 (Data)** :
  - [ ] Participer aux tests de robustesse face aux variations d'échelle et aux traits fins/épais.
  - [ ] Générer un jeu de données de test indépendant pour la validation croisée.
- **Tâches Membre 3 (AI)** :
  - [ ] Analyser les matrices de confusion pour identifier les confusions fréquentes (ex. : `apple` vs `sun` ou `car` vs `house`).
  - [ ] Ajuster éventuellement l'augmentation de données (légère rotation, translation) si des confusions persistent.
- **Tâches Membre 4 (UI)** :
  - [ ] Lier les signaux de l'interface aux fonctions de prédiction.
  - [ ] Ajouter des indicateurs visuels de chargement (spinner / statut textuel "Analyse en cours...").
  - [ ] Soigner l'ergonomie : raccourcis clavier (`Ctrl+Z` pour effacer le dernier trait, `C` pour effacer tout, `Espace` pour prédire).
- **Tâches Membre 5 (QA & Bonus)** :
  - [ ] Développer la fonction d'exportation du dessin : sauvegarde sous `exports/croquis_YYYYMMDD_HHMMSS.png`.
  - [ ] Écrire la suite de tests d'intégration simulant le cycle complet : Dessin -> Inférence -> Affichage.
- **Critère de validation J4** : L'utilisateur lance `python main.py`, trace un dessin sur le canvas, clique sur "Reconnaître" et obtient immédiatement le bon label avec sa jauge de probabilité.

---

### 🔹 Jour 5 (Vendredi) : Fonctionnalités Bonus, Filtres OpenCV & Durcissement

- **Objectif** : Valoriser le projet avec des innovations créatives et éliminer tous les bugs identifiés.
- **Tâches Membre 1 (Toi - Lead)** :
  - [ ] Revue globale de la qualité du code (respect de la PEP 8, typage avec `typing`, docstrings complètes Google Style).
  - [ ] Optimisation de la latence de traitement image / OpenCV.
- **Tâches Membre 2 (Data)** :
  - [ ] Rédiger la documentation du pipeline de données dans le rapport technique.
- **Tâches Membre 3 (AI)** :
  - [ ] Exporter les graphiques officiels d'entraînement (Loss train/val, Accuracy train/val) en haute résolution pour le rapport.
- **Tâches Membre 4 (UI)** :
  - [ ] Intégrer les contrôles de fonctionnalités bonus dans l'interface (menu déroulant des filtres artistiques OpenCV).
  - [ ] Améliorer le design visuel : mode sombre (Dark Theme), polices lisibles, animations légères.
- **Tâches Membre 5 (QA & Bonus)** :
  - [ ] Finaliser le module `bonus_filters.py` proposant :
    1. **Filtre Canny Sketch** : mise en valeur des arêtes et contours.
    2. **Filtre Cartoon / Inversion Néon** : fond sombre et tracé luminescent coloré.
    3. **Filtre Lissage Bilatéral** : effet aquarelle/dessin doux.
  - [ ] Exécuter la couverture de tests `pytest --cov`.
- **Critère de validation J5** : L'application dispose des bonus fonctionnels, sans crash lors d'actions utilisateur imprévues.

---

### 🔹 Jour 6 & 7 (Samedi & Dimanche) : Documentation, Rapport & Répétition Démo

- **Objectif** : Produire une documentation irréprochable et préparer la soutenance.
- **Tâches collectives coordonnées par Membre 1 (Lead)** :
  - [ ] **Rapport de Synthèse (3 à 5 pages)** :
    - Section 1 : Contexte, problématique et objectifs du projet AI Sketch.
    - Section 2 : Architecture logicielle et flux de données modulaire.
    - Section 3 : Pipeline de Machine Learning (choix du CNN, régularisation, résultats chiffrés).
    - Section 4 : Interface utilisateur, ergonomie et fonctionnalités bonus OpenCV.
    - Section 5 : Conclusion, limites et perspectives d'évolution (reconnaissance de séquence RNN/LSTM, mode multijoueur Pictionary).
  - [ ] **README.md finalisé** :
    - Capture d'écran représentative de l'application en action.
    - Guide pas à pas d'installation (Windows / Linux / macOS).
    - Tableau des membres et attributions.
  - [ ] **Enregistrement vidéo de démonstration (2 à 3 minutes)** :
    - Scénario : Lancement de l'app -> Dessin d'un chat -> Reconnaissance instantanée -> Dessin d'une voiture -> Application d'un filtre néon -> Exportation du croquis.
  - [ ] Création de l'archive finale ZIP conforme aux directives académiques.

---

### 🔹 Jour 8 (Mardi 20 Octobre) : Livrable Final & Démonstration

- **Point de contrôle final** :
  - [ ] Vérification de la compatibilité de l'archive ZIP sur machine vierge.
  - [ ] Vérification du fonctionnement du script d'exécution rapide `run.bat` / `python main.py`.
  - [ ] Dépôt final du code et du rapport avant l'heure limite.

---

## 🎯 3. Critères d'Évaluation & Facteurs Clés de Succès

1. **Exactitude du Modèle (AI Performance)** : Précision supérieure à 90% sur le jeu de test des catégories retenues.
2. **Latence d'Inférence (UX Responsiveness)** : Moins de 100 ms entre la demande et l'affichage du score.
3. **Robustesse du Centrage d'Image** : Le système doit reconnaître un croquis même s'il est dessiné petit ou dans un coin du canvas (grâce au recadrage par boîte englobante).
4. **Qualité du Code & Modularité** : Séparation stricte entre UI, Inférence, Entraînement et Données. Aucun couplage fort.
5. **Expérience Utilisateur & Bonus** : Interface intuitive, présence de retours visuels clairs et fonctionnalités artistiques avec OpenCV.
