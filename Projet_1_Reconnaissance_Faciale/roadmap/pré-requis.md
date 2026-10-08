Avant de coder un système de reconnaissance faciale, il est essentiel de maîtriser les préliminaires. Voici une synthèse des points à connaître absolument, organisée pour que nous puissions bien structurer la réflexion.

---

## 🧠 Clarifier les concepts : détection vs identification vs vérification

La CNIL (Commission Nationale de l'Informatique et des Libertés) souligne une distinction fondamentale : **la détection de visage** (repérer la présence d'un visage) ne doit pas être confondue avec **la reconnaissance faciale**, qui vise à identifier une personne.

Dans ce projet, nous utiliserons les deux :
- **Détection** : pour localiser les visages dans l'image (bounding box).
- **Identification** (aussi appelée "reconnaissance 1:N") : pour trouver à qui appartient le visage parmi les étudiants enregistrés.

Il existe aussi la **vérification** (1:1), qui confirme si une personne est bien celle qu'elle prétend être, mais ce n'est pas ce qui est demandé ici.

**À retenir** : Ce système fera de l'identification (1:N) avec rejet possible ("Unknown") si le score de similarité est trop faible.

---

## ⚖️ Cadre éthique et légal : la base incontournable

La reconnaissance faciale traite des **données biométriques**, considérées comme des données sensibles au sens du RGPD (article 4-14). Sans cadre juridique approprié, on s'expose à des risques légaux et éthiques majeurs.

### Consentement explicite
Le RGPD exige un consentement **"libre, spécifique, éclairé et univoque"**. Pour ce projet pédagogique, chaque étudiant photographié doit donner son accord écrit après avoir été informé de l'usage strictement scolaire des données.

### Principes de responsabilité (Responsible AI)
Un guide de Microsoft Azure sur les considérations de détection et reconnaissance faciales recommande plusieurs bonnes pratiques :
- **Accès limité** : n'active la reconnaissance que pour des cas d'usage justifiés et autorisés.
- **Minimisation des données** : ne collecte que ce qui est strictement nécessaire.
- **Transparence** : informe clairement les utilisateurs sur la collecte et le traitement.
- **Sécurité** : chiffre les données au repos et en transit, contrôle les accès, journalise les opérations.

### Fairness et biais
Évaluez les performances du système sur différents sous-groupes (âge, genre, carnation) pour détecter d'éventuels biais. Utilisez un dataset diversifié pour l'entraînement et les tests.

**Pour ce projet** : Rédigez une note d'information et une autorisation d'utilisation d'image signée par chaque étudiant. Précise la durée de conservation (ex. : jusqu'à la fin de l'année scolaire) et l'usage exclusivement pédagogique.

---

## 🏗️ Architecture d'un système de reconnaissance faciale

Un système complet suit un **pipeline séquentiel** en quatre étapes principales :

1. **Détection de visage** : Localiser le visage dans l'image (bounding box). Des modèles comme YOLO ou les cascades de Haar sont couramment utilisés.
2. **Analyse des caractéristiques** : Cartographier les points nodaux (distance entre les yeux, largeur du nez, contour de la mâchoire) pour extraire des repères stables malgré les variations.
3. **Encodage (embedding)** : Convertir la géométrie analysée en un **vecteur numérique** (empreinte faciale). Chaque visage devient un point dans un espace vectoriel de haute dimension (souvent 128, 256 ou 512 dimensions).
4. **Mappage (matching)** : Comparer l'embedding du visage détecté à ceux de la base de données. Si le score de similarité dépasse un **seuil de confiance** prédéfini, l'identité est validée ; sinon, "Unknown".

**Point clé** : L'embedding est au cœur du système. Sa qualité détermine directement la précision de l'identification.

---

## 📚 Bibliothèques Python : comment choisir ?

Une étude comparative de 2026 a analysé trois bibliothèques majeures : `face_recognition`, `OpenCV` et `DeepFace`.

| Bibliothèque | Principe | Avantages | Inconvénients |
|:--|:--|:--|:--|
| **face_recognition** | Basée sur dlib (HOG + SVM + ResNet) | Simple, idéale pour débuter, tourne sur CPU | Précision limitée sur profils, pas de détection de vivacité |
| **OpenCV** (méthodes classiques : Eigenfaces, Fisherfaces, LBPH) | Apprentissage statistique | Rapide, léger | Faible précision en conditions réelles |
| **DeepFace** | Ensemble de modèles deep learning (VGG-Face, ArcFace, Facenet) | Haute précision (ArcFace ≈ 99,65 % sur MegaFace) | Nécessite plus de puissance de calcul (120 ms/frame) |

### Recommandations pour ce projet (5 étudiants, ~100-200 images)
Pour un **petit dataset pédagogique**, `face_recognition` est un excellent point de départ : elle est facile à prendre en main et suffisamment précise pour 5 personnes. Si l'évolution en compétence sur les embeddings est à considérer, `DeepFace` avec le modèle **Facenet** ou **ArcFace** offre un bon compromis.

Pour un système de taille moyenne (10 000 – 50 000 images), une étude de 2026 montre que **SFace** (un CNN compact produisant des embeddings 128-D) offre le meilleur équilibre : >95 % de précision sur LFW, 37 Mo de modèle, tourne efficacement sur CPU.

**À éviter** : Les méthodes classiques d'OpenCV (Eigenfaces, LBPH) pour un système réel ; elles sont obsolètes face aux approches par embedding.

---

## 🖼️ Préparation des données : le nerf de la guerre

Notre dataset doit être **structuré, propre et représentatif**. Voici les étapes clés :

### 1. Collecte
- **5 étudiants** × **20 à 40 images** chacun.
- **Variabilité contrôlée** : plusieurs orientations (face, profil, trois-quarts), expressions (neutre, sourire), distances (proche, éloignée), conditions d'éclairage (jour, nuit, intérieur, extérieur).
- **Éviter** : les images floues, trop sombres, ou avec des occultations majeures (masque, lunettes de soleil).

### 2. Prétraitement
- **Détection et alignement** : recadrer le visage autour des yeux et du nez pour normaliser la pose. Une étude sur un système embarqué pour le transport scolaire aligne les yeux horizontalement puis redimensionne à 128×128 pixels.
- **Redimensionnement** : taille fixe (ex. : 112×112 pour FaceNet, 128×128 pour d'autres modèles).
- **Normalisation des couleurs** : conversion en niveaux de gris ou normalisation des canaux RGB.
- **Augmentation de données** (optionnel) : rotations légères, variations de luminosité pour enrichir le dataset.

### 3. Organisation du dataset
```
dataset/
├── etudiant_1/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── etudiant_2/
│   └── ...
└── ...
```

**Pour le test d'absence** : Gardez une personne entièrement hors de la base d'entraînement. Elle servira à vérifier que le système retourne bien "Unknown" avec un score de confiance faible.

---

## 📊 Évaluation : quelles métriques utiliser ?

Pour évaluer ce système, ne nous limitons pas à l'**accuracy**. Utilisons un ensemble de métriques complémentaires :

| Métrique | Définition | Utilité |
|:--|:--|:--|
| **Précision** | TP / (TP + FP) | Proportion de bonnes identifications parmi celles prédites |
| **Rappel** | TP / (TP + FN) | Capacité à retrouver toutes les vraies identités |
| **F1-score** | Moyenne harmonique de précision et rappel | Équilibre entre les deux |
| **AUC (ROC)** | Aire sous la courbe ROC | Capacité à distinguer les classes à tous les seuils |

**Interprétation** : Un F1-score élevé signifie que le système identifie correctement les étudiants sans trop de faux positifs (identifier quelqu'un à tort) ni de faux négatifs (ne pas reconnaître un étudiant présent).

**Test crucial** : Mesure le taux de **faux positifs** sur la personne absente. Si le système l'identifie à tort comme un étudiant connu, le seuil de confiance est trop bas.

---

## 🛠️ Stack technique recommandée pour démarrer

Voici une stack minimale et cohérente pour ce projet :

| Composant | Choix recommandé | Justification |
|:--|:--|:--|
| **Langage** | Python 3.10+ | Standard pour la vision par ordinateur |
| **Détection** | `face_recognition` ou `OpenCV` (DNN) | Simple, efficace pour un petit projet |
| **Embedding** | `face_recognition` (128-D) ou `DeepFace` (Facenet/ArcFace) | Bon compromis précision/ressources |
| **Similarité** | Distance euclidienne ou similarité cosinus | Standard pour comparer des embeddings |
| **Interface** | `OpenCV` (webcam) + `Tkinter` ou `Streamlit` | Léger et rapide à prototyper |
| **Notebook** | Jupyter | Pour l'entraînement et l'expérimentation |
| **Gestion des dépendances** | `requirements.txt` | Livrable demandé |

**Note** : La bibliothèque `face_recognition` encapsule dlib et fournit une API simple : `face_locations()` pour la détection, `face_encodings()` pour l'embedding, `compare_faces()` pour le matching.

---

## 📝 Récapitulatif : ce qu'il faut retenir avant de coder

1. **Conceptuel** : Détection ≠ identification. Ce système fait de l'identification 1:N avec rejet.
2. **Légal** : Consentement explicite, minimisation des données, transparence, sécurité.
3. **Architecture** : Pipeline en 4 étapes (détection → analyse → embedding → matching).
4. **Bibliothèques** : `face_recognition` pour débuter ; `DeepFace` pour monter en compétence.
5. **Dataset** : 5 étudiants × 20-40 images, variabilité contrôlée, alignement et redimensionnement.
6. **Évaluation** : Précision, rappel, F1-score, AUC ; test avec une personne absente.
7. **Stack** : Python, OpenCV, face_recognition/DeepFace, Jupyter.

Une fois ces préliminaires maîtrisés, nous pourrons passer à la conception détaillée puis au code.
