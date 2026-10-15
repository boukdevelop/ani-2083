# 🧠 Pré-requis & Fondations Théoriques — AI Sketch Recognizer

> Ce document synthétise les principes scientifiques, théoriques et méthodologiques indispensables pour comprendre et mener à bien le projet **AI Sketch Recognizer**.
> Il aborde la nature du jeu de données Google Quick, Draw!, le fonctionnement interne des Réseaux de Neurones Convolutifs (CNN) appliqués aux croquis vectoriels/matriciels, et les étapes fondamentales du traitement d'image.

---

## 1. Comprendre le Dataset Google Quick, Draw!

### 1.1 Qu'est-ce que Quick, Draw! ?

En 2016, Google a lancé le jeu en ligne expérimental _Quick, Draw!_ où des millions de joueurs devaient dessiner un concept (ex. : chat, avion, horloge) en 20 secondes tandis qu'une intelligence artificielle tentait de deviner l'objet en temps réel.
Le résultat est la **plus grande base de données de croquis au monde**, contenant plus de **50 millions de dessins** répartis sur 345 catégories.

### 1.2 Formats de données disponibles

Google propose ce jeu de données sous plusieurs représentations :

1. **Raw Vector data (`.ndjson`)** : Coordonnées temporelles $(x, y, t)$ précises de chaque trait de pinceau. Idéal pour des modèles récurrents (RNN / LSTM).
2. **Simplified Vector data** : Versions allégées des vecteurs lissés avec l'algorithme de Ramer-Douglas-Peucker.
3. **Numpy Bitmap (`.npy`) — FORMAT CHOISI** :
   - Chaque croquis a été rendu sur une grille bitmap de **$28 \times 28$ pixels en niveaux de gris**.
   - Chaque image est aplatie sous forme d'un vecteur 1D de **784 octets** (valeurs entières comprises entre $0$ et $255$).
   - Ce format est directement analogue au célèbre jeu de données **MNIST** ou **Fashion-MNIST**, ce qui le rend idéal pour l'apprentissage par Réseau de Neurones Convolutif (CNN).

### 1.3 Convention de couleur : Le Piège Incontournable

- **Dans le fichier `.npy` Quick, Draw!** :
  - Fond de l'image = **Noir** (valeur de pixel = `0`).
  - Tracé du dessin = **Blanc** (valeur de pixel allant jusqu'à `255`).
- **Sur l'interface utilisateur humaine** :
  - Fond du canvas = **Blanc** (`255`).
  - Trait de l'utilisateur = **Noir** (`0`) ou foncé.
- **Règle fondamentale** : Lors de la phase d'inférence, une étape d'**inversion de polarité** est obligatoire :
  $$\text{pixel}_{\text{inversé}} = 255 - \text{pixel}_{\text{canvas}}$$
  Sans cette opération mathématique élémentaire, le modèle recevra un fond blanc alors qu'il a appris sur un fond noir, conduisant à un taux d'erreur de 100 %.

---

## 2. Réseaux de Neurones Convolutifs (CNN) Appliqués aux Croquis

### 2.1 Pourquoi un CNN plutôt qu'un perceptron multicouche classique (MLP) ?

Un perceptron standard (couches `Dense` uniquement) traite chaque pixel comme une variable indépendante, perdant totalement la notion spatiale et de voisinage des pixels.
Le croquis se caractérise par des **lignes, courbures, boucles et intersections**. Un CNN excelle dans l'extraction hiérarchique de ces caractéristiques géométriques grâce à trois mécanismes clés :

1. **Invariance par translation** : Une oreille de chat a la même signature convolutive qu'elle soit dessinée au centre ou décalée vers la droite.
2. **Partage des poids (Weight Sharing)** : Le même noyau de convolution glisse sur toute l'image, réduisant drastiquement le nombre de paramètres par rapport à un réseau dense.
3. **Hiérarchie des motifs** :
   - Couches 1 & 2 : Détection des contours élémentaires (traits horizontaux, obliques, arcs de cercle).
   - Couches 3 & 4 : Détection de sous-structures (roues de voiture, yeux, branches d'arbre).
   - Couches denses finales : Association globale pour la décision de classification.

---

### 2.2 Anatomie des Couches du Modèle

```
Input (28x28x1)
       │
       ▼
[Conv2D (32 filtres 3x3) + ReLU]  ──► Extraction de caractéristiques locales
       │
[BatchNormalization]              ──► Stabilisation et accélération de l'apprentissage
       │
[MaxPooling2D (2x2)]              ──► Réduction spatiale de dimension (28x28 -> 14x14)
       │
[Dropout (0.25)]                  ──► Régularisation anti-surapprentissage (surfitting)
       │
       ▼
[Conv2D (64 filtres 3x3) + ReLU]  ──► Extraction de motifs plus complexes
       │
[BatchNormalization]
       │
[MaxPooling2D (2x2)]              ──► Réduction spatiale (14x14 -> 7x7)
       │
[Dropout (0.25)]
       │
       ▼
[Flatten]                         ──► Aplatissement du tenseur 3D en vecteur 1D
       │
[Dense (256 neurones) + ReLU]     ──► Combinaison non-linéaire des caractéristiques
       │
[Dropout (0.4)]
       │
       ▼
[Dense (K neurones) + Softmax]    ──► Distribution de probabilités sur les K catégories
```

#### Mathématiques de la couche Softmax

Pour un vecteur de sorties brutes $z = [z_1, z_2, \dots, z_K]$ issues de la dernière couche linéaire :
$$P(\text{classe } i \mid \mathbf{x}) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
Cette fonction garantit que chaque score est compris dans l'intervalle $]0, 1[$ et que la somme de toutes les probabilités vaut exactement $1$.

---

## 3. Prétraitement & Vision par Ordinateur avec OpenCV

L'exactitude d'un système de reconnaissance dépend à **80% de la qualité du prétraitement de l'image**. Un utilisateur réel ne dessine jamais exactement au format $28 \times 28$ au centre de l'écran.

### 3.1 Pipeline de Redimensionnement Centré (Bounding Box Cropping)

```
[Dessin utilisateur sur Canvas (ex: 400x400)]
                     │
                     ▼
       1. Binarisation & Inversion
  (Détection des pixels actifs > 0)
                     │
                     ▼
   2. Calcul de la Boîte Englobante
         (cv2.boundingRect)
       [X_min, Y_min, W, H]
                     │
                     ▼
    3. Rognage carré avec Padding
  (Préservation stricte du ratio d'aspect)
                     │
                     ▼
     4. Interpolation d'aire (cv2.INTER_AREA)
         Redimensionnement en 28x28
                     │
                     ▼
           5. Normalisation [0, 1]
```

### 3.2 Pourquoi `cv2.INTER_AREA` ?

Lorsqu'on effectue une forte réduction de résolution (par exemple passer d'une zone de $300 \times 300$ pixels à $28 \times 28$ pixels) :

- L'interpolation bilinéaire ou plus proche voisin produit un phénomène d'**aliasing (crénelage)** et peut faire disparaître des traits fins d'un seul pixel.
- L'interpolation par échantillonnage d'aire (`INTER_AREA`) calcule la moyenne pondérée de tous les pixels du bloc d'origine, préservant ainsi la continuité et l'intensité des traits.

---

## 4. Métriques d'Évaluation du Modèle

Pour évaluer la fiabilité du système au-delà d'une simple précision moyenne, nous exploitons quatre métriques formelles :

1. **Exactitude Globale (Accuracy)** :
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
   Donne une vision d'ensemble du taux de réussite sur un jeu de données équilibré.

2. **Précision par classe (Precision)** :
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   Mesure la capacité du modèle à ne pas qualifier d'objet $A$ un objet $B$ (ex. : éviter de crier "Chat !" quand c'est une "Voiture").

3. **Rappel par classe (Recall)** :
   $$\text{Recall} = \frac{TP}{TP + FN}$$
   Mesure la proportion de vrais objets retrouvés par le modèle.

4. **Matrice de Confusion** :
   Tableau à deux dimensions comparant les étiquettes réelles (en lignes) et les prédictions (en colonnes). Elle permet d'identifier visuellement les confusions géométriques typiques (ex. : forme ronde d'une pomme vs forme ronde du soleil).

---

## 5. Bonnes Pratiques d'Ingénierie Logicielle

1. **Modularité Découplée** : L'interface utilisateur ne doit rien savoir de l'implémentation interne de TensorFlow. Elle appelle simplement `classifier.predict(image)`.
2. **Reproductibilité** : Tous les splits de données et initialisations de poids doivent fixer une graine aléatoire (`seed = 42`).
3. **Gestion des Cas Limites (Edge Cases)** :
   - Écran vide $\rightarrow$ Ne pas solliciter le CNN, lever une exception ou renvoyer un statut vide.
   - Tracé avec un seul point $\rightarrow$ Avertissement visuel "Dessin trop succinct".
4. **Optimisation des Performances** :
   - Le modèle est chargé une seule fois au démarrage de l'application et maintenu en mémoire vive. Ne jamais recharger le modèle à chaque clic sur "Prédire".
