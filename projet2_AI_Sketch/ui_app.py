'''ce code sert a creer une fenetre de dessin a la souris.
le dessin obtenu sera Convertit en une matrice/tableau NumPy (au format RVB)
afin qu'il puisse être exploité et analysé par un modèle d'intelligence artificielle
la methode qui retourne cette matrice est la fonction get_image_as_numpy() de la classe DrawingCanvas
Sont retour se trouve a la ligne 191'''

import sys  # Importation du module natif 'sys' (appel du module système pour gérer les arguments et la sortie du programme)
import numpy as np  # Importation de la bibliothèque 'numpy' sous l'alias 'np' (appel du module pour le calcul matriciel et la manipulation de tableaux de pixels)

# Importation de classes spécifiques (composants graphiques) depuis le module PyQt6.QtWidgets
from PyQt6.QtWidgets import (
    QApplication,  # Classe pour gérer l'initialisation et la boucle principale de l'application
    QMainWindow,  # Classe pour créer la fenêtre principale de l'application
    QWidget,  # Classe de base pour tous les objets visuels et conteneurs
    QVBoxLayout,  # Gestionnaire de disposition vertical (empile les éléments de haut en bas)
    QHBoxLayout,  # Gestionnaire de disposition horizontal (aligne les éléments côte à côte)
    QPushButton,  # Classe pour créer des boutons cliquables
    QLabel,  # Classe pour afficher du texte statique ou des informations
    QProgressBar,  # Classe pour créer des barres de progression horizontales (jauges de score)
    QGroupBox,  # Classe pour créer un cadre visuel étiqueté regroupant des éléments
)

# Importation de classes utilitaires et constantes depuis PyQt6.QtCore
from PyQt6.QtCore import (
    Qt,  # Objet global contenant des constantes d'alignement, de style et d'énumérations
    QPoint,  # Classe représentant un point géométrique 2D muni de coordonnées (x, y)
)

# Importation des classes graphiques de rendu et de gestion d'image depuis PyQt6.QtGui
from PyQt6.QtGui import (
    QPainter,  # Classe de dessin (pinceau/crayon virtuel pour dessiner sur une image ou un widget)
    QPen,  # Classe définissant l'épaisseur, la couleur et le style des traits tracés
    QImage,  # Classe représentant une image matricielle stockée en mémoire vive
    QColor,  # Classe gérant les couleurs au format RVB (Rouge, Vert, Bleu)
)


'''classe de Gestion de la zone de dessin. la classe 'DrawingCanvas' qui hérite de la classe parente 'QWidget' '''
class DrawingCanvas(
    QWidget
):  
    '''Le constructeur de la classe. Il initialise le widget de dessin,
      définit sa taille fixe, configure les paramètres initiaux du pinceau (épaisseur, état inactif, point de départ)
        et crée une image tampon en mémoire (QImage) entièrement remplie de blanc pour servir de support aux tracés.'''
    
    def __init__(
        self, parent=None
    ):  
        super().__init__(
            parent
        )  # Appel de la fonction constructeur de la classe parente 'QWidget' via 'super()' pour l'initialiser
        self.setFixedSize(
            600, 600
        )  # Appel de la méthode 'setFixedSize' sur l'instance courante ('self') pour fixer la taille du canvas à exactement 300x300 pixels
        self.pen_width = (
            12  # Création et assignation de la variable d'instance 'self.pen_width' avec la valeur entière 12 (épaisseur du pinceau)
        )
        self.drawing = (
            False  # Création de la variable d'état booléenne 'self.drawing' initialisée à False (indique si l'utilisateur est en train de tracer)
        )
        self.last_point = (
            QPoint()  # Instanciation d'un objet 'QPoint' vide assigné à 'self.last_point' pour stocker la dernière position connue de la souris
        )

        # Instanciation d'un objet 'QImage' (tampon invisible en mémoire) aux dimensions du widget et au format RVB 32 bits, assigné à 'self.image'
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(
            Qt.GlobalColor.white
        )  # Appel de la méthode 'fill' sur l'objet 'self.image' pour colorer tout son fond en blanc pur ('Qt.GlobalColor.white')

    '''Effacer complètement le contenu du dessin en repeignant l'image en mémoire en blanc,
      puis déclencher un rafraîchissement visuel de l'interface (update()).'''
    def clear(
        self,
    ) -> (
        None
    ):  # Définition d'une méthode 'clear' (sans valeur de retour '-> None') pour effacer le dessin
        self.image.fill(
            Qt.GlobalColor.white
        )  # Appel de la méthode 'fill' sur 'self.image' pour réinitialiser tous les pixels en blanc
        self.update()  # Appel de la méthode 'update()' sur le widget pour déclencher un rafraîchissement visuel de l'interface à l'écran

    '''Intercepter le clic de la souris. 
    Lorsque l'utilisateur clique avec le bouton gauche, 
    la fonction active l'état de dessin (self.drawing = True) 
    et enregistre les coordonnées exactes du point de départ.'''
    def mousePressEvent(
        self, event
    ) -> (
        None
    ):  # Surcharge de la méthode native 'mousePressEvent' appelée automatiquement lorsqu'on clique sur le canvas
        if (
            event.button() == Qt.MouseButton.LeftButton
        ):  # Opération conditionnelle (if) : vérifie si le bouton cliqué (via l'appel de 'event.button()') est le bouton gauche ('Qt.MouseButton.LeftButton')
            self.drawing = (
                True  # Modification de l'état 'self.drawing' à True (l'utilisateur commence à dessiner)
            )
            self.last_point = (
                event.position().toPoint()
            )  # Appel des méthodes 'event.position().toPoint()' pour récupérer les coordonnées exactes du curseur et les stocker dans 'self.last_point'


    '''Gère le tracé en continu lorsque l'utilisateur déplace la souris tout en maintenant le bouton gauche enfoncé.
      Elle utilise un crayon (QPen) et un pinceau (QPainter) pour dessiner une ligne continue 
      sur l'image en mémoire entre la position précédente et la position actuelle, puis actualise l'affichage.'''
    def mouseMoveEvent(
        self, event
    ) -> (
        None
    ):  # Surcharge de la méthode native 'mouseMoveEvent' appelée en continu lorsque la souris bouge sur le canvas
        # Opération conditionnelle combinée (if) : vérifie si 'self.drawing' est True ET si le bouton gauche est maintenu enfoncé (opération binaire '&' sur 'event.buttons()')
        if self.drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            painter = QPainter(
                self.image
            )  # Instanciation d'un objet 'QPainter' nommé 'painter' lié directement à l'image en mémoire 'self.image'
            # Instanciation d'un objet 'QPen' nommé 'pen' (stylo) configuré avec une couleur noire, l'épaisseur, un trait continu, des bouts et jointures arrondis
            pen = QPen(
                QColor(0, 0, 0),
                self.pen_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(
                pen
            )  # Appel de la méthode 'setPen' sur 'painter' pour appliquer le stylo configuré
            painter.drawLine(
                self.last_point, event.position().toPoint()
            )  # Appel de la méthode 'drawLine' pour tracer une ligne sur l'image entre l'ancienne position ('self.last_point') et la position actuelle ('event.position().toPoint()')
            self.last_point = (
                event.position().toPoint()
            )  # Mise à jour de 'self.last_point' avec la position actuelle de la souris
            self.update()  # Appel de la méthode 'update()' pour rafraîchir l'affichage graphique du widget à l'écran


    '''Détecte le relâchement du bouton de la souris pour désactiver l'état de dessin (self.drawing = False).'''
    def mouseReleaseEvent(
        self, event
    ) -> (
        None
    ):  # Surcharge de la méthode native 'mouseReleaseEvent' appelée lorsqu'on relâche un bouton de la souris
        if (
            event.button() == Qt.MouseButton.LeftButton
        ):  # Opération conditionnelle : vérifie si c'est le bouton gauche qui a été relâché
            self.drawing = (
                False  # Modification de l'état 'self.drawing' à False (l'utilisateur arrête de dessiner)
            )

    '''Méthode de rendu graphique native appelée automatiquement pour afficher 
    et actualiser le contenu visuel du widget à l'écran en copiant l'image tampon de la mémoire.'''
    def paintEvent(
        self, event
    ) -> (
        None
    ):  # Surcharge de la méthode native 'paintEvent' appelée automatiquement pour dessiner le widget à l'écran
        canvas_painter = QPainter(
            self
        )  # Instanciation d'un 'QPainter' nommé 'canvas_painter' attaché au widget lui-même ('self')
        canvas_painter.drawImage(
            self.rect(), self.image, self.image.rect()
        )  # Appel de la méthode 'drawImage' pour copier l'image en mémoire 'self.image' sur la zone rectangulaire du widget ('self.rect()')


    '''Convertit le dessin réalisé par l'utilisateur en une matrice/tableau NumPy (au format RVB)
      afin qu'il puisse être exploité et analysé par un modèle d'intelligence artificielle.'''
    def get_image_as_numpy(
        self,
    ) -> (
        np.ndarray
    ):  # Définition de la méthode 'get_image_as_numpy' pour convertir le dessin en tableau NumPy (retourne un np.ndarray)[cite: 2]
        width = (
            self.image.width()
        )  # Appel de la méthode 'width()' pour récupérer la largeur de l'image et l'assigner à 'width'
        height = (
            self.image.height()
        )  # Appel de la méthode 'height()' pour récupérer la hauteur de l'image et l'assigner à 'height'

        ptr = (
            self.image.bits()
        )  # Appel de la méthode 'bits()' pour récupérer le pointeur mémoire brut des pixels de l'image, assigné à 'ptr'
        ptr.setsize(
            self.image.sizeInBytes()
        )  # Appel de la méthode 'setsize()' sur le pointeur pour s'ajuster à la taille totale en octets de l'image (via 'self.image.sizeInBytes()')
        arr = np.array(ptr).reshape(
            height, width, 4
        )  # Appel de la fonction 'np.array()' pour convertir le flux en tableau NumPy, puis appel de la méthode '.reshape()' pour structurer le tableau en matrice 3D de dimensions (hauteur, largeur, 4 canaux RGBA)

        rgb_arr = arr[
            :, :, :3
        ]  # Opération de découpage (slicing) sur le tableau NumPy 'arr' : on garde toutes les lignes et colonnes, mais on filtre pour ne garder que les 3 premiers canaux (:3, RVB) en ignorant le canal alpha, assigné à 'rgb_arr'
        return rgb_arr  # Instruction de retour (return) renvoyant la matrice NumPy finale

'''classe de gestion de la Fenêtre principale et interface utilisateur'''
class MainWindow(
    QMainWindow
):  # Définition de la classe principale 'MainWindow' héritant de 'QMainWindow'

    '''Le constructeur de la fenêtre principale.
      Il initialise la fenêtre, mémorise le modèle de classification (s'il est fourni),
        définit le titre ainsi que les dimensions globales, puis appelle la fonction de création de l'interface.'''
    def __init__(
        self, classifier=None
    ):  # Constructeur de la fenêtre principale, acceptant un paramètre optionnel 'classifier' (initialisé à None par défaut)
        super().__init__()  # Appel du constructeur de la classe parente 'QMainWindow' via 'super()'
        self.classifier = (
            classifier  # Assignation du paramètre reçu à la variable d'instance 'self.classifier'
        )
        self.setWindowTitle(
            "AI Sketch Recognizer - Membre 4 (UI)"
        )  # Appel de la méthode 'setWindowTitle' pour définir le titre affiché en haut de la fenêtre
        self.setFixedSize(
            700, 700
        )  # Appel de la méthode 'setFixedSize' pour fixer les dimensions globales de la fenêtre à 450x550 pixels

        self.init_ui()  # Appel de la méthode interne 'self.init_ui()' pour construire et agencer tous les composants graphiques


    '''Construit et assemble tous les éléments graphiques de l'application (le titre,
      le canvas de dessin, les boutons « Effacer » et « Reconnaître », 
      ainsi que les barres de progression et labels du Top 3 des résultats) en utilisant des gestionnaires de disposition.'''
    def init_ui(
        self,
    ):  # Définition de la méthode interne 'init_ui' pour configurer les widgets et la mise en page
        central_widget = (
            QWidget()
        )  # Instanciation d'un widget conteneur générique 'central_widget'
        self.setCentralWidget(
            central_widget
        )  # Appel de la méthode 'setCentralWidget' pour définir ce widget comme le cœur de la fenêtre principale
        main_layout = QVBoxLayout(
            central_widget
        )  # Instanciation d'un gestionnaire de disposition vertical 'QVBoxLayout' attaché à 'central_widget' (les éléments s'empilent de haut en bas)

        title_label = QLabel(
            "🎨 Dessinez un objet ci-dessous :"
        )  # Instanciation d'un composant texte 'QLabel' avec le libellé du titre
        title_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; margin-bottom: 5px;"
        )  # Appel de la méthode 'setStyleSheet' pour appliquer un style CSS au texte (taille, gras, marge)
        main_layout.addWidget(
            title_label, alignment=Qt.AlignmentFlag.AlignCenter
        )  # Appel de la méthode 'addWidget' sur 'main_layout' pour insérer le label en le centrant horizontalement ('Qt.AlignmentFlag.AlignCenter')

        self.canvas = (
            DrawingCanvas()
        )  # Instanciation de notre canvas de dessin personnalisé, assigné à la variable d'instance 'self.canvas'
        self.canvas.setStyleSheet(
            "border: 2px solid #ccc; background-color: white;"
        )  # Appel de la méthode 'setStyleSheet' pour ajouter une bordure grise et un fond blanc au canvas
        main_layout.addWidget(
            self.canvas, alignment=Qt.AlignmentFlag.AlignCenter
        )  # Appel de la méthode 'addWidget' pour insérer le canvas dans le layout vertical, centré

        btn_layout = (
            QHBoxLayout()
        )  # Instanciation d'un gestionnaire de disposition horizontal 'QHBoxLayout' nommé 'btn_layout' pour aligner les boutons côte à côte

        self.btn_clear = QPushButton(
            "Effacer"
        )  # Instanciation d'un bouton cliquable 'QPushButton' avec le texte "Effacer", assigné à 'self.btn_clear'
        self.btn_clear.setStyleSheet(
            "background-color: #ff4d4d; color: white; font-weight: bold; padding: 8px 15px; border-radius: 4px;"
        )  # Appel de la méthode 'setStyleSheet' pour styliser le bouton en rouge
        self.btn_clear.clicked.connect(
            self.canvas.clear
        )  # Appel de la méthode '.clicked.connect()' pour lier le signal de clic du bouton directement à l'exécution de la méthode 'self.canvas.clear'
        btn_layout.addWidget(
            self.btn_clear
        )  # Appel de la méthode 'addWidget' pour ajouter le bouton "Effacer" dans le layout horizontal 'btn_layout'

        self.btn_predict = QPushButton(
            "Reconnaître"
        )  # Instanciation d'un bouton cliquable 'QPushButton' avec le texte "Reconnaître", assigné à 'self.btn_predict'
        self.btn_predict.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 15px; border-radius: 4px;"
        )  # Appel de la méthode 'setStyleSheet' pour styliser le bouton en vert
        self.btn_predict.clicked.connect(
            self.on_predict_clicked
        )  # Appel de la méthode '.clicked.connect()' pour lier le clic du bouton à la méthode interne 'self.on_predict_clicked'
        btn_layout.addWidget(
            self.btn_predict
        )  # Appel de la méthode 'addWidget' pour ajouter le bouton "Reconnaître" dans 'btn_layout'

        main_layout.addLayout(
            btn_layout
        )  # Appel de la méthode 'addLayout' sur 'main_layout' pour insérer la rangée complète de boutons sous le canvas

        results_group = QGroupBox(
            "Résultats de la Prédiction (Top 3)"
        )  # Instanciation d'un groupe visuel 'QGroupBox' avec une bordure et un titre pour la section des résultats
        results_layout = (
            QVBoxLayout()
        )  # Instanciation d'un gestionnaire vertical 'results_layout' pour organiser les lignes de score à l'intérieur du groupe

        self.prediction_labels = (
            []
        )  # Création d'une liste vide 'self.prediction_labels' pour stocker les étiquettes textuelles des prédictions
        self.progress_bars = (
            []
        )  # Création d'une liste vide 'self.progress_bars' pour stocker les barres de progression graphiques

        # Boucle 'for' s'exécutant 3 fois (de i = 0 à 2) pour créer les éléments du Top 3 des prédictions
        for i in range(3):
            row_layout = (
                QHBoxLayout()
            )  # Instanciation d'un layout horizontal 'row_layout' pour chaque ligne (label + barre)
            lbl = QLabel(
                f"Top {i+1} : -"
            )  # Instanciation d'un label 'QLabel' affichant par défaut "Top 1 : -", etc.
            lbl.setMinimumWidth(
                110
            )  # Appel de la méthode 'setMinimumWidth' pour fixer une largeur minimale de 110 pixels au label
            bar = QProgressBar()  # Instanciation d'une barre de progression 'QProgressBar' nommée 'bar'
            bar.setRange(
                0, 100
            )  # Appel de la méthode 'setRange' pour définir les limites de la barre de 0 à 100
            bar.setValue(
                0
            )  # Appel de la méthode 'setValue' pour initialiser la valeur de la barre à 0%
            bar.setStyleSheet(
                "QProgressBar { border: 1px solid #bbb; border-radius: 3px; text-align: center; } QProgressBar::chunk { background-color: #2196F3; }"
            )  # Appel de 'setStyleSheet' pour appliquer un style CSS personnalisé à la barre (fond gris, remplissage bleu)

            row_layout.addWidget(
                lbl
            )  # Ajout du label dans le layout de la ligne horizontale
            row_layout.addWidget(
                bar
            )  # Ajout de la barre de progression dans le layout de la ligne horizontale
            results_layout.addLayout(
                row_layout
            )  # Ajout de la ligne complète dans le layout vertical des résultats

            self.prediction_labels.append(
                lbl
            )  # Appel de la méthode '.append()' pour stocker le label dans la liste 'self.prediction_labels'
            self.progress_bars.append(
                bar
            )  # Appel de la méthode '.append()' pour stocker la barre dans la liste 'self.progress_bars'

        results_group.setLayout(
            results_layout
        )  # Appel de la méthode 'setLayout' pour assigner le layout vertical 'results_layout' au groupe 'results_group'
        main_layout.addWidget(
            results_group
        )  # Appel de la méthode 'addWidget' pour insérer le groupe de résultats dans le layout principal de la fenêtre


    '''Fonction déclenchée lors d'un clic sur le bouton « Reconnaître ».
      Elle récupère l'image du dessin sous forme de tableau NumPy,
        gère un mode de test autonome si aucun classifieur n'est branché,
          ou transmet l'image au modèle d'IA pour obtenir les prédictions.'''
    def on_predict_clicked(
        self,
    ):  # Définition de la méthode 'on_predict_clicked' appelée lorsque l'utilisateur clique sur le bouton "Reconnaître"
        img_np = (
            self.canvas.get_image_as_numpy()
        )  # Appel de la méthode 'get_image_as_numpy()' sur l'objet 'self.canvas' pour récupérer le dessin sous forme de matrice NumPy, assignée à 'img_np'

        # Opération conditionnelle (if) : vérifie si aucun classifieur n'a été fourni (mode test autonome, self.classifier est None)
        if self.classifier is None:
            # Appel de la méthode d'affichage avec des valeurs fictives de test (chat à 92%, etc.)
            self.update_results_display(
                [("chat (test)", 0.92), ("maison", 0.05), ("arbre", 0.03)]
            )
            return  # Instruction de sortie anticipée de la fonction (return)

        try:  # Bloc de protection de code (try...except) pour intercepter d'éventuelles erreurs d'exécution
            # Appel de la fonction officielle de prédiction du classifieur (contrat d'interface du Membre 1) en passant l'image NumPy et top_k=3[cite: 2]
            predictions = self.classifier.predict(img_np, top_k=3)
            self.update_results_display(
                predictions
            )  # Appel de la méthode interne 'update_results_display' en lui passant les prédictions reçues
        except (
            Exception
        ) as e:  # Capture de toute exception technique survenue dans le bloc try
            print(
                f"Erreur d'inférence : {e}"
            )  # Affichage d'un message d'erreur formaté dans la console via la fonction 'print()'


    '''Met à jour dynamiquement l'interface graphique avec les résultats renvoyés par l'IA.
      Elle convertit les scores en pourcentages et actualise à la fois les étiquettes textuelles 
      (ex: Chat : 92%) et les barres de progression correspondantes pour le Top 3.'''
    def update_results_display(
        self, predictions
    ):  # Définition de la méthode 'update_results_display' pour actualiser l'interface avec les résultats de l'IA
        # Boucle 'for' combinée avec la fonction 'enumerate()' pour parcourir chaque élément de 'predictions' (récupérant l'indice 'i', le nom 'class_name' et le score décimal 'score')
        for i, (class_name, score) in enumerate(predictions):
            # Opération conditionnelle : vérifie si l'indice 'i' est strictement inférieur à la taille de la liste des labels affichés
            if i < len(self.prediction_labels):
                percentage = int(
                    score * 100
                )  # Conversion mathématique : multiplication du score décimal (ex: 0.92) par 100, puis conversion en entier (int) pour obtenir un pourcentage (92)
                self.prediction_labels[i].setText(
                    f"{class_name} : {percentage}%"
                )  # Appel de la méthode 'setText' sur le label d'indice 'i' pour modifier son texte affiché
                self.progress_bars[i].setValue(
                    percentage
                )  # Appel de la méthode 'setValue' sur la barre de progression d'indice 'i' pour actualiser sa jauge visuelle

'''zone de test : a supprimer avant d'appeler ce fichier ailleur'''
if (
    __name__ == "__main__"
):  # Opération conditionnelle globale : vérifie si le script est exécuté directement en tant que programme principal (et non importé ailleurs)
    app = QApplication(
        sys.argv
    )  # Instanciation de l'objet principal de l'application 'QApplication' en lui passant les arguments de ligne de commande 'sys.argv'
    window = MainWindow(
        classifier=None
    )  # Instanciation de la fenêtre principale 'MainWindow' en mode autonome (classifier=None), assignée à la variable 'window'
    window.show()  # Appel de la méthode 'show()' sur l'objet 'window' pour rendre la fenêtre visible à l'écran
    sys.exit(
        app.exec()
    )  # Appel de la fonction 'sys.exit()' pour fermer proprement le programme lorsque la boucle événementielle de l'application ('app.exec()') se termine