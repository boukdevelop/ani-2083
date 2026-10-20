"""
ce code sert a creer une fenetre de dessin a la souris.
le dessin obtenu sera Convertit en une matrice/tableau NumPy (au format RVB)
afin qu'il puisse être exploité et analysé par un modèle d'intelligence artificielle
la methode qui retourne cette matrice est la fonction get_image_as_numpy() de la classe DrawingCanvas

"""

import sys  # Importation du module natif 'sys' (appel du module système pour gérer les arguments et la sortie du programme)
import os
import numpy as np  # Importation de la bibliothèque 'numpy' sous l'alias 'np' (appel du module pour le calcul matriciel et la manipulation de tableaux de pixels)
import cv2

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
    QSlider,
    QSpinBox,
    QColorDialog,
    QFrame,
    QSplitter,
)

# Importation de classes utilitaires et constantes depuis PyQt6.QtCore
from PyQt6.QtCore import Qt, QPoint, QSize

# Importation des classes graphiques de rendu et de gestion d'image depuis PyQt6.QtGui
from PyQt6.QtGui import QPainter, QPen, QImage, QColor, QFont, QIcon

from bonus_filters import BonusFilters


class DrawingCanvas(QWidget):
    """
    Widget personnalisable pour le dessin interactif à la souris.
    Offre le choix de la couleur, de l'épaisseur du pinceau et gère le redimensionnement.
    """
    # constructeur de la classe DrawingCanvas.
    def __init__(self, parent=None):
        super().__init__(parent) # Appelle le constructeur de la classe parente (QWidget de PyQt6) pour initialiser correctement le composant graphique de base.
        self.setMinimumSize(400, 400) #Fixe la taille minimale du canvas à 400x400 pixels, évitant qu'il ne devienne trop petit lors du redimensionnement de la fenêtre.

        self.pen_width = 12 # Initialise l'épaisseur par défaut du pinceau à 12 pixels.
        self.pen_color = QColor(0, 0, 0) # Définit la couleur initiale du pinceau en noir au format RGB (0, 0, 0).
        self.drawing = False # Crée un indicateur (un drapeau booléen) qui vaut False par défaut. Il passera à True uniquement lorsque l'utilisateur cliquera et maintiendra le bouton de la souris enfoncé pour dessiner.
        self.last_point = QPoint() # Initialise un objet point vide (QPoint) qui servira à mémoriser la dernière position de la souris afin de relier les points entre eux par un trait continu.

        # Image en mémoire tampon
        self.image = QImage(QSize(400, 400), QImage.Format.Format_RGB32) # Crée une image en mémoire tampon de 400x400 pixels avec un encodage des couleurs sur 32 bits (RGB32). C'est crucial : le dessin ne se fait pas directement à l'écran, mais sur cette image cachée en mémoire pour éviter qu'il ne s'efface lors des rafraîchissements de la fenêtre.
        self.image.fill(Qt.GlobalColor.white) # Remplit cette image en mémoire avec un fond entièrement blanc pour démarrer sur une "feuille blanche"

    """
    fonction permet de modifier dynamiquement l'épaisseur (la taille) du pinceau utilisé pour dessiner.

    """
    def set_pen_width(self, width: int): # Définit la méthode qui prend en paramètre un entier (width) représentant la nouvelle taille du pinceau en pixels.
        self.pen_width = width # Met à jour la variable interne self.pen_width de l'objet avec la nouvelle valeur reçue. Cette valeur sera ensuite utilisée lors de la création du stylo (QPen) lorsque l'utilisateur tracera un trait sur le canvas.

    """
    Cette fonction permet de modifier dynamiquement la couleur du pinceau utilisé pour dessiner sur le canvas.

    """
    def set_pen_color(self, color: QColor): # Définit la méthode qui prend en paramètre un objet de type QColor (provenant de PyQt6) représentant la nouvelle couleur à appliquer.
        self.pen_color = color # Met à jour la variable interne self.pen_color de l'objet avec cette nouvelle couleur. Cette couleur sera utilisée lors de la création du stylo (QPen) lorsque l'utilisateur tracera un trait


    """
    Cette fonction permet de réinitialiser entièrement la zone de dessin en effaçant tout ce que l'utilisateur a tracé pour retrouver une "feuille blanche".
    """
    def clear(self) -> None: # Définit la méthode clear qui ne prend aucun paramètre supplémentaire (à part self) et ne retourne aucune valeur (None).
        self.image.fill(Qt.GlobalColor.white) # Remplit de nouveau l'image stockée en mémoire tampon (self.image) avec du blanc (Qt.GlobalColor.white), ce qui a pour effet d'effacer tous les traits qui y avaient été dessinés.
        self.update() # Force le widget à se rafraîchir en déclenchant un événement de peinture (paintEvent). Cela permet de mettre à jour immédiatement l'affichage à l'écran pour que l'utilisateur voie le canvas redevenir blanc.

    """
    Cette fonction est un gestionnaire d'événements natif de PyQt6. Elle est déclenchée automatiquement par le système dès que l'utilisateur clique avec la souris sur le canvas. 
    """
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    """
    Cette fonction est également un gestionnaire d'événements natif de PyQt6. Elle est appelée en continu par le système chaque fois que l'utilisateur déplace la souris sur le canvas. C'est le cœur du système de dessin : c'est elle qui trace les traits à l'écran lorsque la souris bouge.
    """
    def mouseMoveEvent(self, event) -> None:
        if self.drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            painter = QPainter(self.image)
            pen = QPen(
                self.pen_color,
                self.pen_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    """
    Cette fonction est également un gestionnaire d'événements natif de PyQt6. Elle est déclenchée automatiquement dès que l'utilisateur relâche le bouton de la souris après avoir dessiné.
    """
    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    """
    Cette fonction est un gestionnaire d'événements natif crucial de PyQt6. Elle est appelée automatiquement par le système chaque fois que le widget doit être affiché ou rafraîchi à l'écran (par exemple, au lancement de l'application, lorsque la fenêtre est redimensionnée, ou chaque fois que la fonction self.update() est appelée après un coup de pinceau).
    """
    def paintEvent(self, event) -> None:
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(0, 0, self.image)

    """
    Cette fonction est un gestionnaire d'événements natif de PyQt6. Elle est déclenchée automatiquement par le système chaque fois que l'utilisateur redimensionne la fenêtre ou le widget du canvas. Son rôle crucial est d'empêcher la perte du dessin lorsque la zone de travail s'agrandit.
    """
    def resizeEvent(self, event) -> None:
        """Conserve le contenu du dessin lors du redimensionnement de la fenêtre."""
        if self.width() > self.image.width() or self.height() > self.image.height():
            new_w = max(self.width(), self.image.width())
            new_h = max(self.height(), self.image.height())
            new_image = QImage(new_w, new_h, QImage.Format.Format_RGB32)
            new_image.fill(Qt.GlobalColor.white)

            painter = QPainter(new_image)
            painter.drawImage(QPoint(0, 0), self.image)
            self.image = new_image

        super().resizeEvent(event)

    """
    
    """
    def get_image_as_numpy(self) -> np.ndarray:
        """Convertit le dessin en matrice NumPy (RGB) pour l'IA."""
        width = self.image.width()
        height = self.image.height()

        ptr = self.image.bits()
        ptr.setsize(self.image.sizeInBytes())
        arr = np.array(ptr).reshape(height, width, 4)

        return arr[:, :, :3]  # Retourne les 3 canaux RVB


class MainWindow(QMainWindow):
    """
    Fenêtre Principale Redimensionnable avec Style Professionnel Modernisé.
    """

    """
    C'est le constructeur de la fenêtre principale. Il est appelé automatiquement lors de l'instanciation de la classe. Son rôle est de configurer la structure générale de la fenêtre (titre, dimensions minimales, style) et de déclencher la construction de l'interface utilisateur.
    """
    def __init__(self, classifier=None): # Appelle le constructeur de la classe parente (QMainWindow de PyQt6) pour initialiser la fenêtre principale.
        super().__init__()
        self.classifier = classifier
        self.setWindowTitle("AI Sketch Recognizer - Studio Pro")
        self.resize(1150, 750)
        self.setMinimumSize(900, 600)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 10)
        main_layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. EN-TÊTE PROFESSIONNEL (Header Bar)
        # -------------------------------------------------------------
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e2d;
                border-bottom: 2px solid #3b3b54;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 12, 20, 12)

        header_title = QLabel("🎨 AI SKETCH RECOGNIZER PRO")
        header_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_title.setStyleSheet("color: #ffffff;")

        header_subtitle = QLabel("Interface Interactive de Reconnaissance de Croquis")
        header_subtitle.setFont(QFont("Segoe UI", 10))
        header_subtitle.setStyleSheet("color: #a2a3b7;")

        header_layout.addWidget(header_title)
        header_layout.addSpacing(15)
        header_layout.addWidget(header_subtitle)
        header_layout.addStretch()

        main_layout.addWidget(header_frame)

        # -------------------------------------------------------------
        # 2. CONTENU PRINCIPAL (Séparation Gauche / Droite)
        # -------------------------------------------------------------
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(15, 0, 15, 0)
        content_layout.setSpacing(15)

        # --- PANNEAU GAUCHE : Zone de Dessin & Barre d'Outils ---
        left_box = QVBoxLayout()

        # Barre d'outils supérieure du dessin
        toolbar_group = QGroupBox("Outils du Pinceau")
        toolbar_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar_group)

        # Sélecteurs de couleur rapide
        toolbar_layout.addWidget(QLabel("Couleur :"))
        colors = [
            ("#000000", "Noir"),
            ("#2196F3", "Bleu"),
            ("#F44336", "Rouge"),
            ("#4CAF50", "Vert"),
            ("#FF9800", "Orange"),
            ("#FFFFFF", "Gomme"),
        ]
        for hex_code, name in colors:
            btn_color = QPushButton()
            btn_color.setToolTip(name)
            btn_color.setFixedSize(26, 26)
            btn_color.setStyleSheet(f"background-color: {hex_code}; border: 1px solid #888; border-radius: 13px;")
            btn_color.clicked.connect(lambda _, c=hex_code: self.change_color(c))
            toolbar_layout.addWidget(btn_color)

        btn_custom_color = QPushButton("🎨 Palette")
        btn_custom_color.setStyleSheet("padding: 4px 8px;")
        btn_custom_color.clicked.connect(self.choose_custom_color)
        toolbar_layout.addWidget(btn_custom_color)

        toolbar_layout.addSpacing(15)

        # Taille du pinceau
        toolbar_layout.addWidget(QLabel("Épaisseur :"))
        self.slider_width = QSlider(Qt.Orientation.Horizontal)
        self.slider_width.setRange(2, 50)
        self.slider_width.setValue(12)
        self.slider_width.setFixedWidth(120)

        self.spin_width = QSpinBox()
        self.spin_width.setRange(2, 50)
        self.spin_width.setValue(12)

        self.slider_width.valueChanged.connect(self.spin_width.setValue)
        self.spin_width.valueChanged.connect(self.slider_width.setValue)
        self.spin_width.valueChanged.connect(self.change_pen_width)

        toolbar_layout.addWidget(self.slider_width)
        toolbar_layout.addWidget(self.spin_width)

        toolbar_layout.addStretch()
        left_box.addWidget(toolbar_group)

        # Canvas de dessin
        self.canvas = DrawingCanvas()
        self.canvas.setStyleSheet("border: 2px solid #b0b0b0; border-radius: 6px; background-color: white;")
        left_box.addWidget(self.canvas, stretch=1)

        # Boutons d'actions
        action_layout = QHBoxLayout()

        self.btn_clear = QPushButton("🗑️ Effacer Tout")
        self.btn_clear.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_clear.clicked.connect(self.canvas.clear)
        action_layout.addWidget(self.btn_clear)

        self.btn_save = QPushButton("💾 Enregistrer")
        self.btn_save.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_save.clicked.connect(self.on_save_clicked)
        action_layout.addWidget(self.btn_save)

        self.btn_sketch = QPushButton("✏️ Effet Esquisse")
        self.btn_sketch.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_sketch.clicked.connect(self.on_sketch_clicked)
        action_layout.addWidget(self.btn_sketch)

        self.btn_predict = QPushButton("⚡ Reconnaître (IA)")
        self.btn_predict.setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold; padding: 10px; border-radius: 5px; font-size: 14px;")
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        action_layout.addWidget(self.btn_predict)

        left_box.addLayout(action_layout)

        content_layout.addLayout(left_box, stretch=2)

        # --- PANNEAU DROIT : Instructions & Résultats IA ---
        right_box = QVBoxLayout()

        # Group Box : Zone d'Instructions Étape par Étape
        instructions_group = QGroupBox("📌 Guide d'Utilisation")
        instructions_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #f9f9fc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        instructions_layout = QVBoxLayout(instructions_group)

        steps = [
            ("1. Dessinez", "Tracez un objet sur le canvas blanc (chat, maison, arbre...)."),
            ("2. Personnalisez", "Ajustez la couleur et l'épaisseur du pinceau selon vos besoins."),
            ("3. Appliquez les effets", "Sauvegardez votre dessin ou appliquez l'effet Esquisse OpenCV."),
            ("4. Analysez", "Cliquez sur '⚡ Reconnaître' pour afficher la prédiction de l'IA."),
        ]

        for step_title, step_desc in steps:
            step_label = QLabel(f"<b>{step_title}</b> : <br><font color='#555'>{step_desc}</font>")
            step_label.setWordWrap(True)
            step_label.setStyleSheet("margin-bottom: 6px;")
            instructions_layout.addWidget(step_label)

        right_box.addWidget(instructions_group)

        # Group Box : Résultats du Top 3
        results_group = QGroupBox("📊 Résultats de la Prédiction (Top 3)")
        results_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        results_layout = QVBoxLayout(results_group)

        self.prediction_labels = []
        self.progress_bars = []

        for i in range(3):
            row_layout = QVBoxLayout()
            lbl = QLabel(f"Top {i+1} : -")
            lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(22)
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #2196F3;
                    border-radius: 3px;
                }
            """)

            row_layout.addWidget(lbl)
            row_layout.addWidget(bar)
            results_layout.addLayout(row_layout)

            self.prediction_labels.append(lbl)
            self.progress_bars.append(bar)

        right_box.addWidget(results_group, stretch=1)

        content_layout.addLayout(right_box, stretch=1)

        main_layout.addLayout(content_layout)

    # -------------------------------------------------------------
    # FONCTIONS DE GESTION DU PINCEAU
    # -------------------------------------------------------------

    """
    Cette fonction est appelée chaque fois que l'utilisateur clique sur l'un des boutons de couleur rapide de la barre d'outils (noir, bleu, rouge, vert, orange ou gomme/blanc).
    """
    def change_color(self, hex_code: str): # Définit la méthode qui prend en paramètre un code couleur au format hexadécimal sous forme de chaîne de caractères (str), par exemple "#2196F3" pour le bleu.
        self.canvas.set_pen_color(QColor(hex_code)) # Convertit le code hexadécimal en un objet de couleur PyQt (QColor(hex_code)), puis appelle la méthode set_pen_color du widget de dessin (self.canvas) pour mettre à jour la couleur active du pinceau.

    """
    Cette fonction est déclenchée lorsque l'utilisateur clique sur le bouton "🎨 Palette" de la barre d'outils. Son rôle est d'ouvrir la fenêtre système de sélection de couleurs pour permettre à l'utilisateur de choisir n'importe quelle couleur personnalisée.
    """
    def choose_custom_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.set_pen_color(color)

    """
    Cette fonction est appelée chaque fois que l'utilisateur déplace le curseur (QSlider) ou change la valeur numérique dans le sélecteur (QSpinBox) de l'épaisseur du pinceau.
    """
    def change_pen_width(self, width: int):
        self.canvas.set_pen_width(width)

    # -------------------------------------------------------------
    # FONCTIONS D'ACTIONS
    # -------------------------------------------------------------

    """
    Cette fonction est déclenchée lorsque l'utilisateur clique sur le bouton "💾 Enregistrer" de l'interface. Son rôle est de récupérer le dessin actuel, de le convertir pour qu'il soit compatible avec OpenCV, puis de l'envoyer au module externe de filtres pour l'enregistrer dans une galerie.
    """
    def on_save_clicked(self):
        img_np = self.canvas.get_image_as_numpy()
        try:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            filepath = BonusFilters.save_to_gallery(img_bgr)
            print(f"Dessin enregistré avec succès dans : {filepath}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")

    """
    Cette fonction est déclenchée lorsque l'utilisateur clique sur le bouton "✏️ Effet Esquisse" de l'interface. Son rôle est de récupérer le dessin actuel, de le transformer en utilisant un filtre artistique OpenCV via une classe externe, puis d'afficher le résultat dans une fenêtre d'aperçu dédiée.
    """
    def on_sketch_clicked(self):
        img_np = self.canvas.get_image_as_numpy()
        try:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            sketch_img = BonusFilters.apply_sketch_effect(img_bgr)
            cv2.imshow("Aperçu Effet Esquisse", sketch_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Erreur lors de l'application du filtre : {e}")

    """
    Cette fonction est déclenchée lorsque l'utilisateur clique sur le bouton "⚡ Reconnaître (IA)". Son rôle est de récupérer le dessin, de l'envoyer au modèle d'intelligence artificielle pour analyse, puis de mettre à jour l'affichage avec les prédictions (le Top 3).
    """
    def on_predict_clicked(self):
        img_np = self.canvas.get_image_as_numpy()

        if self.classifier is None:
            self.update_results_display(
                [("chat (test)", 0.92), ("maison", 0.05), ("arbre", 0.03)]
            )
            return

        try:
            predictions = self.classifier.predict(img_np, top_k=3)
            self.update_results_display(predictions)
        except Exception as e:
            print(f"Erreur d'inférence : {e}")

    """
    Cette fonction est appelée après chaque prédiction de l'IA (ou lors de la simulation de test). Son rôle est de mettre à jour visuellement les labels textuels et les barres de progression du Top 3 dans l'interface graphique.
    """
    def update_results_display(self, predictions):
        for i, (class_name, score) in enumerate(predictions):
            if i < len(self.prediction_labels):
                percentage = int(score * 100)
                self.prediction_labels[i].setText(f"Top {i+1} : {class_name} ({percentage}%)")
                self.progress_bars[i].setValue(percentage)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(classifier=None)
    window.show()
    sys.exit(app.exec())