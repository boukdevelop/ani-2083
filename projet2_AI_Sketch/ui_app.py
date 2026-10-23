"""
ui_app.py — Interface Utilisateur Professionnelle (PyQt6)
Application interactive de dessin et de reconnaissance de croquis par IA.
"""

import sys
import os
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QFrame,
    QSlider,
    QSpinBox,
    QColorDialog,
)
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QPainter, QPen, QImage, QColor, QFont, QCursor

from bonus_filters import BonusFilters


class DrawingCanvas(QWidget):
    """
    Zone de dessin interactif a la souris avec support du pinceau et de la gomme.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 420)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        self.pen_width = 14
        self.pen_color = QColor(20, 20, 20)
        self.is_eraser = False
        self.drawing = False
        self.last_point = QPoint()

        self.image = QImage(QSize(500, 500), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)

    def set_pen_width(self, width: int):
        self.pen_width = width

    def set_pen_color(self, color: QColor):
        self.pen_color = color
        self.is_eraser = False

    def set_eraser_mode(self, enabled: bool):
        self.is_eraser = enabled

    def clear(self) -> None:
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        if self.drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            active_color = QColor(255, 255, 255) if self.is_eraser else self.pen_color
            active_width = self.pen_width * 2 if self.is_eraser else self.pen_width

            pen = QPen(
                active_color,
                active_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event) -> None:
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(0, 0, self.image)

    def resizeEvent(self, event) -> None:
        if self.width() > self.image.width() or self.height() > self.image.height():
            new_w = max(self.width(), self.image.width())
            new_h = max(self.height(), self.image.height())
            new_image = QImage(new_w, new_h, QImage.Format.Format_RGB32)
            new_image.fill(Qt.GlobalColor.white)

            painter = QPainter(new_image)
            painter.drawImage(QPoint(0, 0), self.image)
            self.image = new_image

        super().resizeEvent(event)

    def get_image_as_numpy(self) -> np.ndarray:
        """Retourne la matrice NumPy (RGB) du dessin actuel."""
        width = self.image.width()
        height = self.image.height()
        ptr = self.image.bits()
        ptr.setsize(self.image.sizeInBytes())
        arr = np.array(ptr).reshape(height, width, 4)
        return arr[:, :, :3]


class MainWindow(QMainWindow):
    """
    Fenetre principale moderne au design professionnel (UI/UX Slate Dark / Clean Pro).
    """

    CLASSES_DISPLAY = {
        "cat": "Chat",
        "house": "Maison",
        "car": "Voiture",
        "tree": "Arbre",
        "sun": "Soleil",
        "umbrella": "Parapluie",
        "apple": "Pomme",
        "airplane": "Avion",
        "fish": "Poisson",
        "star": "Etoile",
    }

    def __init__(self, classifier=None):
        super().__init__()
        self.classifier = classifier
        self.setWindowTitle("AI Sketch Recognizer - Studio Pro")
        self.resize(1180, 760)
        self.setMinimumSize(960, 640)

        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; }
            QWidget { color: #e2e8f0; font-family: 'Segoe UI', Roboto, sans-serif; }
            QFrame#card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
            QLabel { background: transparent; }
            QPushButton { font-weight: 600; border-radius: 8px; padding: 10px 16px; border: none; }
            QSlider::groove:horizontal { height: 6px; background: #334155; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #38bdf8; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #f8fafc; border: 2px solid #0284c7; width: 16px;
                margin-top: -5px; margin-bottom: -5px; border-radius: 8px;
            }
            QSpinBox {
                background-color: #0f172a; border: 1px solid #334155; border-radius: 6px;
                padding: 4px 8px; color: #f8fafc; font-weight: bold; min-width: 45px;
            }
        """)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 16, 20, 20)
        main_layout.setSpacing(16)

        # En-tete
        header_frame = QFrame()
        header_frame.setObjectName("card")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(22, 14, 22, 14)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        header_title = QLabel("AI SKETCH RECOGNIZER")
        header_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_title.setStyleSheet("color: #f8fafc; letter-spacing: 1px;")

        header_subtitle = QLabel("Systeme de classification de croquis en temps reel par Reseau Convolutif (CNN)")
        header_subtitle.setFont(QFont("Segoe UI", 10))
        header_subtitle.setStyleSheet("color: #94a3b8;")

        title_col.addWidget(header_title)
        title_col.addWidget(header_subtitle)
        header_layout.addLayout(title_col)
        header_layout.addStretch()

        is_active = bool(self.classifier and getattr(self.classifier, "model", None))
        self.badge_status = QLabel("Modele Actif" if is_active else "Mode Simulation")
        badge_bg = "#064e3b" if is_active else "#78350f"
        badge_fg = "#34d399" if is_active else "#fde047"
        self.badge_status.setStyleSheet(f"""
            background-color: {badge_bg}; color: {badge_fg}; font-size: 11px;
            font-weight: bold; padding: 6px 14px; border-radius: 14px; border: 1px solid {badge_fg};
        """)
        header_layout.addWidget(self.badge_status)
        main_layout.addWidget(header_frame)

        # Contenu principal
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Studio de dessin (Gauche)
        left_card = QFrame()
        left_card.setObjectName("card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(14)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.btn_tool_brush = QPushButton("Pinceau")
        self.btn_tool_brush.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_tool_brush.setStyleSheet("background-color: #0284c7; color: white; padding: 7px 14px;")
        self.btn_tool_brush.clicked.connect(lambda: self.switch_tool("brush"))

        self.btn_tool_eraser = QPushButton("Gomme")
        self.btn_tool_eraser.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_tool_eraser.setStyleSheet("background-color: #334155; color: #cbd5e1; padding: 7px 14px;")
        self.btn_tool_eraser.clicked.connect(lambda: self.switch_tool("eraser"))

        toolbar.addWidget(self.btn_tool_brush)
        toolbar.addWidget(self.btn_tool_eraser)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #334155;")
        toolbar.addWidget(sep)

        toolbar.addWidget(QLabel("Couleur :"))
        for hex_code, name in [("#0f172a", "Noir"), ("#2563eb", "Bleu"), ("#dc2626", "Rouge"), ("#16a34a", "Vert"), ("#d97706", "Ambre")]:
            btn_c = QPushButton()
            btn_c.setToolTip(name)
            btn_c.setFixedSize(24, 24)
            btn_c.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_c.setStyleSheet(f"background-color: {hex_code}; border: 2px solid #64748b; border-radius: 12px;")
            btn_c.clicked.connect(lambda _, c=hex_code: self.change_color(c))
            toolbar.addWidget(btn_c)

        btn_custom = QPushButton("Palette...")
        btn_custom.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_custom.setStyleSheet("background-color: #1e293b; border: 1px solid #475569; color: #cbd5e1; padding: 5px 10px; font-size: 12px;")
        btn_custom.clicked.connect(self.choose_custom_color)
        toolbar.addWidget(btn_custom)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #334155;")
        toolbar.addWidget(sep2)

        toolbar.addWidget(QLabel("Epaisseur :"))
        self.slider_width = QSlider(Qt.Orientation.Horizontal)
        self.slider_width.setRange(4, 50)
        self.slider_width.setValue(14)
        self.slider_width.setFixedWidth(100)

        self.spin_width = QSpinBox()
        self.spin_width.setRange(4, 50)
        self.spin_width.setValue(14)

        self.slider_width.valueChanged.connect(self.spin_width.setValue)
        self.spin_width.valueChanged.connect(self.slider_width.setValue)

        toolbar.addWidget(self.slider_width)
        toolbar.addWidget(self.spin_width)
        toolbar.addStretch()
        left_layout.addLayout(toolbar)

        # Canvas
        self.canvas_container = QFrame()
        self.canvas_container.setStyleSheet("background-color: #ffffff; border: 2px solid #334155; border-radius: 10px;")
        c_layout = QVBoxLayout(self.canvas_container)
        c_layout.setContentsMargins(4, 4, 4, 4)
        self.canvas = DrawingCanvas()
        c_layout.addWidget(self.canvas)
        left_layout.addWidget(self.canvas_container, stretch=1)

        # Le canvas doit exister avant de brancher le signal de l'épaisseur
        self.spin_width.valueChanged.connect(self.canvas.set_pen_width)

        # Actions inferieures
        actions = QHBoxLayout()
        actions.setSpacing(12)

        self.btn_clear = QPushButton("Effacer Tout")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setStyleSheet("background-color: #334155; color: #f87171; border: 1px solid #475569;")
        self.btn_clear.clicked.connect(self.on_clear_clicked)
        actions.addWidget(self.btn_clear)

        self.btn_save = QPushButton("Enregistrer Galerie")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #0284c7;")
        self.btn_save.clicked.connect(self.on_save_clicked)
        actions.addWidget(self.btn_save)

        self.btn_sketch = QPushButton("Filtre Esquisse")
        self.btn_sketch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sketch.setStyleSheet("background-color: #1e293b; color: #c084fc; border: 1px solid #9333ea;")
        self.btn_sketch.clicked.connect(self.on_sketch_clicked)
        actions.addWidget(self.btn_sketch)

        actions.addStretch()

        self.btn_predict = QPushButton("IDENTIFIER L'OBJET")
        self.btn_predict.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_predict.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_predict.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
                color: #ffffff; padding: 12px 28px; border-radius: 8px;
            }
            QPushButton:hover { background: #10b981; }
        """)
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        actions.addWidget(self.btn_predict)
        left_layout.addLayout(actions)
        content_layout.addWidget(left_card, stretch=3)

        # Panneau d'Analyse IA (Droite)
        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(18)

        sec_title = QLabel("RESULTAT DE LA RECONNAISSANCE")
        sec_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        sec_title.setStyleSheet("color: #94a3b8; letter-spacing: 1px;")
        right_layout.addWidget(sec_title)

        # Carte de resultat
        self.result_box = QFrame()
        self.result_box.setMinimumHeight(180)
        self.result_box.setStyleSheet("background-color: #0f172a; border: 2px solid #334155; border-radius: 12px; padding: 20px;")
        res_layout = QVBoxLayout(self.result_box)
        res_layout.setSpacing(10)
        res_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_result_name = QLabel("En attente d'un dessin...")
        self.lbl_result_name.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.lbl_result_name.setStyleSheet("color: #f8fafc;")
        self.lbl_result_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result_name.setWordWrap(True)
        res_layout.addWidget(self.lbl_result_name)

        self.lbl_result_status = QLabel("Tracez un croquis puis cliquez sur Identifier")
        self.lbl_result_status.setFont(QFont("Segoe UI", 10))
        self.lbl_result_status.setStyleSheet("color: #94a3b8;")
        self.lbl_result_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_layout.addWidget(self.lbl_result_status)

        res_layout.addSpacing(10)

        # Jauge
        gauge_container = QVBoxLayout()
        gauge_container.setSpacing(6)

        gauge_header = QHBoxLayout()
        lbl_conf_title = QLabel("Indice de confiance :")
        lbl_conf_title.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        lbl_conf_title.setStyleSheet("color: #94a3b8;")

        self.lbl_confidence_val = QLabel("0.0%")
        self.lbl_confidence_val.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.lbl_confidence_val.setStyleSheet("color: #38bdf8;")

        gauge_header.addWidget(lbl_conf_title)
        gauge_header.addStretch()
        gauge_header.addWidget(self.lbl_confidence_val)
        gauge_container.addLayout(gauge_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background-color: #1e293b; border: 1px solid #334155; border-radius: 7px; }
            QProgressBar::chunk { background: #0284c7; border-radius: 6px; }
        """)
        gauge_container.addWidget(self.progress_bar)
        res_layout.addLayout(gauge_container)
        right_layout.addWidget(self.result_box)

        # Categories supportees
        classes_frame = QFrame()
        classes_frame.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 14px;")
        classes_layout = QVBoxLayout(classes_frame)
        classes_layout.setSpacing(8)

        classes_title = QLabel("10 Categories Reconnues")
        classes_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        classes_title.setStyleSheet("color: #38bdf8;")
        classes_layout.addWidget(classes_title)

        lbl_tags = QLabel("Chat • Maison • Voiture • Arbre • Soleil\nParapluie • Pomme • Avion • Poisson • Etoile")
        lbl_tags.setFont(QFont("Segoe UI", 9))
        lbl_tags.setStyleSheet("color: #cbd5e1; line-height: 160%;")
        lbl_tags.setWordWrap(True)
        classes_layout.addWidget(lbl_tags)
        right_layout.addWidget(classes_frame)

        right_layout.addStretch()

        self.lbl_action_status = QLabel("Pret.")
        self.lbl_action_status.setFont(QFont("Segoe UI", 9))
        self.lbl_action_status.setStyleSheet("color: #64748b;")
        right_layout.addWidget(self.lbl_action_status)

        content_layout.addWidget(right_card, stretch=2)
        main_layout.addLayout(content_layout)

    def switch_tool(self, tool_type: str):
        if tool_type == "eraser":
            self.canvas.set_eraser_mode(True)
            self.btn_tool_eraser.setStyleSheet("background-color: #f59e0b; color: #0f172a; font-weight: bold; padding: 7px 14px;")
            self.btn_tool_brush.setStyleSheet("background-color: #334155; color: #cbd5e1; padding: 7px 14px;")
            self.lbl_action_status.setText("Outil actif : Gomme")
        else:
            self.canvas.set_eraser_mode(False)
            self.btn_tool_brush.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 7px 14px;")
            self.btn_tool_eraser.setStyleSheet("background-color: #334155; color: #cbd5e1; padding: 7px 14px;")
            self.lbl_action_status.setText("Outil actif : Pinceau")

    def change_color(self, hex_code: str):
        self.canvas.set_pen_color(QColor(hex_code))
        self.switch_tool("brush")

    def choose_custom_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.set_pen_color(color)
            self.switch_tool("brush")

    def on_clear_clicked(self):
        self.canvas.clear()
        self.reset_result_display()
        self.lbl_action_status.setText("Canvas efface.")

    def on_save_clicked(self):
        img_np = self.canvas.get_image_as_numpy()
        try:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            filepath = BonusFilters.save_to_gallery(img_bgr)
            self.lbl_action_status.setText(f"Enregistre : {os.path.basename(filepath)}")
        except Exception as e:
            self.lbl_action_status.setText(f"Erreur enregistrement : {e}")

    def on_sketch_clicked(self):
        img_np = self.canvas.get_image_as_numpy()
        try:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            sketch_img = BonusFilters.apply_sketch_effect(img_bgr)
            cv2.imshow("Apercu Filtre Esquisse (Fermer avec une touche)", sketch_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.lbl_action_status.setText("Filtre esquisse applique.")
        except Exception as e:
            self.lbl_action_status.setText(f"Erreur filtre : {e}")

    def on_predict_clicked(self):
        img_np = self.canvas.get_image_as_numpy()

        if self.classifier is None:
            self.display_result("house", 0.96)
            return

        try:
            # InferenceEngine.predict() retourne directement un tuple (classe, score de confiance)
            top_class, top_score = self.classifier.predict(img_np)

            self.display_result(top_class, top_score)
        except Exception as e:
            self.lbl_action_status.setText(f"Erreur d'inference : {e}")

    def reset_result_display(self):
        self.lbl_result_name.setText("En attente d'un dessin...")
        self.lbl_result_status.setText("Tracez un croquis puis cliquez sur Identifier")
        self.lbl_confidence_val.setText("0.0%")
        self.progress_bar.setValue(0)
        self.set_card_state("neutral")

    def display_result(self, class_name: str, confidence: float):
        percent = int(confidence * 100)
        self.lbl_confidence_val.setText(f"{percent}% ({confidence:.2f})")
        self.progress_bar.setValue(percent)

        if "Canvas vide" in class_name:
            self.lbl_result_name.setText("Canvas vide")
            self.lbl_result_status.setText("Veuillez tracer un dessin avant de lancer l'analyse.")
            self.set_card_state("neutral")
            return

        if any(w in class_name.lower() for w in ["inconnu", "incertain", "pas sur", "pas sûr", "unknown"]):
            self.lbl_result_name.setText("Inconnu / Pas sur")
            self.lbl_result_status.setText("L'IA n'est pas assez certaine (confiance < 55%).")
            self.set_card_state("warning")
            return

        display_label = self.CLASSES_DISPLAY.get(class_name.lower(), class_name.capitalize())
        self.lbl_result_name.setText(display_label)
        self.lbl_result_status.setText(f"Identifie avec succes ({class_name})")
        self.set_card_state("success")
        self.lbl_action_status.setText(f"Reconnaissance terminee : {display_label}")

    def set_card_state(self, state: str):
        if state == "success":
            border, chunk, color = "#059669", "#10b981", "#34d399"
        elif state == "warning":
            border, chunk, color = "#d97706", "#f59e0b", "#f59e0b"
        else:
            border, chunk, color = "#334155", "#0284c7", "#f8fafc"

        self.lbl_result_name.setStyleSheet(f"color: {color}; font-weight: bold; background: transparent;")
        self.lbl_confidence_val.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.result_box.setStyleSheet(f"background-color: #0f172a; border: 2px solid {border}; border-radius: 12px; padding: 20px;")
        self.progress_bar.setStyleSheet(f"QProgressBar {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 7px; }} QProgressBar::chunk {{ background: {chunk}; border-radius: 6px; }}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(classifier=None)
    window.show()
    sys.exit(app.exec())
