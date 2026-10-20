# main.py
import sys
import os
from PyQt6.QtWidgets import QApplication
from ui_app import MainWindow
from inference_engine import InferenceEngine

def main():
    # Initialisation de l'application PyQt
    app = QApplication(sys.argv)
    
    print("[INFO] Initialisation du moteur d'inférence (AI Sketch Recognizer)...")
    try:
        # Chargement du modèle de Deep Learning et des classes
        classifier = InferenceEngine()
    except Exception as e:
        print(f"[ERREUR] Impossible d'initialiser le modèle d'IA : {e}")
        classifier = None

    # Lancement de la fenêtre principale en injectant le classifieur réel
    window = MainWindow(classifier=classifier)
    window.show()
    
    # Exécution de la boucle événementielle de l'application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()