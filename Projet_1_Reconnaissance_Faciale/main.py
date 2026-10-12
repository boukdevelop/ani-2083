import cv2
from gui import AppInterface
# Importe tes modules ici (exemple)
# from camera_stream import CameraStream
# from face_matcher import FaceMatcher

class MainApp:
    def __init__(self):
        self.app = AppInterface()
        
        # Initialisation de la caméra (Remplace par ton module camera_stream.py de l'Étudiant 4)
        self.cap = cv2.VideoCapture(0) # 0 pour la webcam par défaut

        # Lancer la boucle de rafraîchissement vidéo
        self.update_loop()

    def update_loop(self):
        """Boucle principale gérée par after() pour ne pas bloquer l'interface Tkinter"""
        ret, frame = self.cap.read()
        if ret:
            # 1. Traitement de la caméra (ex: redimensionnement, conversion BGR -> RGB)
            # (Géré normalement par l'Étudiant 4 dans camera_stream.py)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. Reconnaissance / Matching (ex: appel de face_matcher.py de l'Étudiant 1)
            # Ici tu dessines les bounding boxes, noms et scores sur 'frame_rgb'

            # 3. Transmission de l'image traitée vers l'UI
            self.app.update_video_feed(frame_rgb)

        # Rappel de la fonction toutes les 15 millisecondes (~60 FPS)
        self.app.after(15, self.update_loop)

    def run(self):
        self.app.mainloop()
        # Libération de la caméra à la fermeture de l'application
        self.cap.release()

if __name__ == "__main__":
    main_app = MainApp()
    main_app.run()