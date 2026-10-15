import cv2
import face_recognition
from gui import AppInterface
from camera_stream import CameraStream
from face_matcher import FaceMatcher

class MainApp:
    def __init__(self):
        self.app = AppInterface()
        
        # Initialisation de la caméra (échelle 0.25 pour fluidifier le traitement)
        self.camera = CameraStream(src=0, scale=0.25)
        if not self.camera.open():
            print("Erreur : Impossible d'ouvrir la caméra.")
            return

        # Chargement du module de reconnaissance (FaceMatcher)
        try:
            self.matcher = FaceMatcher("encoded_faces.pkl", tolerance=0.50)
        except Exception as e:
            print(f"Avertissement FaceMatcher : {e}")
            self.matcher = None

        # Démarrage de la boucle vidéo
        self.update_loop()

    def update_loop(self):
        """Boucle principale exécutée périodiquement pour traiter la vidéo en temps réel."""
        ret, frame_bgr, frame_rgb_small = self.camera.read_frame()
        
        if ret:
            if self.matcher is not None and self.matcher.known_encodings is not None:
                face_locations = face_recognition.face_locations(frame_rgb_small, model="hog")
                face_encodings = face_recognition.face_encodings(frame_rgb_small, face_locations)

                # Pour chaque visage trouvé, on effectue le matching
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name, confidence = self.matcher.match(face_encoding)

                    # Remettre les coordonnées à l'échelle d'origine (puisqu'on a réduit à 0.25, on multiplie par 4)
                    top = int(top * 4)
                    right = int(right * 4)
                    bottom = int(bottom * 4)
                    left = int(left * 4)

                    # Choix de la couleur selon le résultat (Vert si reconnu, Rouge si inconnu)
                    color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)

                    # Dessiner le rectangle et le texte sur l'image BGR d'origine
                    cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
                    label = f"{name} ({confidence:.1f}%)"
                    cv2.putText(frame_bgr, label, (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Conversion finale de BGR vers RGB pour l'affichage dans CustomTkinter (`gui.py`)
            final_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.app.update_video_feed(final_frame_rgb)

        # Rappel de la boucle toutes les 15 millisecondes (~60 FPS)
        self.app.after(15, self.update_loop)

    def run(self):
        self.app.mainloop()
        # Libération de la caméra à la fermeture
        self.camera.release()

if __name__ == "__main__":
    main_app = MainApp()
    main_app.run()