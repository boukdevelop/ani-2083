import cv2
import face_recognition
from gui import AppInterface
from camera_stream import CameraStream
from face_matcher import FaceMatcher

class MainApp:
    def __init__(self):
        self.app = AppInterface()
        
        # Initialisation de la caméra (échelle 0.25 pour fluidifier)
        self.camera = CameraStream(src=0, scale=0.25)
        if not self.camera.open():
            print("Erreur : Impossible d'ouvrir la caméra.")
            return

        # Chargement du module de reconnaissance
        try:
            self.matcher = FaceMatcher("encoded_faces.pkl", tolerance=0.50)
        except Exception as e:
            print(f"Avertissement FaceMatcher : {e}")
            self.matcher = None

        # Variables pour optimiser la cadence (saut de frames)
        self.face_locations = []
        self.face_names = []
        self.frame_count = 0

        # Démarrage de la boucle vidéo
        self.update_loop()

    def update_loop(self):
        """Boucle principale executée periodiquement pour traiter la vidéo en temps réel."""
        ret, frame_bgr, frame_rgb_small = self.camera.read_frame()
        
        if ret:
            self.frame_count += 1

            # On effectue la détection UNIQUEMENT toutes les 2 frames 
            if self.frame_count % 2 == 0:
                if self.matcher is not None and self.matcher.known_encodings is not None:
                    # Détection 
                    self.face_locations = face_recognition.face_locations(frame_rgb_small, model="hog")
                    face_encodings = face_recognition.face_encodings(frame_rgb_small, self.face_locations)

                    self.face_names = []
                    for face_encoding in face_encodings:
                        name, confidence = self.matcher.match(face_encoding)
                        is_known = (name != "Unknown / Inconnu")
                        if is_known:
                            display_name = f"{name} ({confidence:.1f}%)"
                        else:
                            display_name = f"Inconnu ({confidence:.1f}%)"
                        self.face_names.append((display_name, is_known))

            for (top, right, bottom, left), (display_name, is_known) in zip(self.face_locations, self.face_names):
                # Remettre à l'échelle d'origine
                top = int(top * 4)
                right = int(right * 4)
                bottom = int(bottom * 4)
                left = int(left * 4)

                color = (0, 255, 0) if is_known else (0, 0, 255)

                cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
                cv2.putText(frame_bgr, display_name, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Transmission de l'image traitée vers l'UI
            final_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.app.update_video_feed(final_frame_rgb)

        # Cadence de rafraîchissement de l'interface (15ms)
        self.app.after(15, self.update_loop)

    def run(self):
        self.app.mainloop()
        self.camera.release()

if __name__ == "__main__":
    main_app = MainApp()
    main_app.run()