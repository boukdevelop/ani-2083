import cv2

#creation de la classe qui gere la camera
class CameraStream:
    """
    Gère le flux vidéo d'une caméra (webcam) pour la capture d'images,
    le redimensionnement et la conversion des espaces de couleurs 
    (notamment pour des bibliothèques comme face_recognition).
    """

    def __init__(self, src=0, scale=0.25):
        """
        Initialise le flux de la caméra.

        Args:
            src (int or str): Identifiant de la source de la caméra (0 pour la webcam par défaut).
            scale (float): Facteur de réduction pour redimensionner les images (ex: 0.25 pour 25%).
        """
        self.src = src # 0 demande d'utiliser la camera par defaut du pc
        self.scale = scale # 0.25 est le facteur de reduction de 25% 
        self.cap = None 

    def open(self): #la methode d'ouvert de la webcam
        """
        Ouvre la connexion avec la webcam.

        Returns:
            bool: True si la caméra s'est ouverte avec succès, False sinon.
        """
        self.cap = cv2.VideoCapture(self.src) # demarre la webcam 
        return self.cap.isOpened() # renvoi true quand la camera est ouverte

    def read_frame(self):
        """
        Lit une image depuis la caméra, applique un effet miroir, 
        la redimensionne et la convertit de BGR vers RGB.

        Returns:
            tuple: 
                - bool: Succès de la lecture (True/False).
                - numpy.ndarray or None: L'image originale au format BGR.
                - numpy.ndarray or None: L'image réduite et convertie au format RGB.
        """
        if self.cap is None or not self.cap.isOpened():
           return False, None, None

        # capture de l'image brute depuis le flux vidéo
        ret, frame_bgr = self.cap.read()
        if not ret:
            return False, None, None

        # annuler l'effet mirroir horizontal pour plus de naturel
        frame_bgr = cv2.flip(frame_bgr, 1)

        # Réduction de la taille de l'image selon le facteur d'échelle (optimisation des performances) 
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=self.scale, fy=self.scale)

        # Conversion de l'espace de couleurs BGR (utilisé par OpenCV) vers RGB (exigé par face_recognition)
        frame_rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        return True, frame_bgr, frame_rgb_small

    def release(self):
        # eteindre la camera
        """
        Libère les ressources de la caméra et ferme proprement le flux.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
  cam = CameraStream()

  if cam.open():
    print("Caméra lancée ! Appuyez sur 'q' pour quitter.")
    while True:
      ok, image_normale, image_ia = cam.read_frame()
      if not ok:
        break

      # Afficher l'image à l'écran
      cv2.imshow("Test Etudiant 4", image_normale)

      # Quitter si on appuie sur 'q'
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cam.release()  #[cite: 5]
    cv2.destroyAllWindows()  #[cite: 5]
        
