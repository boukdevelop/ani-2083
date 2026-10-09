import cv2

#creation de la classe qui gere la camera
class CameraStream:

    def __init__(self, src=0, scale=0.25): # le constructeur de la classe
        self.src = src # 0 demande d'utiliser la camera par defaut du pc
        self.scale = scale # 0.25 est le facteur de reduction de 25% 
        self.cap = None 

    def open(self): #la methode d'ouvert de la webcam
        self.cap = cv2.VideoCapture(self.src) # demarre la webcam 
        return self.cap.isOpened() # renvoi true quand la camera est ouverte

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened()
           return False, None, None

        # capture de la photo brute 
        ret, frame_bgr = self.cap()
        if not ret:
            return False, None, None

        # annuler l'effet mirroir
        frame_bgr = cv2.flip(frame_bgr, 1)

        # redimension a 25%  
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=self.scale, fy=self.scale)

        # convertir de BGR (opencv) vers RGB (exige par face_recognition) 
        frame_rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        return True, frame_bgr, frame_rgb_small

    def release(self):
        # eteindre la camera
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()



        
