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

        
