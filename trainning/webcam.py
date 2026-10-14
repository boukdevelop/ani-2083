import cv2

webcam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
position = (50,50)
taille = 1
rouge =(0, 0, 255)
epaisseur = 3

while(True) :
    # recuperation fram par fram
    ret, frame = webcam.read()

    frame_ecriture = cv2.putText(frame, "fuck you hervé",
                                position,
                                font,
                                taille, 
                                rouge, 
                                epaisseur, cv2.LINE_AA)
    # on affiche chaque frame
    cv2.imshow('frames', frame_ecriture)
    # on attent que le touche q soit presser pour fermer le program
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
webcam.release()
cv2.destroyAllWindows()