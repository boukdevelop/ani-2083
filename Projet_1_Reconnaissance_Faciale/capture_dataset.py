import cv2
import os
import time

# Liste exacte des 20 consignes sans aucun accent
POSES_20_GUIDE = [
    "1/20 : Vue de FACE -- (De PRES)",
    "2/20 : Vue de FACE -- (Un peu ELOIGNE)",
    "3/20 : Vue de GAUCHE -- (De PRES)",
    "4/20 : Vue de GAUCHE -- (Un peu ELOIGNE)",
    "5/20 : Vue de DROITE -- (De PRES)",
    "6/20 : Vue de DROITE -- (Un peu ELOIGNE)",
    "7/20 : Vue du HAUT -- (De PRES)",
    "8/20 : Vue du HAUT -- (Un peu ELOIGNE)",
    "9/20 : Vue du BAS -- (De PRES)",
    "10/20 : Vue du BAS -- (Un peu ELOIGNE)",
    "11/20 : SOURIRE de face -- (De PRES)",
    "12/20 : SOURIRE de face -- (Un peu ELOIGNE)",
    "13/20 : Tete inclinee GAUCHE -- (De PRES)",
    "14/20 : Tete inclinee GAUCHE -- (Un peu ELOIGNE)",
    "15/20 : Tete inclinee DROITE -- (De PRES)",
    "16/20 : Tete inclinee DROITE -- (Un peu ELOIGNE)",
    "17/20 : Air ETONNE / Neutre -- (De PRES)",
    "18/20 : Air ETONNE / Neutre -- (Un peu ELOIGNE)",
    "19/20 : Yeux legerement fermes/clignes -- (De PRES)",
    "20/20 : Yeux legerement fermes/clignes -- (Un peu ELOIGNE)"
]

def capture_person_dataset():
    person_name = input("Entrez le nom_prenom de la personne (ex: jean_dupont) : ").strip().lower()
    
    if not person_name:
        print("❌ Nom invalide. Annulation.")
        return

    output_dir = os.path.join("Projet_1_Reconnaissance_Faciale\data", "known_faces", person_name)
    os.makedirs(output_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))]
    count = len(existing_files)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'acceder a la webcam.")
        return

    print("\n--- INSTRUCTIONS ---")
    print(" Placez-vous selon la consigne affichee a l'ecran.")
    print(" Appuyez sur [ESPACE] pour valider et prendre la photo.")
    print(" Appuyez sur [Q] pour quitter.")
    print("---------------------\n")

    step_index = 0
    feedback_text = ""
    feedback_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture du flux video.")
            break

        # Inversion miroir pour un affichage normal
        frame = cv2.flip(frame, 1)

        display_frame = frame.copy()
        current_time = time.time()

        # Consigne actuelle
        if step_index < len(POSES_20_GUIDE):
            current_instruction = POSES_20_GUIDE[step_index]
            sub_instruction = "Appuyez sur [ESPACE] pour capturer"
        else:
            current_instruction = "Serie de 20 photos complete !"
            sub_instruction = "Appuyez sur [Q] pour quitter"

        # En-tete avec consigne
        cv2.rectangle(display_frame, (10, 10), (display_frame.shape[1] - 10, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, current_instruction, (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(display_frame, sub_instruction, (20, 53), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Compteur en bas
        cv2.putText(display_frame, f"Personne: {person_name} | Total enregistre: {count}", 
                    (20, display_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Message temporaire lors de la capture (duree : 2 secondes)
        if current_time < feedback_until:
            cv2.putText(display_frame, feedback_text, (20, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Capture Dataset Pas-a-Pas (20 Photos) - ANI-2083", display_frame)

        # Gestion des touches du clavier
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
            
        elif key == ord(' '):  # Appui sur ESPACE pour capturer
            if step_index < len(POSES_20_GUIDE):
                count += 1
                file_path = os.path.join(output_dir, f"img_{count:03d}.jpg")
                
                # Sauvegarde de la frame sans texte
                cv2.imwrite(file_path, frame)
                
                print(f"[{step_index + 1}/20] Photo enregistree : {file_path}")
                
                feedback_text = f"PHOTO {step_index + 1}/20 CAPTUREE !"
                feedback_until = current_time + 2.0  # Disparait apres 2 secondes
                
                step_index += 1
            else:
                print("\nLes 20 photos ont deja toutes ete capturees !")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n Session terminee. Dossier : {output_dir}")

if __name__ == "__main__":
    capture_person_dataset()