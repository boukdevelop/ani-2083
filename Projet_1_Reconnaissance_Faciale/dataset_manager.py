import os
import face_recognition
import cv2

def audit_dataset(base_dir="Projet_1_Reconnaissance_Faciale/data/known_faces"):
    if not os.path.exists(base_dir):
        print(f"Le dossier {base_dir} n'existe pas encore. Lance d'abord capture_dataset.py.")
        return

    print(f"\n--- AUDIT DU DATASET ({base_dir}) ---\n")
    
    person_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not person_dirs:
        print(" Aucun sous-dossier de personne trouvé.")
        return

    total_images_global = 0
    total_valid_global = 0

    for person in person_dirs:
        person_path = os.path.join(base_dir, person)
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f" Personne : {person.upper()} ({len(images)} images trouvees)")
        
        valid_count = 0
        invalid_images = []

        for img_name in images:
            img_path = os.path.join(person_path, img_name)
            
            # Chargement de l'image avec face_recognition
            image = face_recognition.load_image_file(img_path)
            
            # Détection des visages (retourne une liste de coordonnées
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) == 1:
                valid_count += 1
            else:
                invalid_images.append((img_name, len(face_locations)))

        total_images_global += len(images)
        total_valid_global += valid_count

        if invalid_images:
            print(f"   Attention : {len(invalid_images)} image(s) pose(nt) probleme :")
            for img_name, count in invalid_images:
                reason = "Aucun visage detecte" if count == 0 else f"{count} visages detectes"
                print(f"      - {img_name} ({reason})")
        else:
            print(f"   Toutes les images sont valides (1 visage par image).")
        print("-" * 40)

    print(f"\nRESUME GLOBAL :")
    print(f" - Total images analysees : {total_images_global}")
    print(f" - Images valides (exploitables) : {total_valid_global}")
    print(f" - Taux de qualite : {(total_valid_global / total_images_global * 100) if total_images_global > 0 else 0:.1f}%\n")

if __name__ == "__main__":
    audit_dataset()