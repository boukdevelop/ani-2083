import customtkinter as ctk
from PIL import Image, ImageTk

# Configuration initiale de CustomTkinter
ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")

class AppInterface(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Reconnaissance Faciale - Projet 1")
        self.geometry("1000x700")
        self.minsize(800, 600)

        # Grille principale (Layout)
        self.grid_columnconfigure(0, weight=4)  # Zone vidéo
        self.grid_columnconfigure(1, weight=1)  # Panneau de contrôle
        self.grid_rowconfigure(0, weight=1)

        # --- Zone d'affichage du flux vidéo (Gauche) ---
        self.video_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="black")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Flux vidéo en attente...", text_color="white")
        self.video_label.pack(expand=True)

        # --- Panneau de contrôle / Sidebar (Droite) ---
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=10)
        self.sidebar_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.sidebar_frame, text="Panneau de Contrôle", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        # Bouton Capture d'écran
        self.btn_screenshot = ctk.CTkButton(self.sidebar_frame, text="📸 Capturer l'écran", command=self.take_screenshot)
        self.btn_screenshot.pack(padx=20, pady=10, fill="x")

        # Bouton Quitter
        self.btn_quit = ctk.CTkButton(self.sidebar_frame, text="❌ Quitter", fg_color="red", hover_color="darkred", command=self.quit)
        self.btn_quit.pack(padx=20, pady=10, fill="x")

        # Informations / Statuts
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Statut : En cours...", font=ctk.CTkFont(size=12))
        self.status_label.pack(padx=20, side="bottom", pady=20)

        self.current_frame = None

    def update_video_feed(self, frame):
        """Met à jour l'image affichée dans l'interface à partir d'une frame OpenCV (RGB)"""
        self.current_frame = frame
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo  # Garder une référence pour éviter le garbage collection

    def take_screenshot(self):
        """Action déclenchée lors du clic sur le bouton de capture d'écran"""
        if self.current_frame is not None:
            image_rgb = Image.fromarray(self.current_frame)
            filename = "capture_ecran.png"
            image_rgb.save(filename)
            self.status_label.configure(text=f"Sauvegardé : {filename}")
        else:
            self.status_label.configure(text="Erreur : Pas de flux vidéo")