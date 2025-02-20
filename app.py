import streamlit as st  # Importation de Streamlit pour créer l'interface web
import torch  # PyTorch pour charger et exécuter le modèle SwinIR
import numpy as np  # NumPy pour la manipulation des tableaux d'images
from PIL import Image  # PIL (Pillow) pour gérer les images
from models.network_swinir import SwinIR as net  # Importation du modèle SwinIR
from utils import util_calculate_psnr_ssim as util  # Utilitaire pour calculer PSNR et SSIM
import os  # Module pour la gestion des fichiers et dossiers
import shutil  # Pour la suppression de fichiers dans les dossiers

# Titre principal affiché sur l'application web
st.title('Restaurateur d\'Image avec SwinIR')

# Sélection de la tâche (ici, nous utilisons uniquement la super-résolution "real_sr")
task = st.selectbox("Sélectionner la tâche", ["real_sr"])

# Sélection de l'échelle de mise à l'échelle (fixée à x4 pour l'instant)
scale = st.selectbox("Sélectionner l'échelle", [4])

# Chemin du modèle pré-entraîné utilisé pour la restauration d'images
model_path = st.text_input("Chemin vers le modèle", "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth")

# Upload de l'image à restaurer via l'interface web
uploaded_image = st.file_uploader("Choisissez une image à restaurer", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Chargement de l'image sous forme de tableau NumPy
    img = Image.open(uploaded_image)  # Ouvre l'image avec PIL
    img = np.array(img)  # Convertit l'image en tableau NumPy pour le traitement

    # Affichage de l'image originale avant le traitement
    st.image(img, caption="Image originale", use_column_width=True)

    # Prétraitement de l'image avant de l'envoyer au modèle
    img_lq = img.astype(np.float32) / 255.0  # Normalisation des pixels entre [0, 1]

    # Conversion des dimensions d'image pour correspondre au format attendu par le modèle
    img_lq = np.transpose(img_lq, (2, 0, 1))  # Passage de [H, W, C] à [C, H, W] (format PyTorch)

    # Conversion de l'image en un tenseur PyTorch et ajout de la dimension batch
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Définition de la fonction permettant de charger le modèle SwinIR
    def load_model(model_path, task, scale):
        """
        Charge le modèle SwinIR avec les paramètres adaptés à la tâche.
        """
        model = None  # Initialisation du modèle
        
        # Vérification de la tâche (ici uniquement "real_sr")
        if task == "real_sr":
            model = net(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                        depths=[6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                        upsampler='nearest+conv', resi_connection='1conv')

        # Chargement des poids du modèle pré-entraîné
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()  # Passage en mode évaluation (désactive l'entraînement)
        return model

    # Chargement du modèle avec les paramètres définis
    model = load_model(model_path, task, scale)

    # Application du modèle pour restaurer l'image
    with torch.no_grad():  # Désactivation du calcul de gradients pour optimiser l'inférence
        restored_image = model(img_lq)  # Prédiction du modèle

    # Conversion de la sortie du modèle en un format exploitable pour l'affichage
    restored_image = restored_image.squeeze().cpu().numpy()
    restored_image = np.transpose(restored_image, (1, 2, 0))  # Passage de [C, H, W] à [H, W, C]

    # Affichage de l'image restaurée après le traitement
    st.image(restored_image, caption="Image restaurée", use_column_width=True)

    # Calcul des métriques pour évaluer la qualité de l'image restaurée

    # Conversion de l'image d'entrée en format normalisé pour le calcul du PSNR/SSIM
    img_gt = img.astype(np.float32) / 255.0

    try:
        # Calcul du PSNR (Peak Signal-to-Noise Ratio)
        psnr = util.calculate_psnr(np.array(restored_image), img_gt, crop_border=scale)
        st.write(f"PSNR : {psnr:.2f} dB")

        # Calcul du SSIM (Structural Similarity Index)
        ssim = util.calculate_ssim(np.array(restored_image), img_gt, crop_border=scale)
        st.write(f"SSIM : {ssim:.4f}")

    except AssertionError:
        # Gestion d'erreur si la taille des images diffère (évite de casser l'interface utilisateur)
        st.warning("Les dimensions des images d'entrée et de sortie ne correspondent pas. Impossible de calculer le PSNR et le SSIM.")

# Fonction pour vider un dossier donné
def clear_folder(folder_path):
    """
    Supprime tous les fichiers dans un dossier donné.
    Utilisé pour réinitialiser les dossiers avant chaque traitement.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Vérifie que c'est bien un fichier
            os.remove(file_path)  # Supprime le fichier

# Avant chaque nouvelle opération, on vide le dossier de sortie
clear_folder("/home/melissa/Day3M/SwinIR/results/swinir_real_sr_x4")
