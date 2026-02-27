import gdown
import os

def telecharger_modele():
    if not os.path.exists("modeles/nexa_biomasse.onnx"):
        os.makedirs("modeles", exist_ok=True)
        print("📥 Téléchargement du modèle depuis Google Drive...")
        gdown.download(
            "https://drive.google.com/uc?id=COLLE_TON_ID_ICI",
            "modeles/nexa_biomasse.onnx",
            quiet=False
        )
        print("✅ Modèle téléchargé")