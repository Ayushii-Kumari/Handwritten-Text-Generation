import os
from PIL import Image

folder = "Datasets/PaperTextures"

textures = []

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert("RGB")
            textures.append(img)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

print(f"Loaded {len(textures)} notebook paper textures!")

textures[0].show()
