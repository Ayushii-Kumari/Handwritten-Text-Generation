import os
from PIL import Image

input_folder = r"C:\Users\KIIT0001\.cache\kagglehub\datasets\naderabdalghani\iam-handwritten-forms-dataset\versions\1"
output_folder = r"C:\Users\KIIT0001\OneDrive\Documents\Goal\Projects\Handwritten Text Generation\Datasets\resized_images"
os.makedirs(output_folder, exist_ok=True)
image_size = (128, 128)

for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print("Processing:", filename)
            img_path = os.path.join(root, filename)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_folder, filename))

print("Resizing done.")
