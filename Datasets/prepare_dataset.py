# type: ignore
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset_root = r"C:\Users\KIIT0001\.cache\kagglehub\datasets\naderabdalghani\iam-handwritten-forms-dataset\versions\1"
dataset = HandwritingDataset(root_dir=dataset_root, transform=transform)

print(f"Total training samples found: {len(dataset)}")

loader = DataLoader(dataset, batch_size=1, shuffle=True)

for i, batch in enumerate(loader):
    print(f"Batch {i} shape: {batch.shape}")
    if i >= 2:  
        break
