# Fine tuning done on Colab with GPU runtime :- https://colab.research.google.com/drive/1u6Lc_Cv9S4Nzdv03CZ-xjENc4hp6koq8#scrollTo=rD4r7Yn0c_DX
# type: ignore

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model

MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATASET_DIR = "Datasets/resized_images_dataset"  
CAPTION = "handwritten text"
OUTPUT_DIR = "handwriting_lora"

EPOCHS = 1
BATCH_SIZE = 1
LR = 1e-4

device = torch.device("cpu")
dtype = torch.float32

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using device:", device)

class HandwritingDataset(Dataset):
    def __init__(self, folder):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if len(self.files) == 0:
            raise ValueError("❌ Dataset folder is empty")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        image = self.transform(image)
        return image

dataset = HandwritingDataset(DATASET_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    safety_checker=None,
    dtype=dtype
).to(device)

vae = pipe.vae
unet = pipe.unet
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device)

scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.05,
    bias="none"
)


unet = get_peft_model(unet, lora_config)
unet.train()

optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)

print("Starting fine-tuning...")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")

    for images in tqdm(dataloader):
        images = images.to(device)

        latents = vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device
        ).long()

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        tokens = tokenizer(
            [CAPTION],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        with torch.no_grad():
            encoder_hidden_states = text_encoder(
                tokens.input_ids.to(device)
            ).last_hidden_state  
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Loss:", loss.item())

unet.save_pretrained(OUTPUT_DIR)
print("✅ LoRA fine-tuning complete")
print(f"Saved to: {OUTPUT_DIR}")
print("You can now use the fine-tuned LoRA weights for handwriting generation!")