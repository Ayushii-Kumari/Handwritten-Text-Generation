# type: ignore
import torch
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig
from PIL import Image
import cv2
import numpy as np
import os
from groq import Groq
from safetensors.torch import load_file

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def get_chatbot_answer(question: str) -> str:
    print("Asking Chatbot...")
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Give a concise 3-sentence summary: {question}"}],
    )
    return completion.choices[0].message.content

device = "cpu" 
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32
).to(device)

lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["to_q", "to_v"], lora_dropout=0.05
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

LOADED_LORA_PATH = r"C:\Users\KIIT0001\OneDrive\Documents\Goal\Projects\Handwritten Text Generation\handwriting_lora\adapter_model.safetensors"
state_dict = load_file(LOADED_LORA_PATH, device=device)
pipe.unet.load_state_dict(state_dict, strict=False)

def text_to_handwriting(text: str) -> Image.Image:
    sentences = [s.strip() for s in text.split('.') if len(s) > 5]
    line_images = []
    
    print(f"Generating {len(sentences)} lines of handwriting...")
    for i, line in enumerate(sentences):
        prompt = f"handwritten text: {line}"
        img = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
        line_images.append(np.array(img))

    combined_array = np.vstack(line_images)
    return Image.fromarray(combined_array)

NOTEBOOK_PATH = r"C:\Users\KIIT0001\OneDrive\Documents\Goal\Projects\Handwritten Text Generation\Datasets\PaperTextures\1000087134.jpg"

def overlay_on_notebook(handwriting_img: Image.Image, x=150, y=250):
    bg = cv2.imread(NOTEBOOK_PATH)
    if bg is None: raise FileNotFoundError("Paper texture not found!")

    hw = np.array(handwriting_img.convert("RGB"))
    hw = cv2.cvtColor(hw, cv2.COLOR_RGB2BGR)

    h, w = hw.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    scale = (bg_w - 2*x) / w
    new_w, new_h = int(w * scale), int(h * scale)
    hw = cv2.resize(hw, (new_w, new_h))

    roi = bg[y:y+new_h, x:x+new_w]
    blended = cv2.multiply(roi.astype(float), hw.astype(float) / 255.0)
    bg[y:y+new_h, x:x+new_w] = blended.astype(np.uint8)

    cv2.imwrite("final_output.png", bg)
    return bg

if __name__ == "__main__":
    question = "Explain Newton's first law of motion."

    answer = get_chatbot_answer(question)
    print(f"Chatbot says: {answer}")
    
    hw_img = text_to_handwriting(answer)
    final_result = overlay_on_notebook(hw_img)
    
    print("Success! Result saved as final_output.png")
    cv2.imshow("Final Homework", final_result)
    cv2.waitKey(0)