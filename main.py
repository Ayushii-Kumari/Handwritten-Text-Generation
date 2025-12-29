# type: ignore
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from groq import Groq
import textwrap
import random
import os

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PAPER_PATH = r"C:\Users\KIIT0001\OneDrive\Documents\Goal\Projects\Handwritten Text Generation\Datasets\PaperTextures\1000087136.jpg"
FONT_PATH = r"c:\Users\KIIT0001\OneDrive\Documents\Goal\Projects\Handwritten Text Generation\fonts\Vibur-Regular.ttf"
OUTPUT_PATH = "final_homework_perfect.png"

client = Groq(api_key=GROQ_API_KEY)

def get_answer(question):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Answer in 5-6 sentences: {question}"}]
    )
    return response.choices[0].message.content.strip()

def create_perfect_alignment(text):
    paper = cv2.imread(PAPER_PATH)
    if paper is None: raise FileNotFoundError("Paper image not found!")
    
    paper_pil = Image.fromarray(cv2.cvtColor(paper, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(paper_pil)

    font_size = 65    
    start_x = 210     
    start_y = 358      
    line_gap = 65      
    chars_limit = 70 

    font = ImageFont.truetype(FONT_PATH, font_size)

    lines = []
    for paragraph in text.split('\n'):
        lines.extend(textwrap.wrap(paragraph.strip(), width=chars_limit))

    current_y = start_y
    for line in lines:
        x_jitter = random.randint(-3, 3)
        y_jitter = random.randint(-1, 1)

        draw.text(
            (start_x + x_jitter, current_y + y_jitter),
            line,
            fill=(28, 35, 85), 
            font=font
        )

        current_y += line_gap

    final_cv = cv2.cvtColor(np.array(paper_pil), cv2.COLOR_RGB2BGR)
 
    final_cv = cv2.GaussianBlur(final_cv, (1, 1), 0)

    cv2.imwrite(OUTPUT_PATH, final_cv)
    print(f"✅ Success! Saved perfectly aligned homework.")

    cv2.imshow("Perfect Alignment", cv2.resize(final_cv, (800, 1000)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ans = get_answer("what is dbms.")
    create_perfect_alignment(ans)