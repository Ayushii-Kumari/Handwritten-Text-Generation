# Handwritten Text Generation

This project generates **handwritten-style text** from chatbot responses and overlays it on paper texture images. It uses **Stable Diffusion**, **LoRA fine-tuning**, and a **chatbot API** to produce realistic handwritten output, ideal for automated homework sheets, notes, or handwritten documents.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Environment & Security](#environment--security)
- [Git & GitHub Setup](#git--github-setup)
- [Contributing](#contributing)
- [Future Improvements](#future-improvements)

---

## Project Overview
- Converts chatbot answers into handwritten images
- Overlays generated handwriting on custom notebook/paper textures
- Supports CPU-only execution
- Fine-tuned LoRA model for handwriting style
- Handles multi-line text and automatic line wrapping

---

## Features
- Realistic handwritten text generation
- Chatbot integration for automatic content
- Customizable notebook/paper backgrounds
- LoRA fine-tuning for personalized handwriting style
- Multi-line and paragraph support

---

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/Ayushii-Kumari/Handwritten-Text-Generation.git
cd Handwritten-Text-Generation
````

2. **Create and activate a virtual environment**

```bash
python -m venv myenv
myenv\Scripts\activate       # Windows
source myenv/bin/activate    # macOS/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Setup API Keys securely**

* **Do not hardcode API keys** in scripts. Use environment variables or a `.env` file.

```env
GROQ_API_KEY=your_groq_api_key_here
```

* Load in code:

```python
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

5. **Place LoRA model**

```
handwriting_lora/adapter_model.safetensors
```

6. **Add notebook/paper textures**

```
Datasets/PaperTextures/
```

---

## Usage

Run the main script:

```bash
python main.py
```

Steps:

1. Queries chatbot for response
2. Converts text to handwriting
3. Overlays on paper texture
4. Saves output as `final_output.png`

---

## Project Structure

```
Handwritten-Text-Generation/
│
├─ main.py                  # Main script
├─ generate_handwritten.py  # Handwriting generation functions
├─ fine_tuning.py           # LoRA fine-tuning script
├─ requirements.txt         # Dependencies
├─ README.md                # This file
├─ handwriting_lora/        # LoRA model folder
│   └─ adapter_model.safetensors
├─ Datasets/
│   └─ PaperTextures/       # Notebook images
├─ fonts/                   # Optional fonts for rendering
└─ myenv/                   # Python virtual environment
```

---

## Environment & Security

* **Do not commit API keys**: use environment variables or `.env`.
* **Use `.gitignore`** to exclude sensitive files:

```
myenv/
*.env
*.safetensors
```

* GitHub push protection will block commits containing secrets.

---

## Git & GitHub Setup

1. **Initialize git and add remote**

```bash
git init
git remote add origin https://github.com/Ayushii-Kumari/Handwritten-Text-Generation.git
```

2. **Stage and commit files**

```bash
git add .
git commit -m "Initial commit without secrets"
```

3. **Push to main branch**

```bash
git branch -M main
git push -u origin main
```

> If push fails because remote has content, first pull or force push carefully:

```bash
git pull origin main --rebase
git push -u origin main
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Describe your changes"`
4. Push branch: `git push origin feature-name`
5. Open a Pull Request

> Always avoid committing secrets.

---

## Future Improvements

* Support multiple handwriting styles
* GUI for text input and style selection
* PDF output for generated pages
* Multi-language support
* Enhanced AI chatbot integration

```
