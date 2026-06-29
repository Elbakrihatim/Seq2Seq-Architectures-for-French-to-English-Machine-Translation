# Deployment Guide for Bidirectional GRU NMT Model

This document outlines the best strategies, code implementations, and steps to deploy your **Bidirectional GRU with Bahdanau Attention** French-to-English translation model.

Depending on your objective, the two best ways to deploy this model are:
1. **Gradio Web Demo (Recommended for Portfolios)**: A visually appealing, interactive web interface where users can enter French text, get translations, and inspect the attention weights. Great for showing off your work.
2. **FastAPI REST API (Recommended for Web Integration)**: A lightweight, high-performance API endpoint that can be consumed by frontend applications, mobile apps, or external services.

---

## 📋 Table of Contents
1. [Step 1: Exporting Model Weights and Vocabularies](#step-1-exporting-model-weights-and-vocabularies)
2. [Step 2: Interactive Web Demo (Gradio)](#step-2-interactive-web-demo-gradio)
3. [Step 3: Lightweight REST API (FastAPI)](#step-3-lightweight-rest-api-fastapi)
4. [Step 4: Containerization (Docker)](#step-4-containerization-docker)
5. [Step 5: Where to Host (The Best Options)](#step-5-where-to-host-the-best-options)

---

## 1. Exporting Model Weights and Vocabularies

To run inference in a clean production environment without running the whole Jupyter notebook, you must export two assets from your training environment:
1. **Model State Dictionary (`tut3-model.pt`)**
2. **Vocabulary mappings (`fr_vocab` and `en_vocab`)**

Run this snippet at the end of your Jupyter Notebook to save both components:

```python
import pickle
import torch

# 1. Save the model weights (ensure you load the best epoch first)
# model.load_state_dict(torch.load("tut3-model.pt", map_location="cpu"))
torch.save(model.state_dict(), "best_bi_gru_model.pt")

# 2. Save the vocabularies using pickle
vocab_data = {
    "fr_vocab": fr_vocab,
    "en_vocab": en_vocab
}

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab_data, f)

print("Assets successfully exported for deployment!")
```

---

## 2. Interactive Web Demo (Gradio)

For portfolio projects, presenting a live UI is the best way to get noticed. **Gradio** allows you to build a translation interface in pure Python in under 50 lines of code, complete with text boxes, translation outputs, and even attention weight visualizations.

### `app_gradio.py`
```python
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Vocabularies & Define Vocabulary Class ---
# Re-define the Vocabulary class so pickle can reconstruct it
class Vocabulary:
    def __init__(self, special_tokens, unk_token="<unk>"):
        self.special_tokens = special_tokens
        self.unk_token = unk_token
        self.stoi = {}
        self.itos = {}
        for idx, token in enumerate(self.special_tokens):
            self.stoi[token] = idx
            self.itos[idx] = token
        self.unk_idx = self.stoi[self.unk_token]

    def lookup_indices(self, tokens):
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def lookup_tokens(self, indices):
        return [self.itos.get(idx, self.unk_token) for idx in indices]

    def __len__(self):
        return len(self.stoi)

with open("vocab.pkl", "rb") as f:
    vocabs = pickle.load(f)
fr_vocab = vocabs["fr_vocab"]
en_vocab = vocabs["en_vocab"]

# --- 2. Define Model Architectures ---
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_length = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v_fc(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((encoder_hidden_dim * 2) + embedding_dim, decoder_hidden_dim)
        self.fc_out = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim + embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

# --- 3. Initialize & Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(len(fr_vocab), 256, 512, 512, 0.5)
attention = Attention(512, 512)
decoder = Decoder(len(en_vocab), 256, 512, 512, 0.5, attention)
model = Seq2Seq(encoder, decoder, device).to(device)

model.load_state_dict(torch.load("best_bi_gru_model.pt", map_location=device))
model.eval()

# --- 4. Load SpaCy Tokenizer ---
try:
    fr_nlp = spacy.load("fr_core_news_sm")
except OSError:
    import os
    os.system("python -m spacy download fr_core_news_sm")
    fr_nlp = spacy.load("fr_core_news_sm")

# --- 5. Translation Logic & Plotting ---
def translate(sentence):
    tokens = [token.text.lower() for token in fr_nlp.tokenizer(sentence)]
    tokens = ["<sos>"] + tokens + ["<eos>"]
    
    ids = fr_vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(tensor)
        
        inputs = en_vocab.lookup_indices(["<sos>"])
        attentions = torch.zeros(25, 1, len(ids))
        
        for i in range(25):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, attention_weights = model.decoder(inputs_tensor, hidden, encoder_outputs)
            attentions[i] = attention_weights
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab["<eos>"]:
                break
                
        en_tokens = en_vocab.lookup_tokens(inputs)
    
    # Format target sentence (excluding special tokens)
    clean_translation = " ".join([t for t in en_tokens if t not in ["<sos>", "<eos>", "<pad>"]])
    
    # Generate Attention Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    attn_data = attentions[:len(en_tokens)-1].squeeze(1).cpu().numpy()
    
    ax.matshow(attn_data, cmap="viridis")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="left")
    ax.set_yticks(np.arange(len(en_tokens)-1))
    ax.set_yticklabels(en_tokens[1:])
    plt.tight_layout()
    
    return clean_translation, fig

# --- 6. Gradio Interface Layout ---
demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(label="French Source Text", placeholder="Enter a French sentence... e.g. Il fait beau aujourd'hui."),
    outputs=[
        gr.Textbox(label="English Translation Output"),
        gr.Plot(label="Attention Alignment Map")
    ],
    title="French-to-English Neural Machine Translation (Bi-GRU)",
    description="An interactive demo of the Bidirectional GRU model with Bahdanau Additive Attention, mapping source token importance dynamics.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 3. Lightweight REST API (FastAPI)

If you are developing a web app (e.g., using Next.js or React) and need a back-end translation service, **FastAPI** is the best option because it has minimal overhead, operates asynchronously, and automatically validates input.

### `app_api.py`
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import torch
import torch.nn as nn
import spacy

class Vocabulary:
    # Redefined to allow unpickling
    def __init__(self, special_tokens, unk_token="<unk>"):
        self.special_tokens = special_tokens
        self.unk_token = unk_token
        self.stoi = {}
        self.itos = {}
        for idx, token in enumerate(self.special_tokens):
            self.stoi[token] = idx
            self.itos[idx] = token
        self.unk_idx = self.stoi[self.unk_token]

    def lookup_indices(self, tokens):
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def lookup_tokens(self, indices):
        return [self.itos.get(idx, self.unk_token) for idx in indices]

# Model definitions (Encoder, Attention, Decoder, Seq2Seq)
# [Insert the same model definitions from the Gradio script here]

app = FastAPI(title="Bi-GRU Translation API", version="1.0.0")

# Global variables loaded at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
fr_vocab = None
en_vocab = None
fr_nlp = None

@app.on_event("startup")
def load_assets():
    global model, fr_vocab, en_vocab, fr_nlp
    # Load Vocabs
    with open("vocab.pkl", "rb") as f:
        vocabs = pickle.load(f)
    fr_vocab = vocabs["fr_vocab"]
    en_vocab = vocabs["en_vocab"]
    
    # Load SpaCy
    fr_nlp = spacy.load("fr_core_news_sm")
    
    # Initialize Model
    encoder = Encoder(len(fr_vocab), 256, 512, 512, 0.5)
    attention = Attention(512, 512)
    decoder = Decoder(len(en_vocab), 256, 512, 512, 0.5, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load("best_bi_gru_model.pt", map_location=device))
    model.eval()

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translation: str
    tokens: list

@app.post("/translate", response_model=TranslationResponse)
def translate_endpoint(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    tokens = [token.text.lower() for token in fr_nlp.tokenizer(request.text)]
    tokens = ["<sos>"] + tokens + ["<eos>"]
    
    ids = fr_vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(tensor)
        inputs = en_vocab.lookup_indices(["<sos>"])
        
        for _ in range(30):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, _ = model.decoder(inputs_tensor, hidden, encoder_outputs)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab["<eos>"]:
                break
                
        en_tokens = en_vocab.lookup_tokens(inputs)
    
    clean_translation = " ".join([t for t in en_tokens if t not in ["<sos>", "<eos>", "<pad>"]])
    return TranslationResponse(translation=clean_translation, tokens=en_tokens)
```

To run this server locally:
```bash
pip install fastapi uvicorn pydantic spacy torch
python -m spacy download fr_core_news_sm
uvicorn app_api:app --host 0.0.0.0 --port 8000
```
You can access your api swagger docs at `http://localhost:8000/docs`.

---

## 4. Containerization (Docker)

To deploy your application seamlessly without worrying about PyTorch and SpaCy version issues on the target server, compile your app into a **Docker Container**.

Here is a ready-to-use production `Dockerfile` suitable for both the **Gradio UI** and the **FastAPI REST API**.

### `Dockerfile`
```dockerfile
# Use a lightweight official PyTorch base image (CPU-only keeps size down to ~2GB)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy pipeline
RUN python -m spacy download fr_core_news_sm

# Copy app code and exported model artifacts
COPY best_bi_gru_model.pt .
COPY vocab.pkl .
COPY app_gradio.py .
COPY app_api.py .

# Expose port (Gradio uses 7860, FastAPI uses 8000)
EXPOSE 7860

# Run the Gradio demo by default
CMD ["python", "app_gradio.py"]
```

### `requirements.txt`
```text
gradio>=4.0.0
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0
spacy>=3.6.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## 5. Where to Host (The Best Options)

For standard deep learning portfolio apps, CPU-based instances are usually sufficient, keeping the cost at **$0** or very minimal.

| Platform | Best For | Difficulty | Price | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Hugging Face Spaces** | Gradio Web UI Demo | **Very Easy** | **Free** | Integrates directly with Git. Just upload your files (`app_gradio.py`, `requirements.txt`, `best_bi_gru_model.pt`, `vocab.pkl`) to a space, and it builds and hosts the Gradio app instantly. |
| **Render** or **Railway** | FastAPI REST API | **Easy** | **Free / Cheap** ($7/mo) | Deploy from your Github repository. You can use the Dockerfile to set it up. Auto-restarts on code commits. |
| **AWS ECS/App Runner** | Enterprise REST API | **Medium** | Pay-as-you-go | Best if you need standard AWS integration, autoscaling, and dedicated network rules. |
