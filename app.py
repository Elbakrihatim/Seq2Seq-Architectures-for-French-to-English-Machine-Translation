import os
import urllib.request
import zipfile
import pickle
import torch
import spacy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

# Import from src.model_utils
from src.model_utils import (
    Vocabulary,
    Encoder,
    Attention,
    Decoder,
    Seq2Seq,
    load_and_clean_data,
    TranslationDataset,
    tokenize_example,
    translate_sentence
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Seq2Seq Translation Model by Hatim EL BAKRI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Styling for primary buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    
    /* Translation Output Panel Card style */
    .translation-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .translation-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #a0aec0;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .translation-text {
        font-size: 1.5rem;
        color: #e2e8f0;
        font-weight: 500;
        line-height: 1.4;
    }
    
    /* Subtle headers */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Cached Asset Loading to prevent re-processing ---
@st.cache_resource
def load_nlp_and_vocab():
    vocab_path = "models/vocab.pkl"
    en_nlp = spacy.load("en_core_web_sm")
    fr_nlp = spacy.load("fr_core_news_sm")

    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            data = pickle.load(f)
            return fr_nlp, en_nlp, data["fr_vocab"], data["en_vocab"]
    
    # Rebuild vocab dynamically
    data_file = "data/eng-fra.txt"
    if not os.path.exists(data_file):
        with st.spinner("Downloading Tatoeba dataset for vocabulary reconstruction..."):
            url = "https://download.pytorch.org/tutorial/data.zip"
            urllib.request.urlretrieve(url, "data.zip")
            with zipfile.ZipFile("data.zip", "r") as zip_ref:
                zip_ref.extractall(".")
                
    df = load_and_clean_data(data_file)
    train_df = df.sample(frac=0.8, random_state=42)
    train_data = TranslationDataset(train_df)
    
    fn_kwargs = {
        "en_nlp": en_nlp,
        "fr_nlp": fr_nlp,
        "max_length": 1000,
        "lower": True,
        "sos_token": "<sos>",
        "eos_token": "<eos>"
    }
    
    train_data.examples = [tokenize_example(ex, **fn_kwargs) for ex in train_data.examples]
    
    special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
    all_en_tokens = [ex["en_tokens"] for ex in train_data.examples]
    all_fr_tokens = [ex["fr_tokens"] for ex in train_data.examples]
    
    en_vocab = Vocabulary(special_tokens=special_tokens)
    en_vocab.build_vocab(all_en_tokens, min_freq=2)
    
    fr_vocab = Vocabulary(special_tokens=special_tokens)
    fr_vocab.build_vocab(all_fr_tokens, min_freq=2)
    
    with open(vocab_path, "wb") as f:
        pickle.dump({"fr_vocab": fr_vocab, "en_vocab": en_vocab}, f)
        
    return fr_nlp, en_nlp, fr_vocab, en_vocab

@st.cache_resource
def load_nmt_model(_fr_vocab, _en_vocab, checkpoint_path="models/best-model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = Encoder(len(_fr_vocab), 256, 512, 512, 0.5)
    attention = Attention(512, 512)
    decoder = Decoder(len(_en_vocab), 256, 512, 512, 0.5, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Convert fp16 weights back to float32 if on CPU, since CPU GRU doesn't support fp16 operations
        for k, v in state_dict.items():
            if torch.is_tensor(v) and v.dtype == torch.float16:
                state_dict[k] = v.float()
        model.load_state_dict(state_dict)
        model.eval()
        return model, device
    else:
        st.error(f"Checkpoint file '{checkpoint_path}' not found! Please place it in the same directory.")
        return None, device

# --- Load Assets ---
try:
    fr_nlp, en_nlp, fr_vocab, en_vocab = load_nlp_and_vocab()
    model, device = load_nmt_model(fr_vocab, en_vocab)
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# --- Sidebar Controls & Stats ---
with st.sidebar:
    st.title("Settings & Stats")
    st.markdown("---")
    
    # User settings
    max_len = st.slider("Max Output Tokens", min_value=10, max_value=50, value=25, step=1)
    
    st.markdown("### Model Properties")
    st.info("""
    - **Architecture**: Bi-GRU Encoder + Bahdanau Attention
    - **Parameters**: 26.5M trainable
    - **Test Loss**: 1.636
    - **BLEU-4**: ~53.11%
    - **Source Vocabulary**: 12,898 tokens
    - **Target Vocabulary**: 8,219 tokens
    """)

# --- Layout: Main Banner ---
st.markdown("<h1 style='text-align: center; color: #e2e8f0; margin-bottom: 5px;'>🤖 Seq2Seq Machine Translation by Hatim EL BAKRI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0; font-size:1.15rem; margin-bottom: 25px;'>Translate sentences from French to English using a Bidirectional GRU Encoder and Additive Attention</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🇫🇷 Input French Sentence")
    
    # Preset sentences selector for quick testing
    presets = [
        "Il fait beau aujourd'hui.",
        "Je veux une pomme.",
        "Le vent était tellement fort que nous avons presque été poussés en dehors de la route.",
        "À l'aide !"
    ]
    selected_preset = st.selectbox("Or choose a sample sentence:", ["-- Select Preset --"] + presets)
    
    default_text = ""
    if selected_preset != "-- Select Preset --":
        default_text = selected_preset
        
    input_text = st.text_area("Type French text here:", value=default_text, height=120, placeholder="e.g. Je mange la pomme.")
    
    translate_clicked = st.button("Translate 🚀")

with col2:
    st.markdown("### 🇺🇸 English Translation")
    
    if translate_clicked or (selected_preset != "-- Select Preset --" and input_text):
        if input_text.strip():
            with st.spinner("Decoding tokens..."):
                en_tokens, fr_tokens, attention = translate_sentence(
                    input_text,
                    model,
                    fr_nlp,
                    fr_vocab,
                    en_vocab,
                    device,
                    max_output_length=max_len
                )
                
                # Format final text (excluding special tokens)
                clean_tokens = [t for t in en_tokens if t not in ["<sos>", "<eos>", "<pad>"]]
                translation_str = " ".join(clean_tokens)
                
                # Render Translation Box
                st.markdown(f"""
                <div class="translation-card">
                    <div class="translation-header">Translation Result</div>
                    <div class="translation-text">{translation_str}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Plot Attention Heatmap
                st.markdown("### 📊 Attention Weight Alignment")
                fig, ax = plt.subplots(figsize=(7, 7))
                
                # Prepare data
                attention_data = attention.squeeze(1).cpu().numpy()
                cax = ax.matshow(attention_data, cmap="viridis")
                
                # Set axes ticks & labels
                ax.set_xticks(np.arange(len(fr_tokens)))
                ax.set_yticks(np.arange(len(en_tokens) - 1))
                
                ax.set_xticklabels(fr_tokens, rotation=45, ha="left", fontsize=9)
                ax.set_yticklabels(en_tokens[1:], fontsize=9) # Skip <sos> for y labels
                
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                
                fig.colorbar(cax, shrink=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                
        else:
            st.warning("Please enter a sentence or select a preset to translate.")
    else:
        st.info("Translation result and attention alignment graph will appear here.")