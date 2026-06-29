import os
import torch
from src.model_utils import (
    Vocabulary,
    Encoder,
    Attention,
    Decoder,
    Seq2Seq,
    load_and_clean_data,
    TranslationDataset,
    tokenize_example,
)
import pickle
import urllib.request
import zipfile
import spacy

def get_or_build_vocab():
    vocab_path = "models/vocab.pkl"
    if os.path.exists(vocab_path):
        print("Loading saved vocabulary...")
        with open(vocab_path, "rb") as f:
            data = pickle.load(f)
            return data["fr_vocab"], data["en_vocab"]
    
    print("Vocabulary pkl not found. Rebuilding from raw dataset...")
    data_file = "data/eng-fra.txt"
    if not os.path.exists(data_file):
        print("Downloading dataset data.zip...")
        url = "https://download.pytorch.org/tutorial/data.zip"
        urllib.request.urlretrieve(url, "data.zip")
        print("Unzipping data.zip...")
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall(".")
    
    print("Loading SpaCy models...")
    en_nlp = spacy.load("en_core_web_sm")
    fr_nlp = spacy.load("fr_core_news_sm")
    
    print("Cleaning dataset and building data split...")
    df = load_and_clean_data(data_file)
    train_df = df.sample(frac=0.8, random_state=42)
    train_data = TranslationDataset(train_df)
    
    print("Tokenizing train dataset examples...")
    fn_kwargs = {
        "en_nlp": en_nlp,
        "fr_nlp": fr_nlp,
        "max_length": 1000,
        "lower": True,
        "sos_token": "<sos>",
        "eos_token": "<eos>"
    }
    
    train_data.examples = [tokenize_example(ex, **fn_kwargs) for ex in train_data.examples]
    
    print("Assembling vocabulary...")
    special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]
    all_en_tokens = [ex["en_tokens"] for ex in train_data.examples]
    all_fr_tokens = [ex["fr_tokens"] for ex in train_data.examples]
    
    en_vocab = Vocabulary(special_tokens=special_tokens)
    en_vocab.build_vocab(all_en_tokens, min_freq=2)
    
    fr_vocab = Vocabulary(special_tokens=special_tokens)
    fr_vocab.build_vocab(all_fr_tokens, min_freq=2)
    
    print(f"Vocabularies built: French={len(fr_vocab)}, English={len(en_vocab)}")
    
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump({"fr_vocab": fr_vocab, "en_vocab": en_vocab}, f)
    print(f"Saved vocabulary to {vocab_path}.")
    return fr_vocab, en_vocab

if __name__ == "__main__":
    fr_vocab, en_vocab = get_or_build_vocab()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing model...")
    encoder = Encoder(len(fr_vocab), 256, 512, 512, 0.5)
    attention = Attention(512, 512)
    decoder = Decoder(len(en_vocab), 256, 512, 512, 0.5, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    checkpoint_path = "models/best-model.pt"
    print(f"Loading state dict from {checkpoint_path}...")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Convert fp16 weights back to float32 if on CPU, since CPU GRU doesn't support fp16 operations
        for k, v in state_dict.items():
            if torch.is_tensor(v) and v.dtype == torch.float16:
                state_dict[k] = v.float()
        model.load_state_dict(state_dict)
        model.eval()
        print("Model successfully loaded and ready!")
    else:
        print(f"Error: {checkpoint_path} not found.")
