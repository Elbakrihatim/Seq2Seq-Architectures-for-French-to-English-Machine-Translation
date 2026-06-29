import os
import urllib.request
import zipfile
import pickle
import csv
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import spacy

# --- Vocabulary Class Definition ---
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

    def build_vocab(self, tokenized_texts, min_freq):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        idx = len(self.stoi)
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        if isinstance(token, list):
            return [self.stoi.get(t, self.unk_idx) for t in token]
        return self.stoi.get(token, self.unk_idx)

    def __call__(self, tokens):
        return self.__getitem__(tokens)

    def lookup_indices(self, tokens):
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def lookup_tokens(self, indices):
        return [self.itos.get(idx, self.unk_token) for idx in indices]

# --- Model Architecture Definitions ---
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

# --- Utility Data Processing Functions (Re-building Vocabulary) ---
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['src', 'trg'], engine='python', quoting=csv.QUOTE_NONE)
    df['src'] = df['src'].str.strip()
    df['trg'] = df['trg'].str.strip()
    return df

class TranslationDataset:
    def __init__(self, df):
        self.examples = [
            {"en": row["src"], "fr": row["trg"]}
            for _, row in df.iterrows()
        ]

def tokenize_example(example, en_nlp, fr_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
    fr_tokens = [token.text for token in fr_nlp.tokenizer(example["fr"])][:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        fr_tokens = [token.lower() for token in fr_tokens]
    example["en_tokens"] = [sos_token] + en_tokens + [eos_token]
    example["fr_tokens"] = [sos_token] + fr_tokens + [eos_token]
    return example

# --- Translation Logic ---
def translate_sentence(
    sentence,
    model,
    fr_nlp,
    fr_vocab,
    en_vocab,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            fr_tokens = [token.text for token in fr_nlp.tokenizer(sentence)]
        else:
            fr_tokens = [token for token in sentence]
            
        fr_tokens = [token.lower() for token in fr_tokens]
        fr_tokens = ["<sos>"] + fr_tokens + ["<eos>"]
        
        ids = fr_vocab.lookup_indices(fr_tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        
        encoder_outputs, hidden = model.encoder(tensor)
        
        inputs = en_vocab.lookup_indices(["<sos>"])
        attentions = torch.zeros(max_output_length, 1, len(ids))
        
        for i in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, attention_weights = model.decoder(
                inputs_tensor, hidden, encoder_outputs
            )
            attentions[i] = attention_weights
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab["<eos>"]:
                break
                
        en_tokens = en_vocab.lookup_tokens(inputs)
    return en_tokens, fr_tokens, attentions[: len(en_tokens) - 1]
