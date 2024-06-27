import os
from transformers import ElectraTokenizer, ElectraModel
from transformers import BertTokenizer, BertModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_bert():
    tokenizer_path = './bert-base-uncased-tokenizer'
    model_path = './bert-base-uncased-model'
    
    # Check if the tokenizer and model directories exist
    if not os.path.isdir(tokenizer_path):
        print("Tokenizer not found locally. Downloading...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    if not os.path.isdir(model_path):
        print("Model not found locally. Downloading...")
        model = BertModel.from_pretrained('bert-base-uncased')
        model.save_pretrained(model_path)
    else:
        model = BertModel.from_pretrained(model_path).to(device)
    return tokenizer, model

def load_electra():
    tokenizer_path = './electra-small-tokenizerr'
    model_path = './electra-small-model'
    
    # Check if the tokenizer and model directories exist
    if not os.path.isdir(tokenizer_path):
        print("Tokenizer not found locally. Downloading...")
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_path)
    
    if not os.path.isdir(model_path):
        print("Model not found locally. Downloading...")
        model = ElectraModel.from_pretrained('google/electra-small-discriminator')
        model.save_pretrained(model_path)
    else:
        model = ElectraModel.from_pretrained(model_path).to(device)
    return tokenizer, model