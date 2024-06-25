from transformers import ElectraTokenizer, ElectraModel
from transformers import BertTokenizer, BertModel
from json_to_schema_description import *
import time
def sliding_window_chunks(text, window_size = 512, stride = 128):
  chunks = []
  words = text.split()
  size = len(text)
  for i in range(0, size, stride):
    start = i
    end = i + window_size
    if end > len(words):
      end = len(words)
      start = end - stride
    chunk = ' '.join(words[start:end])
    chunks.append(chunk)
  return chunks

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_bert():
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased-tokenizer')
    model = BertModel.from_pretrained('./bert-base-uncased-model').to(device)
    return tokenizer, model

def load_electra():
    tokenizer = ElectraTokenizer.from_pretrained('./electra-small-tokenizer')
    model = ElectraModel.from_pretrained('./electra-small-model').to(device)
    return tokenizer, model

def embed_text_with_bert(text):
    start_time = time.time()
    bert_tokenizer, bert_model = load_bert()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Load bert: {elapsed_time} seconds") 

    start_time = time.time()
    chunks = sliding_window_chunks(text)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Create chunks: {elapsed_time} seconds") 
    
    start_time = time.time()
    inputs = bert_tokenizer(chunks, return_tensors="pt", padding=True, truncation=True).to(device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tokenise chunks: {elapsed_time} seconds")

    start_time = time.time()
    with torch.no_grad():
        outputs = bert_model(**inputs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"embed chunks: {elapsed_time} seconds")

    start_time = time.time()
    embeddings = outputs.last_hidden_state[:, 0, :]
    result =  (embeddings.mean(dim=0)).detach().cpu().numpy()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Result chunks: {elapsed_time} seconds")
    return result

def embed_text_with_electra(text):
    electra_tokenizer, electra_model = load_electra()
    chunks = sliding_window_chunks(text)
    inputs = [electra_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True) for chunk in chunks]
    outputs = [electra_model(**inputs[i]).last_hidden_state[:, 0, :] for i in range(len(inputs))]
    return 

def transformer_feature_extractor(json_objects, model='bert'):
    texts = [json_to_schema_description(json_object) for json_object in json_objects]
    if model=='bert':
        return [embed_text_with_bert(text) for text in texts]
    elif model=='electra':
       return [embed_text_with_bert(text) for text in texts]
