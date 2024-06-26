from transformers import ElectraTokenizer, ElectraModel
from transformers import BertTokenizer, BertModel
from load_models import *
from json_to_schema_description import *
import re
def sliding_window_chunks(text, window_size = 512, stride = 128):
  chunks = []
  words = re.findall(r'\S+|\s+', text)
  size = len(words)
  for i in range(0, size, stride):
    start = i
    end = i + window_size
    if end > len(words):
      end = len(words)
      start = end - stride
    chunk = ''.join(words[start:end])
    chunks.append(chunk)
  return chunks

def embed_text_with_bert(text):
    bert_tokenizer, bert_model = load_bert()
    chunks = sliding_window_chunks(text)
    inputs = bert_tokenizer(chunks, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    result =  (embeddings.mean(dim=0)).detach().cpu().numpy()
    return result

def embed_text_with_electra(text):
    electra_tokenizer, electra_model = load_electra()
    chunks = sliding_window_chunks(text)
    inputs = electra_tokenizer(chunks, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = electra_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    result =  (embeddings.mean(dim=0)).detach().cpu().numpy()
    return result

def transformer_feature_extractor(json_objects, model='bert'):
    texts = [json_to_schema_description(json_object) for json_object in json_objects]
    if model=='bert':
        return [embed_text_with_bert(text) for text in texts]
    elif model=='electra':
       return [embed_text_with_electra(text) for text in texts]
