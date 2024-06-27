from predictors.pretrained_models.load_models import *
from predictors.json_to_schema_description import *
import re
from sklearn.metrics.pairwise import cosine_similarity

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

def load_encoded_features(file_path):
    with open(file_path, 'rb') as f:
        embeddings = np.load(f, allow_pickle=True)
    return embeddings
def get_most_similar(json_objects, query_json, feature_extraction_func, embedings_path=None):
    if embedings_path:
        encoded_features = load_encoded_features(embedings_path)
    else:
        encoded_features = feature_extraction_func(json_objects)
    query_features = feature_extraction_func([query_json])[0]
    similarity_scores = cosine_similarity([query_features], encoded_features)
    most_similar_index = np.argmax(similarity_scores)
    most_similar_json = json_objects[most_similar_index]
    return most_similar_json  

def save_encoded_features(json_objects, feature_extraction_func, embedings_path):
    features = feature_extraction_func(json_objects)
    with open(embedings_path, 'wb') as f:
        np.save(f, features)