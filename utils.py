import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_json(file_path):
    import json
    with open(file_path, 'r') as file:
        return json.load(file)
def load_encoded_features(file_path):
    with open(file_path, 'rb') as f:
        embeddings = np.load(f, allow_pickle=True).item()
    return embeddings
def get_most_similar(json_objects, query_json, feature_extraction_func, embedings_path, encoder=None):
    encoded_features = load_encoded_features(embedings_path)
    query_features = feature_extraction_func([query_json])
    encoded_query = encoder(query_features)
    similarity_scores = cosine_similarity(encoded_query, encoded_features)
    most_similar_index = np.argmax(similarity_scores)
    most_similar_json = json_objects[most_similar_index]
    return most_similar_json  

def save_encoded_features(json_objects, feature_extraction_func, embedings_path, encoder=None):
    features = feature_extraction_func(json_objects)
    if encoder:
        features = encoder(features)
    with open(embedings_path, 'wb') as f:
        np.save(f, features)