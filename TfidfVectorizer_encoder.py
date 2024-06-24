from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import json
from json_to_schema_description import *


def json_to_features(json_objects, query_json):
    target_features = json_to_schema_description(query_json)
    json_features = [json_to_schema_description(json_data) for json_data in json_objects]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([target_features] + json_features)
    
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity_matrix[0]

def tf_idf_encoding_create_embedings(json_objects, encoder, encoder_name):
     embedings_path = 'tf_idf_' + encoder_name + '_embeddings.npy'
     save_encoded_features(json_objects, encoder, json_to_features, embedings_path)

def tf_idf_encoding_compare(json_objects, query_json):
    similarities = json_to_features(json_objects, query_json)
    most_similar_index = similarities.argmax()
    print("Most similar JSON with tf_idf_encoding_compare:", most_similar_index)

    return json_objects[most_similar_index], similarities[most_similar_index]

