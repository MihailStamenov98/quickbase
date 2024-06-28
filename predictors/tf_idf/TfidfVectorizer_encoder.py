from sklearn.feature_extraction.text import TfidfVectorizer
from ..json_to_schema_description import json_to_schema_description
from sklearn.metrics.pairwise import cosine_similarity
from ..predictor import Predictor


class TF_IDF(Predictor):
    def __init__(self, json_objects):
        self.json_objects = json_objects
        self.json_features = [
            json_to_schema_description(json_data) for json_data in self.json_objects
        ]

    def predict(self, query_json):
        similarities = self.json_to_features(query_json)
        most_similar_index = similarities.argmax()
        return most_similar_index

    def json_to_features(self, query_json):
        target_features = json_to_schema_description(query_json)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([target_features] + self.json_features)

        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return similarity_matrix[0]
