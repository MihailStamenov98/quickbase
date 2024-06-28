from predictors.pretrained_models.load_models import *
from predictors.json_to_schema_description import *
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ..predictor import Predictor
from pathlib import Path


class Encoders(Predictor):
    def __init__(self, model_name, json_objects=None):
        self.model_name = model_name
        self.set_path()
        if json_objects:
            texts = [
                json_to_schema_description(json_object) for json_object in json_objects
            ]
            self.embeddings = [self.embed_text(text) for text in texts]
            self.save_encoded_features()
        else:
            self.embeddings = self.load_encoded_features()

    def set_path(self):
        current_path = Path(__file__).resolve()
        self.encodings_path = (
            "bert_encodings" if self.model_name == "bert" else "electra_encodings"
        )
        self.encodings_path = current_path.parent / "encodings" / self.encodings_path

    def predict(self, query_json):
        query_descriprion = json_to_schema_description(query_json)
        query_encodings = self.embed_text(query_descriprion)
        similarity_scores = cosine_similarity([query_encodings], self.embeddings)
        most_similar_index = np.argmax(similarity_scores)
        return most_similar_index

    def sliding_window_chunks(self, text, window_size=512, stride=128):
        chunks = []
        words = re.findall(r"\S+|\s+", text)
        size = len(words)
        for i in range(0, size, stride):
            start = i
            end = i + window_size
            if end > len(words):
                end = len(words)
                start = end - stride
            chunk = "".join(words[start:end])
            chunks.append(chunk)
        return chunks

    def embed_text(self, text):
        tokenizer, model = load_bert() if self.model_name == "bert" else load_electra()
        chunks = self.sliding_window_chunks(text)
        inputs = tokenizer(
            chunks, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        result = (embeddings.mean(dim=0)).detach().cpu().numpy()
        return result

    def load_encoded_features(self):
        with open(self.encodings_path, "rb") as f:
            embeddings = np.load(f, allow_pickle=True)
        return embeddings

    def save_encoded_features(self):
        with open(self.encodings_path, "wb") as f:
            np.save(f, self.embeddings)
