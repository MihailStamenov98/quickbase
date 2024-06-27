from abc import ABC, abstractmethod
import json

class Predictor(ABC):
    def __init__(self, json_objects):
        if isinstance(json_objects, str):
            self.json_objects = json.loads(json_objects)
        else:
            self.json_objects = json_objects

    @abstractmethod
    def predict(self, query_json):
        pass
