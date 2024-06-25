import time
from functools import partial
from transformers import ElectraTokenizer, ElectraModel
from transformers import BertTokenizer, BertModel
from json_to_schema_description import *
from utils import *
from encoders import *
from TfidfVectorizer_encoder import *
start_time = time.time()




json_files = ["app1.json", "app2.json"]
json_objects = [load_json(file) for file in json_files]

schema_description = json_to_schema_description(json_objects[0])
bert_embedding = embed_text_with_bert(schema_description)
#print(bert_embedding)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds") 
print(cosine_similarity([bert_embedding], [bert_embedding]))
#most_similar = get_most_similar(json_objects=json_objects, 
#                                query_json=json_objects[0], 
#                                feature_extraction_func=partial(transformer_feature_extractor, model='bert')
#)
#
#print(most_similar.get("ai_dict", {}).get("name", "Unknown App"))