import time

from json_to_schema_description import *
from utils import *
from encoders import *
from TfidfVectorizer_encoder import *
start_time = time.time()
 
json_files = ["app1.json", "app2.json"]
json_objects = [load_json(file) for file in json_files]

#end_time = time.time()
#
#print(f"Function execution time: {end_time - start_time} seconds")
schema_description = json_to_schema_description(json_objects[0])
#print("Schema Description:\n", schema_description)
print(len(schema_description))
#end_time = time.time()
#
#print(f"Function execution time: {end_time - start_time} seconds")
bert_embedding = embed_text_with_bert(schema_description)
#end_time = time.time()
#
#print(f"Function execution time: {end_time - start_time} seconds")
#electra_embedding = embed_text_with_electra(schema_description)
#
#end_time = time.time()
#
#print(f"Function execution time: {end_time - start_time} seconds")
#tf_idf_encoding_create_embedings(json_objects, 5, 'bert')
#print("BERT Embedding:\n", bert_embedding.shape)
#print("ELECTRA Embedding:\n", electra_embedding.shape)
#
#
#
#from transformers import pipeline
#
#summarizer = pipeline("summarization")
#summary = summarizer(text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
#embedding = embed_text_with_bert(summary)

