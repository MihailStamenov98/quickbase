#import json
#import numpy as np
#from gensim.models import Word2Vec
#from sklearn.metrics.pairwise import cosine_similarity
#from utils import *
#
#def json_to_sentences(json_objects):
#    json_strings = [json.dumps(obj) for obj in json_objects]
#    sentences = [s.split() for s in json_strings]
#    return sentences
#
#def word2vec_embeding_compare(json_objects, encoder, query_json):
#    
#    sentences = json_to_sentences(json_objects)
#
#    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
#
#    def get_embedding(sentence, model):
#        embeddings = [model.wv[word] for word in sentence if word in model.wv]
#        return np.mean(embeddings, axis=0)
#
#    embeddings = [get_embedding(sentence, word2vec_model) for sentence in sentences]
#
#    query_sentence = json_to_sentences([query_json])[0]
#    query_embedding = get_embedding(query_sentence, word2vec_model)
#
#    similarity_scores = cosine_similarity([query_embedding], embeddings)
#
#    most_similar_index = np.argmax(similarity_scores)
#    most_similar_json = json_objects[most_similar_index]
#
#    print("Most similar JSON with word2vec_embeding_compare:", most_similar_json)
#