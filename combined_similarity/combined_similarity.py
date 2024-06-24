from graph_similarity import *
from textual_similarity import *
from table_extraction import *

def combined_similarity(schema1, schema2):
    text_sim = compute_textual_similarity(schema1, schema2)
    graph_sim = normalized_graph_similarity(schema1, schema2)
    return 0.5 * text_sim + 0.5 * graph_sim

# Finding the most similar JSON
def find_most_similar(query_schema, schemas):
    similarities = {}
    for app_id, schema in schemas.items():
        similarities[app_id] = combined_similarity(query_schema, schema)
    return max(similarities, key=similarities.get)

query_schema = schemas.pop('QueryAppID')  # Replace with actual query JSON extraction
most_similar_app_id = find_most_similar(query_schema, schemas)
print(f"The most similar app is: {most_similar_app_id}")
