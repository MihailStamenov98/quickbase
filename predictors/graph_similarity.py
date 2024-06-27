import networkx as nx
from difflib import SequenceMatcher
import networkx as nx
from scipy.optimize import linear_sum_assignment
import numpy as np

def create_graph(tables):
    graph = nx.DiGraph()    
    for table in tables:
        graph.add_node(table['name'], type='table')
        for field in table['fields']:
            graph.add_node(f"{table['name']}.{field['name']}", type='column')
            graph.add_edge(table['name'], f"{table['name']}.{field['name']}")
    for table in tables:
        for field in table['fields']:
            if field.get('parent'):
                parent_table = field['parent']
                graph.add_edge(f"{parent_table}.{field['proxyDisplayField']}", f"{table['name']}.{field['name']}")
    return graph

def node_similarity_score(name1, name2):
    return SequenceMatcher(None, name1, name2).ratio()

def graph_similarity(G1, G2):
    G1_nodes = list(G1.nodes(data=True))
    G2_nodes = list(G2.nodes(data=True))    
    sim_matrix = np.zeros((len(G1_nodes), len(G2_nodes)))
    for i, (node1, data1) in enumerate(G1_nodes):
        for j, (node2, data2) in enumerate(G2_nodes):
            if data1['type'] == data2['type']:
                sim_matrix[i, j] = node_similarity_score(node1, node2)    
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    total_similarity = sim_matrix[row_ind, col_ind].sum()
    
    num_edits = nx.graph_edit_distance(G1, G2)
    combined_score = total_similarity / (1 + num_edits)
    
    return combined_score

def compare_schemas(schema_graphs, new_schema):
    (new_schema_graph, schema_tables) = new_schema
    max_similarity = -1
    most_similar_schema = None
    
    for schema_name, (schema_graph, schema_tables) in schema_graphs.items():
        similarity = graph_similarity(schema_graph, new_schema_graph)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_schema = schema_name
    
    return most_similar_schema, max_similarity

def read_schema_from_json(data):
    return data['ai_dict']['tables']

def graph_sim(json_objects, query_json):
    schema_graphs = {}    
    for schema_file in json_objects:
        tables = read_schema_from_json(schema_file)
        graph = create_graph(tables)
        schema_graphs[schema_file['ai_dict']['name']] = (graph, tables)
    
    new_schema_tables = read_schema_from_json(query_json)
    new_schema_graph = create_graph(new_schema_tables)
    most_similar_schema, similarity = compare_schemas(schema_graphs, new_schema_graph)
