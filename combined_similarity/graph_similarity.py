import networkx as nx

def normalized_graph_similarity(schema1, schema2):
    max_len = max(len(schema1), len(schema2))
    return 1 - (compute_graph_edit_distance(schema1, schema2) / max_len)


def json_to_graph(json_data):
    G = nx.Graph()
    
    tables = json_data.get("ai_dict", {}).get("tables", [])
    
    for table in tables:
        table_name = table.get("name", "Unnamed Table")
        G.add_node(table_name)
        
        fields = table.get("fields", [])
        for field in fields:
            field_name = field.get("name", "Unnamed Field")
            G.add_node(field_name)
            G.add_edge(table_name, field_name)
            
            # Add edges for lookups
            if "lookup" in field:
                lookup = field["lookup"]
                parent_table = lookup.get("parentTable", "Unknown Parent Table")
                parent_field = lookup.get("parentField", "Unknown Parent Field")
                G.add_edge(field_name, f"{parent_table}.{parent_field}")
            
            # Add edges for summaries
            if "summary" in field:
                summary = field["summary"]
                child_table = summary.get("childTable", "Unknown Child Table")
                child_foreign_key_field = summary.get("childForeignKeyField", "Unknown Foreign Key Field")
                G.add_edge(field_name, f"{child_table}.{child_foreign_key_field}")
    
    return G

def compute_graph_edit_distance(json_data1, json_data2):
    G1 = json_to_graph(json_data1)
    G2 = json_to_graph(json_data2)
    
    ged = graph_edit_distance(G1, G2)
    
    # Since graph_edit_distance returns an iterator, we take the first value (approximation)
    return next(ged)

# Example usage with provided JSONs
import json

with open('path_to_your_file/app1.json') as f1, open('path_to_your_file/app2.json') as f2:
    json_data1 = json.load(f1)
    json_data2 = json.load(f2)

distance = compute_graph_edit_distance(json_data1, json_data2)
print(f"Graph Edit Distance: {distance}")
