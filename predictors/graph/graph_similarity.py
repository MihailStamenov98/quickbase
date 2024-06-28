import networkx as nx
from difflib import SequenceMatcher
import networkx as nx
from scipy.optimize import linear_sum_assignment
import numpy as np
from ..predictor import Predictor
import math


class GraphSimilarity(Predictor):
    def __init__(self, json_objects):
        self.schema_graphs = {}
        for schema_file in json_objects:
            tables = self.read_schema_from_json(schema_file)
            graph = self.create_graph(tables)
            self.schema_graphs[schema_file["ai_dict"]["name"]] = graph
        self.calculate_min_max_degrees()

    def predict(self, query_json):
        new_schema_tables = self.read_schema_from_json(query_json)
        new_schema_graph = self.create_graph(new_schema_tables)
        print("Nodes and their attributes:")
        most_similar_schema, similarity = self.compare_schemas(new_schema_graph)
        return most_similar_schema

    def calculate_min_max_degrees(self):
        self.min_in_degree = float("inf")
        self.max_in_degree = float("-inf")
        self.min_out_degree = float("inf")
        self.max_out_degree = float("-inf")

        for graph in self.schema_graphs.values():
            for node in graph.nodes():
                in_degree = graph.in_degree(node)
                out_degree = graph.out_degree(node)
                if in_degree < self.min_in_degree:
                    self.min_in_degree = in_degree
                if in_degree > self.max_in_degree:
                    self.max_in_degree = in_degree
                if out_degree < self.min_out_degree:
                    self.min_out_degree = out_degree
                if out_degree > self.max_out_degree:
                    self.max_out_degree = out_degree

    def create_graph(self, tables):
        graph = nx.DiGraph()

        for table in tables:
            graph.add_node(table["name"], type="table")
            for field in table["fields"]:
                graph.add_node(
                    f"{table['name']}.{field['name']}",
                    type="column",
                    value_type=field["type"],
                )
                graph.add_edge(table["name"], f"{table['name']}.{field['name']}")

        for table in tables:
            for field in table["fields"]:
                if field.get("parent"):
                    parent_table = field["parent"]
                    proxy_display_field = field.get("proxyDisplayField")
                    if (
                        proxy_display_field
                        and f"{parent_table}.{proxy_display_field}" in graph
                    ):
                        graph.add_edge(
                            f"{parent_table}.{proxy_display_field}",
                            f"{table['name']}.{field['name']}",
                        )
                    else:
                        graph.add_edge(
                            f"{parent_table}",
                            f"{table['name']}.{field['name']}",
                        )
                        print(
                            f"Warning: Field '{field['name']}' in table '{table['name']}' has a parent but no proxyDisplayField."
                        )

                if field.get("lookup"):
                    lookup = field["lookup"]
                    parent_table = lookup["parentTable"]
                    parent_field = lookup["parentField"]
                    foreign_key_field = lookup["foreignKeyField"]
                    if f"{parent_table}.{parent_field}" in graph:
                        graph.add_edge(
                            f"{table['name']}.{foreign_key_field}",
                            f"{parent_table}.{parent_field}",
                            relation="lookup",
                        )
                    else:
                        graph.add_edge(
                            f"{table['name']}.{foreign_key_field}",
                            f"{parent_table}",
                            relation="lookup",
                        )

        return graph

    def node_similarity_score(self, name1, G1, name2, G2):
        degrees_node1 = self.get_normalized_node_degrees(G1, name1)
        degrees_node2 = self.get_normalized_node_degrees(G2, name2)
        degree_distance = self.euclidean_distance(degrees_node1, degrees_node2)
        degree_similarity = 1 - degree_distance
        return (
            0.3 * SequenceMatcher(None, name1, name2).ratio() + 0.70 * degree_similarity
        )

    def euclidean_distance(self, node1, node2):
        return math.sqrt(
            (node1["in_degree"] - node2["in_degree"]) ** 2
            + (node1["out_degree"] - node2["out_degree"]) ** 2
        )

    def normalize_degrees(self, degree, min_degree, max_degree):
        if max_degree == min_degree:
            return 0
        return (degree - min_degree) / (max_degree - min_degree)

    def get_normalized_node_degrees(self, graph, node):
        in_degree = self.normalize_degrees(
            graph.in_degree(node), self.min_in_degree, self.max_in_degree
        )
        out_degree = self.normalize_degrees(
            graph.out_degree(node), self.min_out_degree, self.max_out_degree
        )
        return {"in_degree": in_degree, "out_degree": out_degree}

    def graph_similarity(self, G1, G2):
        G1_nodes = list(G1.nodes(data=True))
        G2_nodes = list(G2.nodes(data=True))
        print(G1_nodes[0])
        matrix_size = max(len(G1_nodes), len(G2_nodes))
        sim_matrix = np.zeros((matrix_size, matrix_size))
        for i, (node1, data1) in enumerate(G1_nodes):
            for j, (node2, data2) in enumerate(G2_nodes):
                if data1["type"] == data2["type"]:
                    if "value_type" in data1 and "value_type" in data1:
                        sim_matrix[i, j] = self.node_similarity_score(
                            node1, G1, node2, G2
                        )
                    else:
                        sim_matrix[i, j] = 0.5 * self.node_similarity_score(
                            node1, G1, node2, G2
                        )
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        total_similarity = sim_matrix[row_ind, col_ind].sum() / matrix_size
        return total_similarity

    def compare_schemas(self, new_schema_graph):
        max_similarity = -1
        most_similar_schema = None

        for schema_name, schema_graph in self.schema_graphs.items():
            similarity = self.graph_similarity(schema_graph, new_schema_graph)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_schema = schema_name

        return most_similar_schema, max_similarity

    def read_schema_from_json(self, data):
        return data["ai_dict"]["tables"]
