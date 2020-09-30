"""Collection of utils for graph export."""
import pickle

import pandas as pd
import networkx as nx


def save_nodes(graph, path):
    """Save graph nodes as a pickled pandas.DataFrame."""
    df = pd.DataFrame(index=graph.nodes())
    # We assume that the graphs are homogeneous
    first_node = list(graph.nodes())[0]
    attrs = graph.nodes[first_node].keys()
    for a in attrs:
        df[a] = pd.Series(
            nx.get_node_attributes(graph, a))

    with open(path, "wb") as f:
        pickle.dump(df, f)


def save_to_gephi(graph, prefix, node_attr_mapping,
                  edge_attr_mapping, edge_filter=None):
    """Save the graph for Gephi import."""
    # Gephi asks for node ids to be numerical
    nodes_to_indices = {
        n: i + 1
        for i, n in enumerate(graph.nodes()) if n in graph.nodes()
    }

    ordered_edge_attrs = list(edge_attr_mapping.keys())
    edge_header = "Source;Target;{}\n".format(
        ";".join([
            edge_attr_mapping[attr]
            for attr in ordered_edge_attrs
        ]))

    def generate_edge_repr(u, v):
        return ";".join([
            str(graph.edges[u, v][attr])
            for attr in ordered_edge_attrs])

    edge_repr = "\n".join([
        "{};{};{}".format(
            nodes_to_indices[u],
            nodes_to_indices[v],
            generate_edge_repr(u, v))
        for u, v in graph.edges()
        if edge_filter is None or edge_filter(u, v, graph.edges[u, v])
    ])

    with open("{}_edges.csv".format(prefix), "w+") as f:
        f.write(edge_header + edge_repr)

    ordered_node_attrs = list(node_attr_mapping.keys())
    node_header = "Id;Label;{}\n".format(
        ";".join([
            node_attr_mapping[attr]
            for attr in ordered_node_attrs
        ]))

    def generate_node_repr(n):
        return ";".join([
            str(graph.nodes[n][attr])
            for attr in ordered_node_attrs])

    node_repr = "\n".join([
        "{};{};{}".format(
            nodes_to_indices[n],
            n,
            generate_node_repr(n)
        )
        for n in graph.nodes()
    ])

    with open("{}_nodes.csv".format(prefix), "w+") as f:
        f.write(node_header + node_repr)


def load_network(edge_path, node_path, edge_attr=None):
    if edge_attr is None:
        edge_attr = [
            "frequency", "ppmi", "npmi", "distance_ppmi", "distance_npmi"
        ]
    with open(edge_path, "rb") as f:
        edge_list = pickle.load(f)
    network = nx.from_pandas_edgelist(
        edge_list,
        edge_attr=edge_attr)

    with open(node_path, "rb") as f:
       node_list = pickle.load(f)
    nx.set_node_attributes(network, node_list.to_dict("index"))

    return network
