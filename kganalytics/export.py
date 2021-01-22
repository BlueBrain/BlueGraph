#
# Blue Brain Graph is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blue Brain Graph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Blue Brain Graph. If not, see <https://choosealicense.com/licenses/lgpl-3.0/>.

"""Collection of utils for graph export."""
import pickle

import pandas as pd
import networkx as nx


def save_to_gephi(graph, prefix, node_attr_mapping,
                  edge_attr_mapping, edge_filter=None):
    """Save the graph for Gephi import.

    Saves the graph as two `.csv` files one with nodes (`<prefix>_nodes.csv`)
    and one with edges (`<prefix>_edges.csv`). Node IDs are replaced by
    interger identifiers (Gephi asks for node IDs to be numerical) and
    entity names are added as the node property 'Label'.
    """
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


def save_nodes(graph, path):
    """Save graph nodes as a pickled pandas.DataFrame."""
    df = pd.DataFrame(index=graph.nodes())
    # We assume that the graphs are homogeneous
    # (attribute keys are the same across all the nodes)
    first_node = list(graph.nodes())[0]
    attrs = graph.nodes[first_node].keys()
    for a in attrs:
        df[a] = pd.Series(
            nx.get_node_attributes(graph, a))

    with open(path, "wb") as f:
        pickle.dump(df, f)


def save_network(network, prefix):
    """Save networkx object into pickled node/edge tables.

    Saves the graph as two `.pkl` files one with nodes (`<prefix>_node_list.pkl`)
    and one with edges (`<prefix>_edge_list.pkl`).

    Parameters
    ----------
    network : nx.Graph
        NetworkX graph to serialize

    """
    edgelist = nx.to_pandas_edgelist(network)
    edgelist.to_pickle("{}_edge_list.pkl".format(prefix))
    save_nodes(network, "{}_node_list.pkl".format(prefix))


def load_network(edge_path, node_path, edge_attr=None):
    """Load a graph from provided pickled dataframes with edges an nodes.

    Parameters
    ----------
    edge_path : str
        Path to the `.pkl` file with pickled pandas.Dataframe
        containing the edge list
    node_path : str
        Path to the `.pkl` file with pickled pandas.Dataframe
        containing the edge list
    edge_attr : iterable, optional
        Set of edge attributes to import. By default, tries to import
        the attributes "frequency", "ppmi", "npmi", "distance_ppmi",
        "distance_npmi".

    Returns
    -------
    network : nx.Graph
        Loaded NetworkX graph
    """
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
