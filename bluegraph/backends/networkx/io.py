import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix

import pandas as pd

from bluegraph.core.io import GraphProcessor, PandasPGFrame


def pgframe_to_networkx(pgframe, directed=True):
    """Create a NetworkX graph from the PGFrame."""
    if directed:
        create_using = nx.DiGraph
    else:
        create_using = nx.Graph
    edge_attr = True if len(pgframe.edge_properties()) > 0 else None
    graph = nx.from_pandas_edgelist(
        pgframe._edges.reset_index(),
        source="@source_id", target="@target_id",
        edge_attr=edge_attr, create_using=create_using)
    nx.set_node_attributes(graph, pgframe._nodes.to_dict("index"))
    return graph


def networkx_to_pgframe(nx_object):
    """Create a PGFrame from the networkx object."""
    pgframe = PandasPGFrame(nodes=nx_object.nodes())

    aggregated_props = {}
    for n, d in nx_object.nodes.items():
        for k in d:
            if k not in aggregated_props:
                aggregated_props[k] = {}
            aggregated_props[k][n] = d[k]

    for k in aggregated_props:
        pgframe.add_node_properties(
            pd.DataFrame(
                aggregated_props[k].items(), columns=["@id", k]))

    edges = nx.to_pandas_edgelist(nx_object).rename(
        columns={
            "source": "@source_id", "target": "@target_id"
        }).set_index(["@source_id", "@target_id"])
    pgframe._edges = edges
    return pgframe


class NXGraphProcessor(GraphProcessor):
    """NetworkX graph processor.

    The provided interface allows to convert NetworkX objects
    into PGFrames and vice versa.
    """

    @staticmethod
    def _generate_graph(pgframe, directed=True):
        return pgframe_to_networkx(pgframe, directed=directed)

    def _generate_pgframe(self, node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        return networkx_to_pgframe(self.graph)

    @staticmethod
    def _is_directed(graph):
        return graph.is_directed()

    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        return new_property

    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
        nx.set_node_attributes(
            self.graph, new_property, property_name)

    def _get_adjacency_matrix(self, nodes, weight=None):
        return adjacency_matrix(self.graph, nodelist=nodes, weight=weight)

    def _get_node_property_values(self, prop, nodes):
        attrs = nx.get_node_attributes(self.graph, prop)
        props = [attrs[n] for n in nodes]
        return props

    def nodes(self, properties=False):
        return list(self.graph.nodes(data=properties))

    def get_node(self, node):
        return self.graph.node[node]

    def remove_node(self, node):
        self.graph.remove_node(node)

    def rename_nodes(self, node_mapping):
        nx.relabel_nodes(self.graph, node_mapping, copy=False)

    def set_node_properties(self, node, properties):
        self.graph.nodes[node].clear()
        self.graph.nodes[node].update(properties)

    def edges(self, properties=False):
        return list(self.graph.edges(data=properties))

    def get_edge(self, edge):
        return self.graph.edge[edge]

    def add_edge(self, source, target, properties):
        self.graph.add_edge(source, target, **properties)

    def neighbors(self, node_id):
        """Get neighors of the node."""
        return list(self.graph.neighbors(node_id))

    def subgraph(self, nodes_to_include=None, edges_to_include=None,
                 nodes_to_exclude=None, edges_to_exclude=None):
        """Produce a graph induced by the input nodes."""
        if nodes_to_include is None:
            nodes_to_include = self.nodes()

        if edges_to_include is None:
            edges_to_include = self.edges()

        if nodes_to_exclude is None:
            nodes_to_exclude = []

        if edges_to_exclude is None:
            edges_to_exclude = []

        nodes_to_include = [
            n for n in self.graph.nodes()
            if n in nodes_to_include and n not in nodes_to_exclude
        ]

        subgraph = self.graph.subgraph(nodes_to_include)

        if edges_to_exclude is not None:
            subgraph = subgraph.edge_subgraph(
                [
                    e for e in subgraph.edges()
                    if e in edges_to_include and e not in edges_to_exclude
                ]
            )

        return subgraph
