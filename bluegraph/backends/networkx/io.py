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
