import networkx as nx
import pandas as pd

from bluegraph.core.io import GraphProcessor, PandasPGFrame


def pgframe_to_networkx(pgframe, directed=True):
    """Create a NetworkX graph from the PGFrame."""
    if directed:
        create_using = nx.DiGraph
    else:
        create_using = nx.Graph
    graph = nx.from_pandas_edgelist(
        pgframe._edges.reset_index(),
        source="@source_id", target="@target_id",
        edge_attr=True, create_using=create_using)
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
