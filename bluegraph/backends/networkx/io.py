import networkx as nx


def pgframe_to_networkx(pgframe, directed=True):
    """Create a NetworkX graph from the PGFrame."""
    graph = nx.from_pandas_edgelist(
        pgframe._edges.reset_index(),
        source="@source_id", target="@target_id",
        edge_attr=True)
    nx.set_node_attributes(graph, pgframe._nodes.to_dict("index"))
    return graph


def networkx_to_pgframe(nx_object):
    pass
