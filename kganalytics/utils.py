import pandas as pd
import networkx as nx


def top_n(data_dict, n, smallest=False):
    df = pd.DataFrame(dict(data_dict).items(), columns=["id", "value"])
    if smallest:
        df = df.nsmallest(n, columns=["value"])
    else:
        df = df.nlargest(n, columns=["value"])
    return(list(df["id"]))


def subgraph_by_types(graph, type_data, types_to_include, types_to_exclude=None, include_nodes=None):
    """Construct a subgraph of the source graph containing only the nodes of particular type."""
    if include_nodes is None:
        include_nodes = []
    
    if types_to_exclude is None:
        types_to_exclude = []

    def has_needed_type(n):
        n_types = type_data[n]
        include_found = False
        exclude_found = False
        for t in n_types:
            if t in types_to_include:
                include_found = True
            if t in types_to_exclude:
                exclude_found = True
        return include_found and not exclude_found
    
    nodes = [
        n for n in graph.nodes()
        if has_needed_type(n) or n in include_nodes
    ]
    return nx.Graph(graph.subgraph(nodes))