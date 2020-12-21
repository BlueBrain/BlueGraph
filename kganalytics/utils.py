"""Module containing a collection of generic utils for the package."""
import pandas as pd
import networkx as nx


def keys_by_value(d, val):
    """Return a list of keys associated with the input value."""
    result = []
    for k, v in d.items():
        if v == val:
            result.append(k)
    return result


def top_n(data_dict, n, smallest=False):
    """Return top `n` keys of the input dictionary by their value."""
    df = pd.DataFrame(dict(data_dict).items(), columns=["id", "value"])
    if smallest:
        df = df.nsmallest(n, columns=["value"])
    else:
        df = df.nlargest(n, columns=["value"])
    return(list(df["id"]))


def subgraph_by_types(graph, type_data, types_to_include,
                      types_to_exclude=None, include_nodes=None):
    """Construct a subgraph containing only the nodes of particular types

    Parameters
    ----------
    graph : nx.(Di)Graph
        Input graph
    type_data : dict
        Dictionary containing typing data, keys are node IDs and values are
        their types (we assume there can be multiple types per node).
    types_to_include : iterable, optional
        Set of types to include in the resulting subgraph
    types_to_exclude : iterable, optional
        Set of types to exclude from the resulting graph
    include_nodes : iterable, optional
        Set of nodes to add to the resulting graph disregarding their types.

    Returns
    -------
    result_graph : nx.Graph
        Resulting subnetwork

    """
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

    result_graph = nx.Graph(graph.subgraph(nodes))

    return result_graph


def merge_attrs(target_attrs, collection_of_attrs, attr_resolver,
                attrs_to_ignore=None):
    """Merge two attribute dictionaries into the target using the input resolver.

    Parameters
    ----------
    target_attrs : dict
        Target dictionary with attributes (the other attributes will be
        merged into it)
    collection_of_attrs : iterable of dict
        Collection of dictionaries to merge into the target dictionary
    attr_resolver : dict
        Dictionary containing attribute resolvers, its keys are attribute
        names and its values are functions applied to the set of attribute
        values in order to resolve this set to a single value
    attrs_to_ignore : iterable, optional
        Set of attributes to ignore (will not be included in the merged
        node or edges incident to this merged node)
    """
    if attrs_to_ignore is None:
        attrs_to_ignore = []

    all_keys = set(
        sum([list(attrs.keys()) for attrs in collection_of_attrs], []))

    for k in all_keys:
        if k not in attrs_to_ignore:
            if k in attr_resolver:
                target_attrs[k] = attr_resolver[k](
                    ([target_attrs[k]] if k in target_attrs else []) + [
                        attrs[k] for attrs in collection_of_attrs if k in attrs
                    ]
                )
            else:
                target_attrs[k] = None


def merge_nodes(graph, nodes_to_merge, new_name=None, attr_resolver=None,
                copy=False):
    """Merge the input set of nodes.

    Parameters
    ----------
    graph : nx.Graph
        Input graph object
    nodes_to_merge: iterable
        Collection of node IDs to merge
    new_name : str, optional
        New name to use for the result of merging
    attr_resolver : dict, optional
        Dictionary containing attribute resolvers, its keys are attribute
        names and its values are functions applied to the set of attribute
        values in order to resolve this set to a single value
    copy : bool, optional
        Flag indicating whether the merging should be performed in-place or
        by creating a copy of the input graph

    Returns
    -------
    graph : nx.Graph
        Resulting graph (references to the input graph, if `copy` is False,
        or to another object if `copy` is True).
    """
    if copy:
        graph = graph.copy()

    if len(nodes_to_merge) < 2:
        raise ValueError("At least two nodes are required for merging")

    if attr_resolver is None:
        raise ValueError("Attribute resolver should be provided")

    # We merge everything into the target node
    if new_name is None:
        new_name = nodes_to_merge[0]

    if new_name not in graph.nodes():
        nx.relabel_nodes(graph, {nodes_to_merge[0]: new_name}, copy=False)
        nodes_to_merge = nodes_to_merge[1:]

    target_node = new_name
    other_nodes = [n for n in nodes_to_merge if n != target_node]

    # Resolve node attrs
    merge_attrs(
        graph.nodes[target_node],
        [graph.nodes[n] for n in other_nodes],
        attr_resolver)

    # Merge edges
    edge_attrs = {}

    for n in other_nodes:
        neighbors = graph.neighbors(n)
        for neighbor in neighbors:
            if neighbor != target_node and neighbor not in other_nodes:
                if neighbor in edge_attrs:
                    edge_attrs[neighbor].append(graph.edges[n, neighbor])
                else:
                    edge_attrs[neighbor] = [graph.edges[n, neighbor]]

    for k, v in edge_attrs.items():
        target_neighbors = graph.neighbors(target_node)
        if k not in target_neighbors:
            graph.add_edge(target_node, k)
        merge_attrs(graph.edges[target_node, k], v, attr_resolver)

    for n in other_nodes:
        graph.remove_node(n)

    return graph
