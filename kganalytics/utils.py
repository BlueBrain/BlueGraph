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


def merge_attrs(old_attrs, collection_of_attrs, attr_resolver, attrs_to_ignore=None):
    
    if attrs_to_ignore is None:
        attrs_to_ignore = []
    
    all_keys = set(sum([list(attrs.keys()) for attrs in collection_of_attrs], []))
    
    for k in all_keys:       
        if k not in attrs_to_ignore:
            if k in attr_resolver:
                old_attrs[k] = attr_resolver[k](
                    ([old_attrs[k]] if k in old_attrs else []) + [
                        attrs[k] for attrs in collection_of_attrs if k in attrs
                    ]
                )
            else:
                old_attrs[k] = None


def merge_nodes(graph, nodes_to_merge, new_name=None, attr_resolver=None, copy=False):
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