import graph_tool as gt


def pgframe_to_graph_tool(pgframe, directed=True):
    """Create a graph-tool graph from the PGFrame."""
    graph = gt.Graph(directed=False)

    # Add nodes
    graph.add_vertex(pgframe.number_of_nodes())

    # Encode original node id's as a new property `@label`
    prop = graph.new_vertex_property(
        "string", vals=pgframe._nodes.index)
    graph.vertex_properties["id"] = prop

    # Add node properties
    for c in pgframe._nodes.columns:
        prop_type = "string"
        if pgframe._node_prop_types[c] == "numeric":
            prop_type = "double"

        prop = graph.new_vertex_property(
            prop_type, vals=pgframe._nodes[c])
        graph.vertex_properties[c] = prop

    # Add edges and edge properties
    new_props = []
    for c in pgframe._edges.columns:
        prop_type = "string"
        if pgframe._edge_prop_types[c] == "numeric":
            prop_type = "double"
        props = graph.new_edge_property(prop_type)
        graph.edge_properties[c] = props
        new_props.append(props)

    gt_edges = pgframe._edges.reset_index()
    gt_edges["@source_id"] = gt_edges["@source_id"].apply(
        lambda x: pgframe._nodes.index.get_loc(x))
    gt_edges["@target_id"] = gt_edges["@target_id"].apply(
        lambda x: pgframe._nodes.index.get_loc(x))
    graph.add_edge_list(gt_edges.values.tolist(), eprops=new_props)
    return graph


def graph_tool_to_pgframe(gt_object):
    pass
