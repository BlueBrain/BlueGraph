import pandas as pd
import graph_tool as gt


from bluegraph.core.io import PandasPGFrame, GraphProcessor


NUMERIC_TYPES = [
    "int16_t", "int32_t", "int64_t", "double", "long double"
]


def pgframe_to_graph_tool(pgframe, directed=False):
    """Create a graph-tool graph from the PGFrame."""
    graph = gt.Graph(directed=directed)

    # Add nodes
    graph.add_vertex(pgframe.number_of_nodes())

    # Encode original node id's as a new property `@label`
    prop = graph.new_vertex_property(
        "string", vals=pgframe._nodes.index)
    graph.vertex_properties["@id"] = prop

    # Add node properties
    for c in pgframe._nodes.columns:
        prop_type = "string"
        if c in pgframe._node_prop_types and\
           pgframe._node_prop_types[c] == "numeric":
            prop_type = "double"

        prop = graph.new_vertex_property(
            prop_type, vals=pgframe._nodes[c])
        graph.vertex_properties[c] = prop

    # Add edges and edge properties
    new_props = []
    for c in pgframe._edges.columns:
        prop_type = "string"
        if c in pgframe._edge_prop_types and\
           pgframe._edge_prop_types[c] == "numeric":
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


def graph_tool_to_pgframe(graph):
    pgframe = PandasPGFrame(nodes=graph.vp["@id"])
    for k, v in graph.vp.items():
        if k != "@id":
            prop = pd.DataFrame(list(v), index=graph.vp["@id"], columns=[k])
            if k == "@type":
                pgframe.assign_node_types(prop)
            else:
                prop_type = v.value_type()
                result_type = "category"
                if prop_type in NUMERIC_TYPES:
                    result_type = "numeric"
                pgframe.add_node_properties(prop)
                pgframe._set_node_prop_type(k, result_type)
    return pgframe


class GTGraphProcessor(GraphProcessor):

    @staticmethod
    def _generate_graph(pgframe):
        return pgframe_to_graph_tool(pgframe)

    def _generate_pgframe(self, node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        return graph_tool_to_pgframe(self.graph)
