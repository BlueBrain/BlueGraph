import pandas as pd
import graph_tool as gt


from bluegraph.core.io import PandasPGFrame, GraphProcessor


NUMERIC_TYPES = [
    "int16_t", "int32_t", "int64_t", "double", "long double"
]


def pgframe_to_graph_tool(pgframe, directed=True):
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
    """Create a PGFrame from the graph-tool object."""
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

    edges = [
        (graph.vp["@id"][e.source()], graph.vp["@id"][e.target()])
        for e in graph.edges()]
    pgframe.add_edges(edges)

    for k, v in graph.ep.items():
        prop = pd.DataFrame(
            [
                [edges[i][0], edges[i][1], el]
                for i, el in enumerate(list(v))
            ],
            columns=["@source_id", "@target_id", k]
        )
        prop.set_index(["@source_id", "@target_id"])
        if k == "@type":
            pgframe.assign_edge_types(prop)
        else:
            prop_type = v.value_type()
            result_type = "category"
            if prop_type in NUMERIC_TYPES:
                result_type = "numeric"
            pgframe.add_edge_properties(prop)
            pgframe._set_edge_prop_type(k, result_type)

    return pgframe


class GTGraphProcessor(GraphProcessor):
    """graph-tool graph processor.

    The provided interface allows to convert graph-tool objects
    into PGFrames and vice versa.
    """

    @staticmethod
    def _generate_graph(pgframe, directed=True):
        return pgframe_to_graph_tool(pgframe, directed=directed)

    def _generate_pgframe(self, node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        return graph_tool_to_pgframe(self.graph)

    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        return dict(
            zip(list(self.graph.vertex_properties["@id"]), new_property.a))

    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
        self.graph.vertex_properties[property_name] = new_property
