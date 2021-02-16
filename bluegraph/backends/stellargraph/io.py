import numpy as np
import pandas as pd

import stellargraph as sg


def pgframe_to_stellargraph(pgframe, directed=True, include_type=False,
                            feature_vector_prop=None, feature_props=None,
                            edge_weight=None):
    """Convert a PGFrame to a StellarGraph object."""
    if feature_props is None:
        feature_props = []

    feature_array = None
    if include_type:
        nodes = {}
        for t in pgframe.node_types():
            index = pgframe.nodes(typed_by=t)
            if feature_vector_prop is not None:
                feature_array = np.array(
                    pgframe.get_node_property_values(
                        feature_vector_prop, typed_by=t).to_list())
            elif len("feature_props") > 0:
                feature_array = pgframe.nodes(
                    raw_frame=True, typed_by=t)[feature_props].to_numpy()
            nodes[t] = sg.IndexedArray(feature_array, index=index)
    else:
        if feature_vector_prop is not None:
            feature_array = np.array(
                pgframe.get_node_property_values(
                    feature_vector_prop).to_list())
        elif len("feature_props") > 0:
            feature_array = pgframe.nodes(
                raw_frame=True)[feature_props].to_numpy()
        nodes = sg.IndexedArray(feature_array, index=pgframe.nodes())

    if pgframe.number_of_edges() > 0:
        edges = pgframe.edges(
            raw_frame=True,
            include_index=True,
            filter_props=lambda x: ((x == "@type") if include_type else False) or x == edge_weight,
            rename_cols={'@source_id': 'source', "@target_id": "target"})
    else:
        edges = pd.DataFrame(columns=["source", "target"])

    if directed:
        graph = sg.StellarDiGraph(
            nodes=nodes,
            edges=edges,
            edge_weight_column=edge_weight,
            edge_type_column="@type" if include_type else None)
    else:
        graph = sg.StellarGraph(
            nodes=nodes,
            edges=edges,
            edge_weight_column=edge_weight,
            edge_type_column="@type" if include_type else None)
    return graph


def stellargraph_to_pgframe(sg_object):
    pass
