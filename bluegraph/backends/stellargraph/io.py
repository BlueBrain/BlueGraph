import numpy as np
import pandas as pd

import stellargraph as sg


def pgframe_to_stellargraph(pgframe, directed=True, include_type=True,
                            feature_prop=None):
    """Convert a PGFrame to a StellarGraph object."""
    feature_array = None
    if include_type:
        nodes = {}
        for t in pgframe.node_types():
            index = pgframe.nodes(typed_by=t)
            if feature_prop:
                feature_array = np.array(
                    pgframe.get_node_property_values(
                        feature_prop,
                        typed_by=t).to_list())
            nodes[t] = sg.IndexedArray(feature_array, index=index)
    else:
        if feature_prop:
            feature_array = np.array(
                pgframe.get_node_property_values(
                    feature_prop).to_list())
        nodes = sg.IndexedArray(feature_array, index=pgframe.nodes())

    if pgframe.number_of_edges() > 0:
        edges = pgframe.edges(
            raw_frame=True,
            include_index=True,
            filter_props=lambda x: (x == "@type") if include_type else False,
            rename_cols={'@source_id': 'source', "@target_id": "target"})
    else:
        edges = pd.DataFrame(columns=["source", "target"])

    if directed:
        graph = sg.StellarDiGraph(
            nodes=nodes,
            edges=edges,
            edge_type_column="@type" if include_type else None)
    else:
        graph = sg.StellarGraph(
            nodes=nodes,
            edges=edges,
            edge_type_column="@type" if include_type else None)
    return graph


def stellargraph_to_pgframe(sg_object):
    pass
