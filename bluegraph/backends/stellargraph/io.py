# BlueGraph: unifying Python framework for graph analytics and co-occurrence analysis. 

# Copyright 2020-2021 Blue Brain Project / EPFL

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import numpy as np
import pandas as pd

import stellargraph as sg

from bluegraph.core.io import PandasPGFrame


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


def stellargraph_to_pgframe(sg_object, node_prop_types=None,
                            edge_prop_types=None):
    # Create nodes data frame
    node_dicts = []
    node_ids = sg_object.nodes()
    if sg_object.node_types != {"default"}:
        for node_type in sg_object.node_types:
            node_ids = sg_object.nodes(node_type=node_type)
            features = sg_object.node_features(node_ids, node_type=node_type)
            for node_id, node_features in zip(node_ids, features):
                node_dict = {"@id": node_id, "@type": node_type}
                if len(node_features) > 0:
                    node_dict["features"] = node_features
                node_dicts.append(node_dict)
    else:
        features = sg_object.node_features(node_ids)
        for node_id, node_features in zip(node_ids, features):
            node_dict = {"@id": node_id}
            if len(node_features) > 0:
                node_dict["features"] = node_features
            node_dicts.append(node_dict)

    nodes = pd.DataFrame(node_dicts).set_index("@id")
    # Create edges data frame
    edge_dicts = [
        {"@source_id": s, "@target_id": t, "@type": etype, "weight": weight}
        for s, t, etype, weight in
        zip(
            sg_object._nodes.ids.from_iloc(sg_object._edges.sources),
            sg_object._nodes.ids.from_iloc(sg_object._edges.targets),
            sg_object._edges.type_of_iloc(slice(None)),
            sg_object._edges.weights
        )
    ]
    edges = pd.DataFrame(edge_dicts).set_index(["@source_id", "@target_id"])
    return PandasPGFrame.from_frames(
        nodes=nodes, edges=edges, node_prop_types=node_prop_types,
        edge_prop_types=edge_prop_types)
