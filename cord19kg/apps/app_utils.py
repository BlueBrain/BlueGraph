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

"""Collection of utils for CORD-19 analysis apps."""
import copy

# from kganalytics.utils import merge_attrs
from cord19kg.utils import CORD_ATTRS_RESOLVER


SUPPORTED_JUPYTER_DASH_MODE = ["jupyterlab", "inline", "external"]


def min_with_inf(x):
    x = [el for el in x if el is not None]
    return min(x) if len(x) > 0 else None


ATTRS_RESOLVER = CORD_ATTRS_RESOLVER.copy()
ATTRS_RESOLVER.update({
    "paper_frequency": sum,
    "paragraph_frequency": sum,
    "paper_frequency_size": max,
    "paper_frequency_font_size": max,
    "degree_frequency_size": max,
    "degree_frequency_font_size": max,
    "pagerank_frequency_size": max,
    "pagerank_frequency_font_size": max,
    "frequency_size": max,
    "ppmi_size": max,
    "npmi_size": max,
    "frequency": sum,
    "distance_ppmi": min_with_inf,
    "distance_npmi": min_with_inf,
    "distance_frequency": min_with_inf
})


def save_run(app, initial_port, mode="jupyterlab", debug=False,
             inline_exceptions=None, ascending=True, retry_limit=5):
    if mode not in SUPPORTED_JUPYTER_DASH_MODE:
        raise Exception(
            "Please provide one of the following mode value: " + str(
                SUPPORTED_JUPYTER_DASH_MODE)
        )

    port = initial_port
    for i in range(retry_limit):
        try:
            app._app.run_server(mode=mode, width="100%", port=port)
            break
        except OSError as ose:
            if ascending:
                new_port = port + 1
            else:
                new_port = port - 1
            print(
                f"Opening port number {port} failed: {str(ose)}. "
                f"Trying port number {new_port} ...")
            port = new_port


def merge_cyto_elements(elements, nodes_to_merge, new_name=None):
    # We merge everything into the target node
    if new_name is None:
        new_name = nodes_to_merge[0]

    removed_nodes = {}
    removed_edges = {}
    added_edges = []

    node_dict = {
        el["data"]["id"]: el
        for el in elements if el["data"] and "source" not in el["data"]
    }
    edge_dict = {
        el["data"]["id"]: el
        for el in elements if el["data"] and "source" in el["data"]
    }

    old_target_name = None
    if new_name not in node_dict:
        old_target_name = nodes_to_merge[0]
        removed_nodes[nodes_to_merge[0]] = copy.deepcopy(
            node_dict[nodes_to_merge[0]])

        node_dict[nodes_to_merge[0]]["data"]["id"] = new_name
        node_dict[nodes_to_merge[0]]["data"]["name"] = new_name
        node_dict[nodes_to_merge[0]]["data"]["value"] = new_name

        node_dict[new_name] = node_dict[nodes_to_merge[0]]
        del node_dict[nodes_to_merge[0]]

        nodes_to_merge = nodes_to_merge[1:]

    target_node = new_name
    other_nodes = [n for n in nodes_to_merge if n != target_node]

    # Resolve node attrs
    merge_attrs(
        node_dict[target_node]["data"],
        [node_dict[n]["data"] for n in other_nodes],
        ATTRS_RESOLVER, attrs_to_ignore=["id", "name", "value", "type"])

    # Merge edges
    target_neighbours = {}
    edge_attrs = {}

    for e_id, el in edge_dict.items():
        if el["data"]["source"] in other_nodes + [target_node] + (
                [old_target_name] if old_target_name else []):
            neighbour = el["data"]["target"]
            if neighbour not in other_nodes and neighbour != target_node and\
               neighbour != old_target_name:
                if el["data"]["source"] == target_node:
                    target_neighbours[neighbour] = el["data"]
                else:
                    if neighbour in edge_attrs:
                        edge_attrs[neighbour].append(el["data"])
                    else:
                        edge_attrs[neighbour] = [el["data"]]
            removed_edges[el["data"]["id"]] = copy.deepcopy(el)

        elif el["data"]["target"] in other_nodes + [target_node] + (
                [old_target_name] if old_target_name else []):
            neighbour = el["data"]["source"]
            if neighbour not in other_nodes and neighbour != target_node and\
               neighbour != old_target_name:
                if el["data"]["target"] == target_node:
                    target_neighbours[neighbour] = el["data"]
                else:
                    if neighbour in edge_attrs:
                        edge_attrs[neighbour].append(el["data"])
                    else:
                        edge_attrs[neighbour] = [el["data"]]
            removed_edges[el["data"]["id"]] = copy.deepcopy(el)

    for k, v in edge_attrs.items():
        if k not in target_neighbours:
            # generate new edge id
            edge_id = "{}_{}".format(target_node, k).replace(
                ",", "_").replace(" ", "_")
            added_edges.append(edge_id)
            target_neighbours[k] = {
                "id": edge_id,
                "type": "edge",
                "source": target_node,
                "target": k
            }
            elements.append({"data": target_neighbours[k]})
        else:
            removed_edges[target_neighbours[k]["id"]] = {
                "data": copy.deepcopy(target_neighbours[k])}
        merge_attrs(
            target_neighbours[k], v, ATTRS_RESOLVER,
            attrs_to_ignore=["id", "source", "target", "type"])
        target_neighbours[k]["type"] = "edge"

    # remove other nodes & edges
    for n in other_nodes:
        removed_nodes[n] = copy.deepcopy(node_dict[n])
        del node_dict[n]

    elements = [
        el
        for el in elements
        if "data" in el and el["data"] and
           el["data"]["id"] not in removed_nodes and
           el["data"]["id"] not in removed_edges
    ]

    for k in target_neighbours:
        elements.append({"data": target_neighbours[k]})

    return elements, target_node, {
        "removed_elements": {**removed_nodes, **removed_edges},
        "added_elements": [target_node] + added_edges
    }
