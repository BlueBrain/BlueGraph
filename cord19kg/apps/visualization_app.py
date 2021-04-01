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

import ast
import os
import time
import math
import numpy as np
import networkx as nx

from colorsys import rgb_to_hls, hls_to_rgb

import copy

from jupyter_dash import JupyterDash

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from bluegraph.core.utils import top_n
from bluegraph.backends.networkx import pgframe_to_networkx

import cord19kg
from cord19kg.apps.resources import (CYTOSCAPE_STYLE_STYLESHEET,
                                     MIN_NODE_SIZE,
                                     MAX_NODE_SIZE,
                                     MIN_FONT_SIZE,
                                     MAX_FONT_SIZE,
                                     MIN_EDGE_WIDTH,
                                     MAX_EDGE_WIDTH,
                                     LAYOUT_CONFIGS,
                                     COLORS,
                                     CORD19_PROP_TYPES)
import cord19kg.apps.components as components
from cord19kg.apps.app_utils import (save_run,
                                     merge_cyto_elements,
                                     ATTRS_RESOLVER)

from cord19kg.utils import (BACKEND_MAPPING,
                            build_cytoscape_data, generate_paper_lookup,
                            merge_nodes)


def adjust_color_lightness(r, g, b, factor):
    h, _l, s = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    _l = max(min(_l * factor, 1.0), 0.0)
    r, g, b = hls_to_rgb(h, _l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def lighten_color(r, g, b, factor=0.1):
    return adjust_color_lightness(r, g, b, 1 + factor)


def filter_nodes_by_attr(processor, key, values):
    result = []
    for n, properties in processor.nodes(properties=True):
        if key in properties:
            if properties[key] in values:
                result.append(n)
    return result


def subgraph_from_clusters(processor, cluster_type, clustersearch,
                           nodes_to_keep):
    nodes = dict(processor.nodes(properties=True))
    if cluster_type and clustersearch:
        graph_object = processor.subgraph(
            nodes_to_include=[
                n for n in nodes
                if nodes[n][cluster_type] in clustersearch or
                n in nodes_to_keep
            ])
    return graph_object


def get_top_n_nodes(graph_processor, n, node_subset=None, nodes_to_keep=None):
    if nodes_to_keep is None:
        nodes_to_keep = []

    nodes = dict(graph_processor.nodes(properties=True))

    if n is None:
        n = len(nodes)

    if node_subset is None:
        node_subset = list(nodes)

    if n <= len(node_subset):
        node_frequencies = {}
        for node in node_subset:
            node_properties = nodes[node]
            node_frequencies[node] = len(node_properties["paper"])
        nodes_to_include = top_n(node_frequencies, n)
    else:
        nodes_to_include = node_subset

    return nodes_to_include + [
        el for el in nodes_to_keep if el not in nodes_to_include
    ]


def top_n_subgraph(graph_processor, n, node_subset=None, nodes_to_keep=None):
    """Build a subgraph with top n nodes."""
    nodes_to_include = get_top_n_nodes(
        graph_processor, n, node_subset, nodes_to_keep)

    return graph_processor.subgraph(nodes_to_include=nodes_to_include)


def top_n_spanning_tree(graph_processor, n, backend, node_subset=None,
                        nodes_to_keep=None):
    nodes_to_include = get_top_n_nodes(
        graph_processor, n, node_subset, nodes_to_keep)
    subgraph = graph_processor.subgraph(nodes_to_include=nodes_to_include)
    path_finder = BACKEND_MAPPING[backend]["paths"].from_graph_object(
        subgraph)
    tree = path_finder.minimum_spanning_tree(distance="distance_npmi")
    return tree


def get_cytoscape_data(graph_processor, positions=None):
    # Generate a cytoscape repr of the input graph
    cyto_repr = build_cytoscape_data(graph_processor, positions=positions)

    # add some extra attrs to nodes
    weights = ["paper_frequency", "degree_frequency", "pagerank_frequency"]
    set_sizes_from_weights(
        cyto_repr, weights, MIN_NODE_SIZE, MAX_NODE_SIZE,
        MIN_FONT_SIZE, MAX_FONT_SIZE)

    # add some extra attrs to nodes
    weights = ["npmi", "ppmi", "frequency"]
    set_sizes_from_weights(
        cyto_repr, weights, MIN_EDGE_WIDTH, MAX_EDGE_WIDTH)

    return cyto_repr


def generate_sizes(start, end, weights, func="linear"):
    sorted_indices = np.argsort(weights)
    if func == "linear":
        sizes = np.linspace(start, end, len(weights))
    elif func == "log":
        sizes = np.logspace(start, end, len(weights))
    sizes = [
        int(round(sizes[el]))
        for el in np.argsort(sorted_indices)
    ]
    return sizes


def set_sizes_from_weights(cyto_repr, weights, min_size, max_size,
                           min_font_size=None, max_font_size=None):
    for weight in weights:
        all_values = [
            el["data"][weight]
            for el in cyto_repr
            if weight in el["data"]
        ]
        sizes = generate_sizes(min_size, max_size, all_values)
        if min_font_size and max_font_size:
            font_sizes = generate_sizes(
                min_font_size, max_font_size, all_values)

        j = 0
        for i in range(len(cyto_repr)):
            el = cyto_repr[i]
            if weight in el["data"]:
                el["data"]["{}_size".format(weight)] = sizes[j]
                if min_font_size and max_font_size:
                    el["data"]["{}_font_size".format(weight)] = font_sizes[j]
                j += 1


def generate_clusters(elements, cluster_type):
    new_elements = copy.deepcopy(elements)
    clusters = dict()
    for el in new_elements:
        if cluster_type in el["data"]:
            cluster_id = el["data"][cluster_type]
            el["data"]["parent"] = "cluster_node_{}".format(cluster_id)
            clusters[cluster_id] = "cluster_node_{}".format(cluster_id)

    for k, v in clusters.items():
        new_elements.append({
            "data": {
                "id": v,
                "type": "cluster",
                cluster_type: str(k)
            }
        })
    return new_elements


def clear_grouping(elements):
    new_elements = []
    for el in elements:
        if "type" not in el["data"] or el["data"]["type"] != "cluster_node":
            new_element = {"data": {}}
            for k, v in el["data"].items():
                if k != "parent":
                    new_element['data'][k] = v
            new_elements.append(new_element)
    return new_elements


def generate_gml(elements, node_freq_type=None, edge_freq_type=None):
    result = (
        """
graph
[
    Creator "bbg_app"
    directed 0
        """
    )
    for el in elements:
        if "source" not in el["data"]:
            x = y = 0.0
            if "position" in el:
                x = el["position"]["x"]
                y = el["position"]["y"]

            w = h = 10
            if node_freq_type:
                w = h = el["data"]["{}_size".format(node_freq_type)]
            if "entity_type" in el["data"]:
                result += (
"""    node
    [
        id "{}"
        label "{}"
        graphics
        [
            x {}
            y {}
            z 0.0
            w {}
            h {}
            d 0.0
            fill "{}"
        ]
    ]
"""
                ).format(
                    el["data"]["id"],
                    el["data"]["id"],
                    x, y, w, h,
                    COLORS[str(el["data"]["entity_type"])])
        else:
            edge_weight = 1
            if edge_freq_type:
                edge_weight = el["data"]["{}_size".format(edge_freq_type)]
            result += (
"""    edge
    [
        id "{}"
        source "{}"
        target "{}"
        value {}
    ]
"""
             ).format(
                 el["data"]["id"],
                 el["data"]["source"],
                 el["data"]["target"],
                 edge_weight)

    result += "]"

    return result


class VisualizationApp(object):
    """JupyterDash-based interactive graph visualization app."""

    def __init__(self, configs=None):
        self._app = JupyterDash(
            "Interactive Graph Visualization App",
            assets_folder=os.path.join(cord19kg.__path__[0], "apps/assets/"))
        self._app.add_bootstrap_links = True

        FONT_AWESOME = "https://pro.fontawesome.com/releases/v5.10.0/css/all.css"
        self._app.external_stylesheets = [
            dbc.themes.CYBORG,
            dbc.themes.BOOTSTRAP,
            FONT_AWESOME
        ]

        self._server = self._app.server

        self._graphs = {}
        self._current_graph = None

        # ---- Create a layout from components ----------------
        self._configs = configs if configs is not None else {}
        self._configure_layout(configs)

        self._app.config['suppress_callback_exceptions'] = True
        self._app.height = "800px"
        self._min_node_weight = None
        self._max_node_weight = None
        self._min_edge_weight = None
        self._max_edge_weight = None

        self._current_layout = components.DEFAULT_LAYOUT
        self._removed_nodes = set()
        self._removed_edges = set()

        self._is_initializing = False

        self._edit_history = {}
        self._backend = None

    def _configure_layout(self, configs=None):
        if configs is None:
            configs = {}

        self._configs = configs
        self.cyto, layout, self.dropdown_items, self.cluster_filter =\
            components.generate_layout(self._graphs, configs)

        self._app.layout = layout

        self._is_initializing = False

        self._edit_history = {}

    def _update_weight_data(self, graph_id, cyto_repr,
                            node_freq_type="degree_frequency",
                            edge_freq_type="npmi"):
        min_value, max_value, marks, step = recompute_node_range(
            cyto_repr, node_freq_type)
        self._graphs[graph_id]["min_node_weight"] = min_value
        self._graphs[graph_id]["max_node_weight"] = max_value
        self._graphs[graph_id]["current_node_value"] = [min_value, max_value]
        self._graphs[graph_id]["node_marks"] = marks
        self._graphs[graph_id]["node_step"] = step

        min_value, max_value, marks, step = recompute_node_range(
            cyto_repr, edge_freq_type)
        self._graphs[graph_id]["min_edge_weight"] = min_value
        self._graphs[graph_id]["max_edge_weight"] = max_value
        self._graphs[graph_id]["current_edge_value"] = [min_value, max_value]
        self._graphs[graph_id]["edge_marks"] = marks
        self._graphs[graph_id]["edge_step"] = step

    def _update_cyto_graph(self, graph_id, processor, top_n_entities=None,
                           positions=None, node_freq_type="degree_frequency",
                           edge_freq_type="npmi", node_subset=None,
                           nodes_to_keep=None):
        if not self._graphs[graph_id]["full_graph_view"]:
            if top_n_entities is None:
                # compute the spanning tree on all the nodes
                reused_tree = False
                if "full_tree" in self._graphs[graph_id]:
                    tree_object = self._graphs[graph_id]["full_tree"]
                    tree_processor = self._graph_processor.from_graph_object(
                        tree_object)
                    if len(tree_processor.nodes()) == len(processor.nodes()) and\
                       (node_subset is None or set(node_subset) == set(
                            processor.nodes())):
                        graph_view = self._graphs[graph_id]["full_tree"]
                        reused_tree = True

                if not reused_tree:
                    graph_view = top_n_spanning_tree(
                        processor, len(processor.nodes()), self._backend,
                        node_subset=node_subset, nodes_to_keep=nodes_to_keep)
            else:
                # compute the spanning tree on n nodes
                graph_view = top_n_spanning_tree(
                    processor, top_n_entities, self._backend,
                    node_subset=node_subset, nodes_to_keep=nodes_to_keep)
        else:
            if top_n_entities is None:
                graph_view = processor.graph
            else:
                graph_view = top_n_subgraph(
                    processor, top_n_entities,
                    node_subset=node_subset, nodes_to_keep=nodes_to_keep)

        if positions is None and top_n_entities is None:
            if "positions" in self._graphs[graph_id]:
                positions = self._graphs[graph_id]["positions"]

        cyto_repr = get_cytoscape_data(
            self._graph_processor.from_graph_object(graph_view),
            positions=positions)

        self._graphs[graph_id]["cytoscape"] = cyto_repr
        self._graphs[graph_id]["top_n"] = top_n_entities

        self._update_weight_data(
            graph_id, cyto_repr,
            node_freq_type=node_freq_type,
            edge_freq_type=edge_freq_type)
        return cyto_repr

    def _update_configs(self, elements, current_graph, nodes_to_keep,
                        top_n_slider_value, node_freq_type, edge_freq_type,
                        nodefreqslider, edgefreqslider, cluster_type,
                        clustersearch, searchpathfrom, searchpathto,
                        searchnodetotraverse, searchpathlimit,
                        searchpathoverlap, nestedpaths, pathdepth):
        self._configs["elements"] = elements
        self._configs["current_graph"] = current_graph
        self._configs["nodestokeep"] = nodes_to_keep
        self._configs["top_n"] = top_n_slider_value
        self._configs["node_weight"] = node_freq_type
        self._configs["edge_weight"] = edge_freq_type
        self._configs["nodefreqslider"] = nodefreqslider
        self._configs["edgefreqslider"] = edgefreqslider
        self._configs["cluster_type"] = cluster_type
        self._configs["clustersearch"] = clustersearch
        self._configs["searchpathfrom"] = searchpathfrom
        self._configs["searchpathto"] = searchpathto
        self._configs["searchnodetotraverse"] = searchnodetotraverse
        self._configs["searchpathlimit"] = searchpathlimit
        self._configs["searchpathoverlap"] = searchpathoverlap
        self._configs["nestedpaths"] = nestedpaths
        self._configs["pathdepth"] = pathdepth
        self._configs["current_layout"] = self._current_layout

    def get_configs(self):
        """Get current app configs."""
        return self._configs

    def set_backend(self, backend):
        """Set graph processing backend (currently, 'networkx' or 'graph_tool')."""
        if backend not in ["networkx", "graph_tool"]:
            raise ValueError(
                "Unknown backend '{}', available backends: ".format(backend) +
                "'networkx', 'graph_tool'")
        self._backend = backend
        self._graph_processor = BACKEND_MAPPING[backend]["object_processor"]

    def add_graph(self, graph_id, graph, tree=None,
                  positions=None, default_top_n=None, full_graph_view=False):
        """Set a graph to display.

        Parameters
        ----------
        graph_id : str
            Graph identifier to use in the app
        graph_object : PandasPGFrame
            Input graph object
        tree_object : PandasPGFrame, optional
            Pre-computed minimum spanning tree object
        positions : dict, optional
            Dictionary containing pre-computed node positions
        default_top_n : int, optional
            Top n entities to display by default
        full_graph_view : bool, optional
            Flag indicating whether the current graph should
            be displayed as a spanning tree or entirely
            (spanning tree is shown by default)
        """
        if self._backend is None:
            raise ValueError(
                "Cannot add a graph: backend is not set, use the "
                "`set_backend` method before adding graphs (available "
                "backends are 'networkx' and 'graph_tool')")

        # Generate a paper lookup table
        paper_lookup = generate_paper_lookup(graph)

        if self._graphs is None:
            self._graphs = {}

        # Create a craph object
        graph_object = BACKEND_MAPPING[self._backend]["from_pgframe"](
            graph, directed=False)
        graph_object_backup = graph_object.copy()
        if tree:
            tree_object = BACKEND_MAPPING[self._backend]["from_pgframe"](
                tree, directed=False)

        self._graphs[graph_id] = {
            "object_backup": graph_object_backup,
            "object": graph_object,
            "positions": positions,
            "paper_lookup": paper_lookup,
            "full_graph_view": full_graph_view
        }

        if tree_object:
            self._graphs[graph_id]["full_tree"] = tree_object

        processor = self._graph_processor.from_graph_object(graph_object)

        if default_top_n:
            default_top_n = (
                len(processor.nodes())
                if default_top_n > len(processor.nodes())
                else default_top_n
            )
            self._graphs[graph_id]["default_top_n"] = default_top_n

        # Build a cyto repr with the spanning tree with default top n nodes
        self._update_cyto_graph(
            graph_id, processor, default_top_n, positions)

        self.dropdown_items.options = [
            {'label': val.capitalize(), 'value': val}
            for val in list(self._graphs.keys())
        ]

        return

    def set_current_graph(self, graph_id):
        """Set current graph.

        Parameters
        ----------
        graph_id : str
            Graph identifier to set as the current graph.
        """
        self._current_graph = graph_id
        self.dropdown_items.value = graph_id
        self.cyto.elements = self._graphs[self._current_graph]["cytoscape"]

        types = set([
            el["data"]["entity_type"]
            for el in self._graphs[self._current_graph]["cytoscape"]
            if "entity_type" in el["data"]
        ])
        self.cluster_filter.options = [
            {"label": t, "value": t}
            for t in types
        ]

    def run(self, port, mode="jupyterlab", debug=False, inline_exceptions=False):
        """Run the graph visualization app.

        Parameters
        ----------
        port : int
            Port number to launch the app (`localhost:<port>`).
        mode : str, optional
            Mode in which the app should be launched. Possible values:
            `inline` inside the current Jupyter notebook, `external`
            given an external link that can be opened in a browser,
            `jupyterlab` as a new tab of JupyterLab.
        debug : bool, optional
            Flag indicating whether the app is launched in the debug mode
            (traceback will be printed).
        inline_exceptions : bool, optional
            Flag indicating whether app exceptions should be printed in
            the current active notebook cell.
        """
        if len(self._graphs) == 0:
            raise ValueError(
                "Cannot run the visualization app: "
                "not graphs are added to display, use the `add_graph` "
                "methods to add graphs.")
        try:
            save_run(
                self, port, mode=mode, debug=debug,
                inline_exceptions=inline_exceptions)
        except:
            pass

    def set_list_papers_callback(self, func):
        """Set the paper lookup callback.

        This function will be called when the user requests
        to see a list of papers associated with a selected
        node or an edge. The visualization app will pass the
        list of paper indentifiers to this function. The function
        is expected to return a list of dictionaries each representing
        a paper's meta-data.
        """
        self._list_papers_callback = func

    def set_aggregated_entities_callback(self, func):
        """Set the aggegated entities lookup callback.

        This function will be called when the user requests
        to see a set of raw entities associated with a selected
        node. The visualization app will pass the selected entity
        to this function. The function is expected to return a
        dictionary whose keys are raw entities and whose values
        are their occurrence frequencies.
        """
        self._aggregated_entities_callback = func

    def set_aggregated_entities_callback(self, func):
        """Set the aggegated entities lookup callback.

        This function will be called when the user requests
        to see a set of raw entities associated with a selected
        node. The visualization app will pass the selected entity
        to this function. The function is expected to return a
        dictionary whose keys are raw entities and whose values
        are their occurrence frequencies.
        """
        self._aggregated_entities_callback = func

    def set_entity_definitons(self, definition_dict):
        """Set the lookup dictionary for entity definitions."""
        self._entity_definitions = definition_dict

    def export_graphs(self, graph_list):
        """Export current graph objects from the app.

        Parameters
        ----------
        graph_list : list of str
            List of graph identifiers to export

        Returns
        -------
        graphs : dict
            Dictionary whose keys represent the input list of
            graph identifiers to export. Eeach graph is given by a
            dictionary with two keys: `graph` giving the full graph
            object and `tree` giving the minimum spanning tree object.

        """
        graphs = {}
        for g in graph_list:
            if g in self._graphs:
                graphs[g] = {}
                processor = self._graph_processor.from_graph_object(
                    self._graphs[g]["object"])
                graphs[g]["graph"] = processor.get_pgframe(
                    node_prop_types=CORD19_PROP_TYPES["nodes"],
                    edge_prop_types=CORD19_PROP_TYPES["edges"]).to_json()
                if "full_tree" in self._graphs[g]:
                    tree_processor = self._graph_processor.from_graph_object(
                        self._graphs[g]["full_tree"])
                    graphs[g]["tree"] = tree_processor.get_pgframe(
                        node_prop_types=CORD19_PROP_TYPES["nodes"],
                        edge_prop_types=CORD19_PROP_TYPES["edges"]).to_json()
        return graphs

    def get_edit_history(self):
        """Export the history of graph edits."""
        return self._edit_history


visualization_app = VisualizationApp()


# ############################## CALLBACKS ####################################

def search(elements, search_value, value, showgraph, diffs=None,
           cluster_type=None, cluster_search=None, nodes_to_keep=None,
           global_scope=False):
    res = []

    if nodes_to_keep is None:
        nodes_to_keep = []

    if diffs is None:
        diffs = []

    if value is None:
        value = []

    if global_scope:
        graph_object = visualization_app._graphs[showgraph]["object"]
        processor = visualization_app._graph_processor.from_graph_object(
            graph_object)
        for n, attrs in processor.nodes(properties=True):
            cluster_matches = False
            if cluster_type is not None and cluster_search is not None:
                if (
                        cluster_type in attrs and
                        attrs[cluster_type] in cluster_search) or\
                   n in nodes_to_keep:
                    cluster_matches = True
            else:
                cluster_matches = True
            # Check if the name matches
            name_matches = False
            if (search_value in n) or (n in search_value) or n in (value or []):
                if n not in diffs:
                    name_matches = True

            if cluster_matches and name_matches:
                res.append({"label": n, "value": n})
    else:
        for ele_data in elements:
            if 'source' not in ele_data['data']:
                el_id = ele_data["data"]["id"]

                cluster_matches = False
                if cluster_type is not None and cluster_search is not None:
                    if (cluster_type in ele_data["data"] and
                        ele_data["data"][cluster_type] in cluster_search) or\
                       el_id in nodes_to_keep:
                        cluster_matches = True
                else:
                    cluster_matches = True

                # Check if the name matches
                name_matches = False

                if (search_value in el_id) or (el_id in search_value) or\
                   (el_id in value):
                    if el_id not in diffs:
                        name_matches = True

                if cluster_matches and name_matches:
                    res.append({"label": el_id, "value": el_id})

    return res


def generate_mark(val):
    return (
        "{}".format(val)
        if isinstance(val, int) else
        "{:.2f}".format(val)
    )


def recompute_node_range(elements, freq_type):
    all_weights = [
        el["data"][freq_type]
        for el in elements
        if freq_type in el["data"]
    ]
    if len(all_weights) > 0:
        min_value = min(all_weights)
        min_value = (
            min_value
            if isinstance(min_value, int)
            else math.floor(min_value)
        )
        max_value = max(all_weights)
        max_value = (
            max_value
            if isinstance(max_value, int)
            else math.ceil(max_value)
        )
    else:
        min_value = 0
        max_value = 0

    marks = {
        min_value: generate_mark(min_value),
        max_value: generate_mark(max_value)
    }

    step = (max_value - min_value) / 100

    return min_value, max_value, marks, step


def filter_elements(input_elements, node_condition, edge_condition=None):
    # filter graph elements by applyting specified node and edge conditions
    nodes_to_keep = [
        el["data"]["id"]
        for el in input_elements
        if "source" not in el["data"] and node_condition(el["data"])
    ]

    edges_to_keep = [
        el["data"]["id"]
        for el in input_elements
        if "source" in el["data"] and (
            el["data"]["source"] in nodes_to_keep and
            el["data"]["target"] in nodes_to_keep and
            (
                edge_condition(el["data"])
                if edge_condition is not None else True
            )
        )
    ]

    elements = [
        el for el in input_elements
        if el["data"]["id"] in nodes_to_keep + edges_to_keep
    ]

    hidden_elements = [
        el for el in input_elements
        if el["data"]["id"] not in nodes_to_keep + edges_to_keep
    ]
    return elements, hidden_elements


def get_all_clusters(processor, cluster_type):
    return list(set([
        properties[cluster_type]
        for n, properties in processor.nodes(properties=True)
        if cluster_type in properties
    ]))


@visualization_app._app.callback(
    [
        Output('nodefreqslider', 'children'),
        Output('edgefreqslider', 'children')
    ],
    [
        Input('node_freq_type', 'value'),
        Input('edge_freq_type', 'value'),
        Input('cytoscape', 'elements')
    ], [
        State('showgraph', 'value')
    ])
def adapt_weight_ranges(node_freq_type, edge_freq_type, elements, val):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "node_freq_type" or button_id == "edge_freq_type":
        visualization_app._update_weight_data(
            val, elements, node_freq_type, edge_freq_type)

    node_freq_slider = dcc.RangeSlider(
        id="nodefreqslider_content",
        min=visualization_app._graphs[val]["min_node_weight"],
        max=visualization_app._graphs[val]["max_node_weight"],
        value=visualization_app._graphs[val]["current_node_value"],
        step=visualization_app._graphs[val]["node_step"],
        marks=visualization_app._graphs[val]["node_marks"]
    )

    edge_freq_slider = dcc.RangeSlider(
        id="edgefreqslider_content",
        min=visualization_app._graphs[val]["min_edge_weight"],
        max=visualization_app._graphs[val]["max_edge_weight"],
        value=visualization_app._graphs[val]["current_edge_value"],
        step=visualization_app._graphs[val]["edge_step"],
        marks=visualization_app._graphs[val]["edge_marks"]
    )

    return [node_freq_slider, edge_freq_slider]


@visualization_app._app.callback(
    [
        Output('searchnodetotraverse', 'disabled'),
        Output('searchpathoverlap', 'options'),
        Output('pathdepth', 'disabled'),
    ],
    [
        Input('nestedpaths', 'value')
    ])
def setup_paths_tab(nestedpaths):
    if len(nestedpaths) > 0:
        # disable traverse and overlapping
        traverse_field_disable = True
        overlapping_disable = [{"value": 1, "disabled": True}]
        # enable depth
        pathdepth_disable = False
    else:
        # enable traverse and overlapping
        traverse_field_disable = False
        overlapping_disable = [{"value": 1}]
        # disable depth
        pathdepth_disable = True
    return [traverse_field_disable, overlapping_disable, pathdepth_disable]


def _handle_paths_search(graph_object, elements, source, target, top_n,
                         searchnodetotraverse, searchpathoverlap,
                         nestedpaths, pathdepth):
    success = False
    no_path_message = ""
    path_finder = BACKEND_MAPPING[visualization_app._backend][
        "paths"].from_graph_object(graph_object)
    try:
        if nestedpaths and pathdepth:
            paths = path_finder.n_nested_shortest_paths(
                source, target, top_n, nested_n=top_n,
                strategy="naive", distance="distance_npmi",
                depth=pathdepth, exclude_edge=True)
        elif searchnodetotraverse:
            intersecting = len(searchpathoverlap) == 1
            a_b_paths, b_c_paths = path_finder.n_shortest_tripaths(
                source, searchnodetotraverse, target, top_n,
                strategy="naive", distance="distance_npmi",
                overlap=intersecting, exclude_edge=True)
            paths = a_b_paths + b_c_paths
        else:
            paths = path_finder.n_shortest_paths(
                source, target, top_n,
                distance="distance_npmi", strategy="naive",
                exclude_edge=True)
        elements = []
        path_graph = path_finder.get_subgraph_from_paths(paths)
        elements = get_cytoscape_data(
            visualization_app._graph_processor.from_graph_object(
                path_graph))
        visualization_app._current_layout = "klay"
        success = True

    except ValueError as e:
        print(e)
        no_path_message = (
            "No undirect paths from '{}' to '{}' were ".format(
                source, target) +
            "found (the nodes are either disconnected or connected by "
            "a direct edge only)"
        )
    return success, elements, no_path_message


def _handle_expand_edge(graph_object, source, target, edge_expansion_limit):
    path_finder = BACKEND_MAPPING[visualization_app._backend][
        "paths"].from_graph_object(graph_object)
    paths = path_finder.n_shortest_paths(
        source, target,
        edge_expansion_limit,
        distance="distance_npmi", strategy="naive", exclude_edge=True)

    path_graph = path_finder.get_subgraph_from_paths(paths)
    new_elements = get_cytoscape_data(
        visualization_app._graph_processor.from_graph_object(path_graph))
    return new_elements


def _handle_neighbor_search(processor, selected_node, neighborlimit):
    weights = {}
    for n in processor.neighbors(selected_node):
        weights[n] = processor.get_edge(selected_node, n)["npmi"]
    top_neighbors = top_n(weights, neighborlimit)

    list_group = dbc.ListGroup([
        dbc.ListGroupItem("{}. {} (NMPI {:.2f})".format(i + 1, n, weights[n]))
        for i, n in enumerate(top_neighbors)
    ])
    return [
        html.H6("Top neighbors of '{}' by NMPI".format(
            selected_node), className="card-title"),
        list_group
    ]


def _handle_reset_elements(current_graph, editing_mode,
                           node_freq_type, edge_freq_type,
                           nodes_to_keep, memory):
    if editing_mode == 1:
        visualization_app._graphs[current_graph]["object"] =\
            visualization_app._graphs[current_graph]["object_backup"].copy()
        processor = visualization_app._graph_processor.from_graph_object(
            visualization_app._graphs[current_graph]["object_backup"])
        elements = visualization_app._update_cyto_graph(
            current_graph, processor,
            visualization_app._graphs[current_graph]["top_n"],
            node_freq_type=node_freq_type,
            edge_freq_type=edge_freq_type,
            nodes_to_keep=nodes_to_keep)
    else:
        # Remove added elements
        elements = [
            el for el in elements
            if el["data"]["id"] not in memory[
                "merging_backup"]["added_elements"]
        ]
        for el in memory["merging_backup"]["added_elements"]:
            if el in visualization_app._graphs[current_graph]["paper_lookup"]:
                del visualization_app._graphs[current_graph]["paper_lookup"][el]

        # Relabel nodes back
        for el in elements:
            if el["data"]["id"] in memory["renamed_elements"]:
                el["data"]["name"] = el["data"]["id"]
                el["data"]["value"] = el["data"]["id"]

        # Bring back removed elements
        for el in memory["removed_elements"].values():
            elements.append(el)

        for el in memory["merging_backup"]["removed_elements"].values():
            elements.append(el)

        # Bring back paper lookup
        visualization_app._graphs[current_graph]["paper_lookup"].update(
            memory["paper_backup"])

        memory["removed_elements"] = {}
        memory["merged_elements"] = {}
        memory["renamed_elements"] = {}
    return elements


def _handle_remove_node(current_graph, processor, elements, memory,
                        editing_mode, selected_node_data, selected_edge_data,
                        node_freq_type, nodes_to_keep):
    nodes_to_remove = {}
    edges_to_remove = {}
    if selected_node_data:
        nodes_to_remove = {
            el["id"]: {"data": el} for el in selected_node_data
        }

        edges_to_remove = {
            el["data"]["id"]: el
            for el in elements
            if "source" in el["data"] and (
                el["data"]["source"] in nodes_to_remove or
                el["data"]["target"] in nodes_to_remove
            )
        }
    if selected_edge_data:
        edges_to_remove = {
            ele_data["id"]: {"data": ele_data}
            for ele_data in selected_edge_data
        }

    if editing_mode == 1:
        for n in nodes_to_remove:
            processor.remove_node(n)

        for edge in edges_to_remove.values():
            if (edge["data"]["source"], edge["data"]["target"]) in processor.edges():
                processor.remove_edge(
                    edge["data"]["source"], edge["data"]["target"])

        if current_graph not in visualization_app._edit_history:
            visualization_app._edit_history[current_graph] = []
        visualization_app._edit_history[current_graph].append(
            {
                "type": "remove",
                "nodes": list(nodes_to_remove.keys()),
                "edges": list(edges_to_remove.keys())
            }
        )

        elements = visualization_app._update_cyto_graph(
            current_graph, processor,
            visualization_app._graphs[current_graph]["top_n"],
            node_freq_type=node_freq_type, edge_freq_type=node_freq_type,
            nodes_to_keep=nodes_to_keep)
    else:
        memory["removed_elements"].update(nodes_to_remove)
        memory["removed_elements"].update(edges_to_remove)
    return elements


def _handle_rename_node(current_graph, processor, memory, elements,
                        selected_node, rename_input_value, editing_mode,
                        node_freq_type, nodes_to_keep):
    rename_error_message = ""
    # Check if the rename input is valid
    rename_invalid = False
    if rename_input_value != selected_node:
        if editing_mode == 1:
            if rename_input_value in processor.nodes():
                rename_invalid = True
                rename_error_message =\
                    "Node with the label '{}' already exists".format(
                        rename_input_value)
        else:
            for el in elements:
                if rename_input_value == el["data"]["id"]:
                    rename_invalid = True
                    rename_error_message =\
                        "Node with the label '{}' already exists".format(
                            rename_input_value)
                    break

    if not rename_invalid:
        if editing_mode == 1:
            # Rename node in the graph
            processor.rename_nodes(
                {selected_node: rename_input_value})
            elements = visualization_app._update_cyto_graph(
                current_graph, processor,
                visualization_app._graphs[current_graph]["top_n"],
                node_freq_type=node_freq_type,
                edge_freq_type=node_freq_type,
                nodes_to_keep=nodes_to_keep)

            visualization_app._graphs[current_graph]["paper_lookup"][
                rename_input_value] = visualization_app._graphs[
                    current_graph]["paper_lookup"][selected_node]
            visualization_app._entity_definitions[rename_input_value] =\
                visualization_app._entity_definitions[selected_node]

            if current_graph not in visualization_app._edit_history:
                visualization_app._edit_history[current_graph] = []
            visualization_app._edit_history[current_graph].append(
                {
                    "type": "rename",
                    "original_node": selected_node,
                    "new_name": rename_input_value
                }
            )
        else:
            memory["renamed_elements"].update(
                {selected_node: rename_input_value})
    return elements, rename_error_message


def _handle_merge_nodes(current_graph, processor, memory,
                        selected_nodes, new_name, editing_mode,
                        node_freq_type, nodes_to_keep):
    if editing_mode == 1:
        if new_name not in visualization_app._entity_definitions:
            # choose a random definiton
            definition = ""
            for n in selected_nodes:
                if n in visualization_app._entity_definitions:
                    definition = visualization_app._entity_definitions[n]
                    break
            visualization_app._entity_definitions[new_name] = definition

        new_graph = merge_nodes(
            processor,
            list(selected_nodes),
            new_name, ATTRS_RESOLVER)
        new_graph_processor =\
            visualization_app._graph_processor.from_graph_object(new_graph)

        elements = visualization_app._update_cyto_graph(
            current_graph, new_graph_processor,
            visualization_app._graphs[current_graph]["top_n"],
            node_freq_type=node_freq_type, edge_freq_type=node_freq_type,
            nodes_to_keep=nodes_to_keep)

        if current_graph not in visualization_app._edit_history:
            visualization_app._edit_history[current_graph] = []
        visualization_app._edit_history[current_graph].append(
            {
                "type": "merge",
                "target": new_name,
                "merged_nodes": list(selected_nodes)
            }
        )
        visualization_app._graphs[current_graph]["paper_lookup"][new_name] =\
            new_graph_processor.get_node(new_name)["paper"]
    else:
        elements, target_node, merging_data = merge_cyto_elements(
            elements, selected_nodes, new_name)

        for k, v in merging_data["removed_elements"].items():
            if k not in memory["merging_backup"]["added_elements"]:
                memory["merging_backup"]["removed_elements"][k] = v

        memory["merging_backup"]["added_elements"] += merging_data[
            "added_elements"]

        papers = set()
        for n in selected_nodes:
            memory["paper_backup"][n] = visualization_app._graphs[
                current_graph]["paper_lookup"][n]
            papers.update(
                visualization_app._graphs[current_graph]["paper_lookup"][n])

        visualization_app._graphs[current_graph][
            "paper_lookup"][new_name] = list(papers)
    return elements


def _handle_show_top_n(current_graph, processor, elements, memory,
                       top_n_slider_value, recompute_spanning_tree,
                       node_freq_type, edge_freq_type, nodes_with_cluster,
                       nodes_to_keep):
    current_top_n = visualization_app._graphs[current_graph]["top_n"]
    total_number_of_entities = len(processor.nodes())
    if current_top_n is None or current_top_n != top_n_slider_value:
        if len(recompute_spanning_tree) > 0:
            elements = visualization_app._update_cyto_graph(
                current_graph, processor,
                top_n_slider_value,
                node_freq_type=node_freq_type,
                edge_freq_type=edge_freq_type,
                node_subset=nodes_with_cluster,
                nodes_to_keep=nodes_to_keep)
        else:
            top_n_nodes = get_top_n_nodes(
                visualization_app._graphs[current_graph]["object"],
                top_n_slider_value,
            )

            elements, hidden_elements = filter_elements(
                elements,
                lambda x:
                    x["id"] in top_n_nodes or x["id"] in nodes_to_keep,
                lambda x:
                    (
                        x["source"] in top_n_nodes or
                        x["source"] in nodes_to_keep
                    ) and (
                        x["target"] in top_n_nodes or
                        x["target"] in nodes_to_keep
                    )
            )
            memory["filtered_elements"] = hidden_elements

    message = "Displaying top {} most frequent entities (out of {})".format(
        top_n_slider_value
        if top_n_slider_value <= total_number_of_entities
        else total_number_of_entities,
        total_number_of_entities)
    return elements, message


def _handle_show_all(current_graph, processor, node_freq_type, edge_freq_type,
                     nodes_with_cluster, nodes_to_keep):
    total_number_of_entities = len(processor.nodes())
    elements = visualization_app._update_cyto_graph(
        current_graph, processor, None,
        node_freq_type=node_freq_type, edge_freq_type=edge_freq_type,
        node_subset=nodes_with_cluster, nodes_to_keep=nodes_to_keep)

    message = "Displaying all {} entities".format(total_number_of_entities)
    return elements, message


def node_range_condition(el, start, end, nodes_to_keep, node_freq_type):
    if el["id"] in nodes_to_keep:
        return True
    if node_freq_type in el:
        if el[node_freq_type] >= start and\
           el[node_freq_type] <= end:
            return True
    return False


def edge_range_condition(el, start, end, edge_freq_type):
    if edge_freq_type in el:
        if el[edge_freq_type] >= start and\
           el[edge_freq_type] <= end:
            return True
    return False


def _handle_node_frequency_filer(current_graph, elements, memory,
                                 nodefreqslider, nodes_to_keep,
                                 node_freq_type, edge_freq_type):
    elements, hidden_elements = filter_elements(
        elements + memory["filtered_elements"],
        lambda x: node_range_condition(
            x, nodefreqslider[0], nodefreqslider[1],
            nodes_to_keep, node_freq_type),
        lambda x: edge_range_condition(
            x,
            visualization_app._graphs[current_graph]["current_edge_value"][0],
            visualization_app._graphs[current_graph]["current_edge_value"][1],
            edge_freq_type))
    visualization_app._graphs[current_graph]["current_node_value"] =\
        nodefreqslider

    memory["filtered_elements"] = hidden_elements

    new_marks = {}
    for k, v in visualization_app._graphs[current_graph]["node_marks"].items():
        if k == visualization_app._graphs[current_graph]["min_node_weight"] or\
           k == visualization_app._graphs[current_graph]["max_node_weight"]:
            new_marks[k] = v

    # add a new value mark
    if nodefreqslider[0] != visualization_app._graphs[current_graph][
            "min_node_weight"]:
        new_marks[nodefreqslider[0]] = generate_mark(nodefreqslider[0])
    if nodefreqslider[1] != visualization_app._graphs[current_graph][
            "max_node_weight"]:
        new_marks[nodefreqslider[1]] = generate_mark(nodefreqslider[1])

    visualization_app._graphs[current_graph]["node_marks"] = new_marks
    return elements


def _handle_edge_frequency_filter(current_graph, elements, memory,
                                  edgefreqslider, nodes_to_keep, node_freq_type,
                                  edge_freq_type):
    elements, hidden_elements = filter_elements(
        elements + memory["filtered_elements"],
        lambda x: node_range_condition(
            x,
            visualization_app._graphs[current_graph]["current_node_value"][0],
            visualization_app._graphs[current_graph]["current_node_value"][1],
            nodes_to_keep, node_freq_type),
        lambda x: edge_range_condition(
            x, edgefreqslider[0], edgefreqslider[1], edge_freq_type))
    visualization_app._graphs[current_graph]["current_edge_value"] =\
        edgefreqslider

    memory["filtered_elements"] = hidden_elements

    new_marks = {}
    for k, v in visualization_app._graphs[current_graph]["edge_marks"].items():
        if k == visualization_app._graphs[current_graph]["min_edge_weight"] or\
           k == visualization_app._graphs[current_graph]["max_edge_weight"]:
            new_marks[k] = v

    # add a new value mark
    if edgefreqslider[0] != visualization_app._graphs[current_graph][
            "min_edge_weight"]:
        new_marks[edgefreqslider[0]] = generate_mark(edgefreqslider[0])
    if edgefreqslider[1] != visualization_app._graphs[current_graph][
            "max_edge_weight"]:
        new_marks[edgefreqslider[1]] = generate_mark(edgefreqslider[1])

    visualization_app._graphs[current_graph]["edge_marks"] = new_marks
    return elements


@visualization_app._app.callback(
    [
        Output("memory", "data"),
        Output('dropdown-layout', 'value'),
        Output('cytoscape', 'zoom'),
        Output('cytoscape', 'elements'),
        Output('display-message', 'children'),
        Output("noPathMessage", "children"),
        Output("merge-button", "disabled"),
        Output("merge-modal", "is_open"),
        Output("rename-modal", "is_open"),
        Output("remove-button", "disabled"),
        Output("rename-button", "disabled"),
        Output("rename-input", "value"),
        Output("rename-input", "invalid"),
        Output("rename-error-message", "children"),
        Output('nodefreqslider_content', 'marks'),
        Output('edgefreqslider_content', 'marks'),
        Output("edit-mode", "value"),
        Output("bt-neighbors", "disabled"),
        Output("neighbors-card-body", "children"),
        Output("bt-expand-edge", "disabled")
    ],
    [
        Input('bt-reset', 'n_clicks'),
        Input('remove-button', 'n_clicks'),
        Input('showgraph', 'value'),
        Input('nodefreqslider_content', 'value'),
        Input('edgefreqslider_content', 'value'),
        Input("searchdropdown", "value"),
        Input('bt-path', 'n_clicks'),
        Input('groupedLayout', "value"),
        Input('cluster_type', "value"),
        Input('top-n-button', "n_clicks"),
        Input('show-all-button', "n_clicks"),
        Input("clustersearch", "value"),
        Input("nodestokeep", "value"),
        Input('cytoscape', 'selectedNodeData'),
        Input('cytoscape', 'selectedEdgeData'),
        Input('cytoscape', 'tapNodeData'),
        Input("merge-button", "n_clicks"),
        Input("merge-close", "n_clicks"),
        Input("merge-apply", "n_clicks"),
        Input('reset-elements-button', "n_clicks"),
        Input("rename-button", "n_clicks"),
        Input("rename-close", "n_clicks"),
        Input("rename-apply", "n_clicks"),
        Input("bt-neighbors", "n_clicks"),
        Input("bt-expand-edge", "n_clicks"),
    ],
    [
        State('node_freq_type', 'value'),
        State('edge_freq_type', 'value'),
        State('cytoscape', 'elements'),
        State('cytoscape', 'zoom'),
        State('searchpathfrom', 'value'),
        State('searchpathto', 'value'),
        State('searchnodetotraverse', 'value'),
        State('searchpathlimit', 'value'),
        State('searchpathoverlap', 'value'),
        State('nestedpaths', 'value'),
        State('pathdepth', 'value'),
        State('top-n-slider', 'value'),
        State('dropdown-layout', 'value'),
        State("merge-modal", "is_open"),
        State("rename-modal", "is_open"),
        State("memory", "data"),
        State("merge-label-input", "value"),
        State("recompute-spanning-tree", "value"),
        State("rename-input", "value"),
        State("edit-mode", "value"),
        State("neighborlimit", "value"),
        State("expand-edge-n", "value"),
        State("graph-view-tab", "className"),
        State("layout-tab", "className"),
        State("path-finder-tab", "className"),
    ]
)
def update_cytoscape_elements(resetbt, removebt, val,
                              nodefreqslider, edgefreqslider,
                              searchvalues, pathbt,
                              grouped_layout,
                              cluster_type, top_n_buttton, show_all_button, clustersearch,
                              nodes_to_keep, selected_node_data, selected_edge_data, tappednode,
                              merge_button, close_merge_button, apply_merge_button, reset_graph_button, rename_button,
                              rename_apply_button, rename_close_button, bt_neighbors, bt_expand,
                              # states
                              node_freq_type, edge_freq_type, cytoelements, zoom, searchpathfrom,
                              searchpathto, searchnodetotraverse, searchpathlimit, searchpathoverlap,
                              nestedpaths, pathdepth, top_n_slider_value, dropdown_layout,
                              open_merge_modal, open_rename_modal, memory, merge_label_value, recompute_spanning_tree,
                              selected_rename_input, editing_mode, neighborlimit, edge_expansion_limit,
                              graph_view_tab, layout_tab, path_tab):
    zoom = 1

    # Get the event that trigerred the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    full_graph_events = [
        "showgraph",
        "clustersearch",
        "top-n-button",
        "show-all-button",
        "bt-reset",
        "nodestokeep",
        "groupedLayout"
    ]

    if editing_mode == 1:
        full_graph_events += [
            "reset-elements-button",
            "remove-button",
            "merge-apply",
            "rename-apply"
        ]

    if button_id in full_graph_events:
        elements = visualization_app._graphs[val]["cytoscape"]
    else:
        elements = cytoelements

    processor = visualization_app._graph_processor.from_graph_object(
        visualization_app._graphs[val]["object"])

    output_editing_mode = editing_mode
    if button_id in full_graph_events:
        output_editing_mode = 1

    # -------- Filter elements by selected clusters to display --------
    if nodes_to_keep is None:
        nodes_to_keep = []

    nodes_with_cluster = filter_nodes_by_attr(
        processor, cluster_type, clustersearch)

    nodes_with_cluster += nodes_to_keep

    element_preserving = [
        "cytoscape",
        "edgefreqslider_content",
        "nodefreqslider_content",
        "searchdropdown",
        "rename-button",
        "rename-close",
        "merge-button",
        "merge-close",
        "bt-neighbors",
        "bt-expand-edge"
    ]

    if editing_mode == 2:
        element_preserving += [
            "reset-elements-button",
            "remove-button",
            "merge-apply",
            "rename-apply"
        ]

    if clustersearch is not None and cluster_type is not None and\
       button_id != "bt-reset" and button_id not in element_preserving:
        memory["filtered_elements"] = []
        memory["removed_elements"] = {}
        memory["renamed_elements"] = {}
        memory["merged_elements"] = {}
        if button_id == "clustersearch" and visualization_app._is_initializing:
            # If app is in the initialization stage, then we fix
            # some of the parameters from the loaded configs
            configs = visualization_app._configs
            elements = configs["elements"] if "elements" in configs else []
            if "current_graph" in configs and "nodefreqslider" in configs:
                visualization_app._graphs[configs["current_graph"]][
                    "current_node_value"] = configs["nodefreqslider"]
            if "current_graph" in configs and "edgefreqslider" in configs:
                visualization_app._graphs[configs["current_graph"]][
                    "current_edge_value"] = configs["edgefreqslider"]
        else:
            if len(recompute_spanning_tree) > 0:
                elements = visualization_app._update_cyto_graph(
                    val, processor,
                    visualization_app._graphs[val]["top_n"],
                    node_freq_type=node_freq_type,
                    edge_freq_type=edge_freq_type,
                    node_subset=nodes_with_cluster,
                    nodes_to_keep=nodes_to_keep)
            else:
                elements, hidden_elements = filter_elements(
                    elements,
                    node_condition=lambda x: (
                        cluster_type in x and x[cluster_type] in clustersearch
                    ) or x["id"] in nodes_to_keep
                )
                memory["filtered_elements"] = hidden_elements

    # -------- Handle node/edge selection --------

    if selected_edge_data is None:
        selected_edge_data = []

    # Mark selected nodes and edges
    selected_nodes = [
        el["id"] for el in (selected_node_data if selected_node_data else [])
    ]

    if searchvalues is None:
        searchvalues = []
    else:
        searchvalues = [searchvalues]

    merge_disabled = True
    remove_disabled = True
    rename_disabled = True
    rename_input_value = ""
    rename_invalid = False
    rename_error_message = None
    neighbor_bt_disabled = True
    expand_edge_disabled = True

    if len(selected_nodes) > 1:
        merge_disabled = False
    if len(selected_nodes) > 0 or len(selected_edge_data) > 0:
        remove_disabled = False
    if len(selected_nodes) == 1:
        rename_disabled = False
        rename_input_value = selected_nodes[0]
        neighbor_bt_disabled = False
    if len(selected_edge_data) == 1:
        expand_edge_disabled = False

    # -------- Handle grouped layout --------

    if button_id == "cluster_type":
        if len(grouped_layout) == 1:
            elements = generate_clusters(elements, cluster_type)
    if button_id == "groupedLayout":
        if len(grouped_layout) == 1:
            elements = generate_clusters(elements, cluster_type)
        else:
            elements = visualization_app._graphs[val]["cytoscape"]

    # -------- Handle remove selected nodes -------

    if button_id == "remove-button":
        elements = _handle_remove_node(
            val, processor, elements, memory,
            editing_mode, selected_node_data, selected_edge_data,
            node_freq_type, nodes_to_keep)

    # -------- Handle merge selected nodes -------

    # Open/close the merge dialog
    if button_id == "merge-button" or button_id == "merge-close" or\
       button_id == "merge-apply":
        open_merge_modal = not open_merge_modal

    # Merge underlying nx_object and tree if applicable
    if button_id == "merge-apply" and merge_label_value:
        # Retreive name
        new_name = merge_label_value
        elements = _handle_merge_nodes(
            val, processor, memory,
            selected_nodes, new_name, editing_mode,
            node_freq_type, nodes_to_keep)

    if button_id == "rename-apply" and len(selected_nodes) == 1:
        if selected_rename_input:
            rename_input_value = selected_rename_input
        elements, rename_error_message = _handle_rename_node(
            val, processor, memory, elements, selected_nodes[0],
            rename_input_value, editing_mode,
            node_freq_type, nodes_to_keep)

    # Open/close the rename dialog
    if button_id == "rename-button" or button_id == "rename-close" or\
       (button_id == "rename-apply" and not rename_invalid):
        open_rename_modal = not open_rename_modal

    # -------- Handle reset graph elements --------
    if button_id == "reset-elements-button":
        elements = _handle_reset_elements(
            val, editing_mode, node_freq_type, edge_freq_type,
            nodes_to_keep, memory)

    # --------- Handle neighbor search -----
    neighbor_card_content = [
        html.H6("No neighbors to display", className="card-title")
    ]
    if button_id == "bt-neighbors":
        neighbor_card_content = _handle_neighbor_search(
            processor, selected_nodes[0], neighborlimit)

    # -------- Handle path search -------
    no_path_message = ""
    if button_id == "bt-path" and pathbt is not None:
        memory["removed_nodes"] = []
        memory["removed_edges"] = []
        memory["filtered_elements"] = []

        output_editing_mode = 2

        success = False
        if searchpathfrom and searchpathto:
            top_n = searchpathlimit if searchpathlimit else 20

            source = searchpathfrom
            target = searchpathto

            # create a subgraph given the selected clusters
            graph_object = subgraph_from_clusters(
                processor, cluster_type, clustersearch, nodes_to_keep)

            success, elements, no_path_message = _handle_paths_search(
                graph_object, elements, source, target, top_n,
                searchnodetotraverse, searchpathoverlap,
                nestedpaths, pathdepth)
        if not success:
            if visualization_app._graphs[val]["top_n"] is None and\
               visualization_app._graphs[val]["positions"] is not None:
                visualization_app._current_layout = "preset"
            else:
                visualization_app._current_layout = "cose-bilkent"

    # -------- Handle edge expansion --------------
    if button_id == "bt-expand-edge" and edge_expansion_limit:
        source = selected_edge_data[0]["source"]
        target = selected_edge_data[0]["target"]

        graph_object = subgraph_from_clusters(
            processor,
            cluster_type, clustersearch, nodes_to_keep)

        new_elements = _handle_expand_edge(
            graph_object, source, target, edge_expansion_limit)

        existing_nodes = [
            el["data"]["id"] for el in elements if "source" not in el["data"]
        ]

        existing_edges = [
            (el["data"]["source"], el["data"]["target"])
            for el in elements if "source" in el["data"]
        ]

        for el in new_elements:
            if "source" not in el["data"]:
                if el["data"]["id"] not in existing_nodes:
                    elements.append(el)
            else:
                if (
                    el["data"]["source"], el["data"]["target"]
                ) not in existing_edges and (
                    el["data"]["target"], el["data"]["source"]
                ) not in existing_edges:
                    elements.append(el)

        visualization_app._current_layout = "klay"

    # -------- Handle 'display a spanning tree on top N nodes' -------
    total_number_of_entities = len(processor.nodes())
    message = (
        "Displaying top {} most frequent entities (out of {})".format(
            visualization_app._graphs[val]["top_n"], total_number_of_entities)
        if visualization_app._graphs[val]["top_n"] is not None
        else "Displaying all {} entities".format(total_number_of_entities)
    )

    # ---- Handle changes in Display Top N or all the entities -------
    if button_id == "top-n-button":
        elements, message = _handle_show_top_n(
            val, processor, elements, memory,
            top_n_slider_value, recompute_spanning_tree,
            node_freq_type, edge_freq_type, nodes_with_cluster,
            nodes_to_keep)
    elif button_id == "show-all-button":
        # Top entities are selected, but button is clicked, so show all
        elements, message = _handle_show_all(
            val, processor, node_freq_type, edge_freq_type,
            nodes_with_cluster, nodes_to_keep)

    # -------- Handle node/edge weight sliders -------

    if nodefreqslider and button_id == "nodefreqslider_content":
        if nodefreqslider[0] != visualization_app._graphs[val][
                "current_node_value"][0] or\
           nodefreqslider[1] != visualization_app._graphs[val][
                "current_node_value"][1]:
            elements = _handle_node_frequency_filer(
                val, elements, memory, nodefreqslider,
                nodes_to_keep, node_freq_type, edge_freq_type)

    elif edgefreqslider and button_id == "edgefreqslider_content":
        if edgefreqslider[0] != visualization_app._graphs[val][
                "current_edge_value"][0] or\
           edgefreqslider[1] != visualization_app._graphs[val][
                "current_edge_value"][1]:
            elements = _handle_edge_frequency_filter(
                val, elements, memory, edgefreqslider, nodes_to_keep,
                node_freq_type, edge_freq_type)

    # -------- Apply filters and masking operations from memory -------
    new_elements = []
    for el in elements:
        el_id = el["data"]["id"]
        if el_id not in memory["removed_elements"] and\
           el_id not in memory["filtered_elements"]:
            # the node element was renamed
            if el_id in memory["renamed_elements"]:
                el["data"]["value"] = memory["renamed_elements"][el_id]
                el["data"]["name"] = memory["renamed_elements"][el_id]

            new_elements.append(el)

    elements = new_elements

    # -------- Automatically switch the layout -------
    if visualization_app._is_initializing:
        visualization_app._current_layout = visualization_app._configs[
            "current_layout"]
        visualization_app._is_initializing = False
    else:
        if button_id in full_graph_events:
            if visualization_app._graphs[val]["top_n"] is None and\
               visualization_app._graphs[val]["positions"] is not None:
                visualization_app._current_layout = "preset"
            else:
                visualization_app._current_layout = components.DEFAULT_LAYOUT
        else:
            if button_id == "bt-path":
                visualization_app._current_layout = "klay"

    node_marks = visualization_app._graphs[val]["node_marks"]
    edge_marks = visualization_app._graphs[val]["edge_marks"]

    # ------- Update configs ---------
    visualization_app._update_configs(
        elements, val, nodes_to_keep, top_n_slider_value,
        node_freq_type, edge_freq_type,
        nodefreqslider, edgefreqslider, cluster_type, clustersearch,
        searchpathfrom, searchpathto, searchnodetotraverse, searchpathlimit,
        searchpathoverlap, nestedpaths, pathdepth)

    return [
        memory,
        visualization_app._current_layout,
        zoom,
        elements,
        message,
        no_path_message,
        merge_disabled,
        open_merge_modal,
        open_rename_modal,
        remove_disabled,
        rename_disabled,
        rename_input_value,
        rename_invalid,
        rename_error_message,
        node_marks,
        edge_marks,
        output_editing_mode,
        neighbor_bt_disabled,
        neighbor_card_content,
        expand_edge_disabled
    ]


@visualization_app._app.callback(
    [
        Output('item-card-body', 'children')
    ],
    [
        Input('cytoscape', 'tapNode'),
        Input('cytoscape', 'tapEdge')
    ],
    [
        State('cytoscape', 'selectedNodeData'),
        State('cytoscape', 'selectedEdgeData'),
        State('showgraph', 'value'),
        State('memory', 'data')
    ])
def display_tap_node(datanode, dataedge, statedatanode, statedataedge,
                     showgraph, memory):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[1]

    papers = []
    res = []
    npmi_message = None

    paper_lookup = visualization_app._graphs[
        showgraph]["paper_lookup"]

    if button_id == "tapNode" and datanode:
        definition = ""
        if "definition" in datanode['data']:
            definition = str(datanode["data"]["definition"])
        elif datanode["data"]["id"] in visualization_app._entity_definitions:
            definition = visualization_app._entity_definitions[
                datanode["data"]["id"]]
        label = datanode['data']['name']
        try:
            _type = str(datanode['data']['entity_type'])
        except Exception:
            pass

        frequency = str(datanode['data']['paper_frequency'])

        res.append([
            html.H5(label, className="card-title"),
            html.H6(_type, className="card-subtitle"),
            html.P(
                definition,
                className="card-text"
            )
        ])
        label = "'" + label + "' mentioned in " + frequency + " papers "
        modal_buttons = [
            dbc.Button(label, id="open-body-scroll", color="primary"),
            dbc.Button(
                "Inspect linked raw entites", id="show-aggregated-entities",
                color="primary", style={"margin-left": "20pt"})
        ]
        if datanode['data']['id'] in paper_lookup:
            papers = paper_lookup[datanode['data']['id']]

    if button_id == "tapEdge" and dataedge:
        label = str(dataedge['style']['label'])
        papers = list(
            set(paper_lookup[dataedge['data']['source']]).intersection(
                set(paper_lookup[dataedge['data']['target']])))
        frequency = str(len(papers))
        mention_label = ''' '{}' mentioned with '{}' in {} papers'''.format(
            dataedge['data']['source'], dataedge['data']['target'], frequency)
        npmi = dataedge["data"]["npmi"]
        label = mention_label if str(dataedge['style']['label']) ==\
            "" else str(dataedge['style']['label'])
        npmi_message = html.P(
            "Normalized pointwise mutual information: {:.2f}".format(npmi),
            id="edgeDesc")

        modal_buttons = [
            dbc.Button(label, id="open-body-scroll", color="primary")
        ]

    papers_in_kg = None
    if len(papers) > 0:
        try:
            if isinstance(papers, str):
                papers = ast.literal_eval(papers)
            papers_in_kg = visualization_app._list_papers_callback(papers)
        except Exception as e:
            print(e)
            error_message = "Error fetching papers for the selected node"
        rows = []

        def _convert_to_str(x):
            return x if isinstance(x, str) else ""

        if papers_in_kg:
            for paper in papers_in_kg:
                title = _convert_to_str(paper["title"])
                authors = _convert_to_str(paper["authors"])
                abstract = _convert_to_str(paper["abstract"])
                journal = _convert_to_str(paper["journal"])
                url = _convert_to_str(paper["url"])
                publish_time = _convert_to_str(paper["publish_time"])

                abstract = (
                    (abstract[:500] + '...')
                    if abstract and len(abstract) > 500 else abstract
                )

                paper_card = dbc.Card(
                    dbc.CardBody([
                        html.H4(title, className="card-title"),
                        html.H5(
                            authors, className="card-subtitle",
                            style={"margin-bottom": "20pt"}),
                        html.H6(
                            journal + " (" + publish_time + ")",
                            className="card-subtitle"),
                        html.P(
                            abstract, className="card-text"),
                        dbc.Button(
                            "View the paper", href=url,
                            target="_blank", color="primary"),
                    ]))
                rows.append(paper_card)

            cards = dbc.Row(rows)
        else:
            cards = html.P(
                error_message,
                id="paperErrorMessage",
                style={"color": "red"})

        modal = html.Div(
            [npmi_message] +
            modal_buttons +
            [
                html.Div(
                    id="aggregated-entity-stats",
                    style={"min-height": "35pt", "margin-bottom": "10pt"}),
                dbc.Modal(
                    [
                        dbc.ModalHeader("{} {}".format(
                            label,
                            "(displaying 200 results)"
                            if int(frequency) > 200 else "")),
                        dbc.ModalBody(cards),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close", id="close-body-scroll",
                                className="ml-auto"))
                    ],
                    id="modal-body-scroll",
                    scrollable=True,
                    size="lg"
                ),
            ]
        )
        if len(res) > 0:
            res[0].append(modal)
        else:
            res.append(modal)
    else:
        res = [html.H5("Select an item for details", className="card-title")]
    return res


@visualization_app._app.callback(
    Output('cytoscape', 'layout'),
    [
        Input('dropdown-layout', 'value'),
        Input('groupedLayout', "value"),
    ],
    [
        State("showgraph", "value"),
    ]
)
def update_cytoscape_layout(layout, grouped, current_graph):
    visualization_app._current_layout = layout

    if layout in LAYOUT_CONFIGS:
        layout_config = LAYOUT_CONFIGS[layout]
    else:
        layout_config = {
            "name": layout
        }

    return layout_config


@visualization_app._app.callback(
    Output('cytoscape', 'stylesheet'),
    [
        Input('cytoscape', 'elements'),
        Input('input-follower-color', 'value'),
        Input('dropdown-node-shape', 'value'),
        Input('showgraph', 'value'),
        Input('node_freq_type', 'value'),
        Input('edge_freq_type', 'value'),
        Input('cluster_type', 'value'),
        Input('groupedLayout', "value"),
    ],
    [
        State('cytoscape', 'stylesheet'),
        State("searchdropdown", "value"),
        State('cytoscape', 'selectedNodeData'),
        State('cytoscape', 'selectedEdgeData'),
    ])
def generate_stylesheet(elements,
                        follower_color, node_shape,
                        showgraph, node_freq_type, edge_freq_type,
                        cluster_type, grouped,
                        # states
                        original_stylesheet, searchvalue,
                        selected_nodes, selected_edges):
    stylesheet = [
        s
        for s in CYTOSCAPE_STYLE_STYLESHEET
        if "selected" not in s["selector"]
    ]

    if node_freq_type:
        stylesheet = [
            style
            for style in stylesheet
            if "style" in style and not (
                style["selector"] == 'node' and "width" in style["style"])
        ]
        stylesheet.append({
            "selector": 'node',
            'style': {
                "shape": node_shape,
                'width': 'data(' + node_freq_type + '_size)',
                'height': 'data(' + node_freq_type + '_size)',
                'font-size': 'data(' + node_freq_type + '_font_size)'
            }

        })

    if cluster_type:
        stylesheet = [
            style
            for style in stylesheet
            if "style" in style and not (
                'node' in style["selector"] and
                'background-color' in style["style"]
            )
        ]

        types = set([
            el['data'][cluster_type]
            for el in elements
            if cluster_type in el["data"]
        ])
        for t in types:
            stylesheet.append({
                "selector": "node[{} = {}]".format(
                    cluster_type,
                    t if isinstance(t, int) else "'{}'".format(t)),
                "style": {
                    "background-color": COLORS[str(t)],
                    "opacity": 1
                }
            })

    if edge_freq_type:
        stylesheet = [
            style
            for style in stylesheet
            if not (
                style["selector"] == 'edge' and
                'width' in style["style"]
            )
        ]
        stylesheet.append({
            "selector": 'edge',
            'style': {
                'width': 'data(' + edge_freq_type + '_size)'
            }
        })

    highlight_color = "rgba({},{},{},1)".format(
        follower_color['rgb']["r"],
        follower_color['rgb']["g"],
        follower_color['rgb']["b"]
    )

    node_style = {
        "selector": "node:selected",
        "style": {
            "border-width": "5px",
            "border-color": highlight_color,
            "text-opacity": 1,
            'z-index': 9999
        }
    }

    stylesheet.append(node_style)

    if searchvalue:
        if selected_nodes is None or\
           searchvalue not in [n["id"] for n in selected_nodes]:
            r, g, b = adjust_color_lightness(
                follower_color['rgb']["r"], follower_color['rgb']["g"],
                follower_color['rgb']["b"], 1.5)
            stylesheet.append({
                "selector": "node[id='{}']".format(searchvalue),
                "style": {
                    "border-width": "7px",
                    "border-color": "rgb({},{},{})".format(r, g, b),
                    'z-index': 9999
                }
            })
    # Highlight selected edges
    if selected_edges:
        for edge in selected_edges:
            stylesheet += [{
                "selector": 'edge[id= "{}"]'.format(edge['id']),
                "style": {
                    "mid-target-arrow-color": highlight_color,
                    "line-color": highlight_color,
                    'opacity': 1,
                    'z-index': 5000
                }
            }]

    # Highlight incident edges
    focused_nodes = (
        [n["id"] for n in selected_nodes] if selected_nodes else []
    ) + ([searchvalue] if searchvalue else [])

    selected_edge_ids = [
        e["id"] for e in selected_edges
    ] if selected_edges else []

    for el in elements:
        if "source" in el["data"] and el["data"]["id"] not in selected_edge_ids:
            if el['data']['source'] in focused_nodes or\
               el['data']['target'] in focused_nodes:
                stylesheet.append({
                    "selector": 'edge[id= "{}"]'.format(el['data']["id"]),
                    "style": {
                        "mid-target-arrow-color": highlight_color,
                        "line-color": highlight_color,
                        'opacity': 0.6,
                        'z-index': 5000
                    }
                })

    return stylesheet


@visualization_app._app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")])
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@visualization_app._app.callback(
    Output("modal-body-scroll", "is_open"),
    [
        Input("open-body-scroll", "n_clicks"),
        Input("close-body-scroll", "n_clicks"),
    ],
    [State("modal-body-scroll", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@visualization_app._app.callback(
    Output("searchdropdown", "options"),
    [Input("searchdropdown", "search_value")],
    [
        State("searchdropdown", "value"),
        State('cytoscape', 'elements'),
        State('showgraph', "value"),
        State('cluster_type', "value"),
        State("clustersearch", "value"),
        State("nodestokeep", "value")
    ],
)
def update_multi_options(search_value, value, elements, showgraph,
                         cluster_type, cluster_search, nodes_to_keep):
    if not search_value:
        raise PreventUpdate
    return search(
        elements, search_value, value, showgraph, [], cluster_type,
        cluster_search, nodes_to_keep)


@visualization_app._app.callback(
    Output("nodestokeep", "options"),
    [
        Input("nodestokeep", "search_value")
    ],
    [
        State("nodestokeep", "value"),
        State('cytoscape', 'elements'),
        State('showgraph', "value"),
    ])
def update_nodes_to_keep(search_value, value, elements, showgraph):
    if not search_value:
        raise PreventUpdate
    return search(elements, search_value, value, showgraph, [])


@visualization_app._app.callback(
    [
        Output("clustersearch", "value"),
        Output("legend-title", "children")
    ],
    [
        Input("showgraph", "value"),
        Input("cluster_type", "value"),
        Input("addAllClusters", "n_clicks"),
        Input("bt-reset", "n_clicks")
    ])
def prepopulate_value(val, cluster_type, add_all_clusters, bt_reset):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'No clicks yet' and\
       "clustersearch" in visualization_app._configs:
        types = visualization_app._configs["clustersearch"]
        visualization_app._is_initializing = True
    else:
        graph_object = visualization_app._graphs[val]["object"]
        processor = visualization_app._graph_processor.from_graph_object(
            graph_object)
        types = get_all_clusters(processor, cluster_type)

    for lbl, v in components.cluster_types:
        if v == cluster_type:
            cluster_label = lbl
    if button_id == "bt-reset":
        visualization_app._graphs[val]["top_n"] = None

    legend_title = "Legend (colored by {})".format(cluster_label)
    return [types, legend_title]


@visualization_app._app.callback(
    Output("clustersearch", "options"),
    [
        Input("showgraph", "value"),
        Input("cluster_type", "value"),
        Input("clustersearch", "search_value")
    ],
    [
        State("clustersearch", "value")
    ],
)
def update_cluster_search(current_graph, cluster_type, search_value, value):
    graph_object = visualization_app._graphs[current_graph]["object"]
    processor = visualization_app._graph_processor.from_graph_object(
        graph_object)
    types = get_all_clusters(processor, cluster_type)
    res = []
    for t in types:
        if search_value:
            if (search_value in t) or (t in search_value) or\
               t in (value or []):
                res.append({"label": t, "value": t})
        else:
            res.append({"label": t, "value": t})
    return res


@visualization_app._app.callback(
    Output("searchpathto", "options"),
    [Input("searchpathto", "search_value")],
    [
        State('cytoscape', "elements"),
        State("searchpathto", "value"),
        State('searchpathfrom', 'value'),
        State('showgraph', 'value'),
        State('cluster_type', "value"),
        State("clustersearch", "value"),
        State("nodestokeep", "value")
    ]
)
def searchpathto(search_value, elements, value, node_from, showgraph,
                 cluster_type, cluster_search, nodes_to_keep):
    if not search_value:
        raise PreventUpdate

    return search(
        elements, search_value, value, showgraph, [node_from],
        cluster_type, cluster_search, nodes_to_keep, global_scope=True)


@visualization_app._app.callback(
    Output("searchnodetotraverse", "options"),
    [Input("searchnodetotraverse", "search_value")],
    [
        State('cytoscape', "elements"),
        State("searchnodetotraverse", "value"),
        State('searchpathfrom', 'value'),
        State('searchpathto', 'value'),
        State('showgraph', 'value'),
        State('cluster_type', "value"),
        State("clustersearch", "value"),
        State("nodestokeep", "value")
    ]
)
def searchpathtraverse(search_value, elements, value, node_from, to, showgraph,
                       cluster_type, cluster_search, nodes_to_keep):
    if not search_value:
        raise PreventUpdate
    return search(
        elements, search_value, value, showgraph, [node_from, to],
        cluster_type, cluster_search, nodes_to_keep, global_scope=True)


@visualization_app._app.callback(
    Output("searchpathfrom", "options"),
    [Input("searchpathfrom", "search_value")],
    [
        State('cytoscape', "elements"),
        State("searchpathfrom", "value"),
        State('showgraph', 'value'),
        State('cluster_type', "value"),
        State("clustersearch", "value"),
        State("nodestokeep", "value")
    ],
)
def searchpathfrom(search_value, elements, value, showgraph,
                   cluster_type, cluster_search, nodes_to_keep):
    if not search_value:
        raise PreventUpdate
    return search(
        elements, search_value, value, showgraph, [],
        cluster_type, cluster_search, nodes_to_keep, global_scope=True)


@visualization_app._app.callback(
    [
        Output('cytoscape', 'generateImage')
    ],
    [
        Input('jpg-menu', 'n_clicks'),
        Input('svg-menu', 'n_clicks'),
        Input('png-menu', 'n_clicks'),
    ]
)
def download_image(jpg_menu, svg_menu, png_menu):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    ftype  = None
    if button_id == "png-menu":
        ftype = "png"
    if button_id == "jpg-menu":
        ftype = "jpg"
    if button_id == "svg-menu":
        ftype = "svg"

    return [{
        'type': ftype,
        'action': "download"
    }]


@visualization_app._app.callback(
    Output('download-gml', 'data'),
    [Input('gml-menu', 'n_clicks')],
    [
        State("cytoscape", "elements"),
        State("node_freq_type", "value"),
        State("edge_freq_type", "value")
    ])
def download_gml(clicks, elements, node_freq_type, edge_freq_type):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "gml-menu":
        data_string = generate_gml(
            elements, node_freq_type, edge_freq_type)
        return dict(filename="graph.gml", content=data_string, type="text/gml")
    else:
        raise PreventUpdate


@visualization_app._app.callback(
    [
        Output('cluster_board', "children")
    ],
    [
        Input('cluster_type', "value"),
        Input("clustersearch", "value"),
    ],
    [
        State('cytoscape', 'elements')
    ])
def generate_legend(cluster_type, cluster_search, elements):

    children = []

    for t in cluster_search:
        children.append(
            dbc.Button(
                [
                    html.Span(
                        className="legend-item",
                        style={
                            "height": "15px",
                            "width": "15px",
                            "border-radius": "50%",
                            "display": "inline-block",
                            "vertical-align": "text-bottom",
                            "margin-right": "5px",
                            "background-color": COLORS[str(t)]
                        }),
                    t
                ],
                color="defalut",
            )
        )
    return [children]


@visualization_app._app.callback(
    Output('merge-input', 'children'),
    [
        Input('merge-options', 'value'),
        Input("cytoscape", "selectedNodeData")
    ])
def update_merge_input(value, selected_nodes):
    if value == 1:
        nodes_to_merge = [
            el["id"] for el in (selected_nodes if selected_nodes else [])
        ]
        options = [
            {"label": n, "value": n} for n in nodes_to_merge
        ]

        return dcc.Dropdown(
            id="merge-label-input", placeholder="Select entity...",
            options=options, searchable=False)
    else:
        return dbc.Input(
            id="merge-label-input", placeholder="New entity name...",
            type="text")


@visualization_app._app.callback(
    [
        Output("collapse-legend", "is_open"),
        Output("collapse-details", "is_open"),
        Output("collapse-edit", "is_open"),
        Output("collapse-neighbors", "is_open"),
    ],
    [
        Input("toggle-legend", "n_clicks"),
        Input("toggle-details", "n_clicks"),
        Input("toggle-edit", "n_clicks"),
        Input("toggle-neighbors", "n_clicks"),
        Input("toggle-hide", "n_clicks")
    ],
    [
        State("collapse-legend", "is_open"),
        State("collapse-details", "is_open"),
        State("collapse-edit", "is_open"),
        State("collapse-neighbors", "is_open")
    ])
def toggle_bottom_tabs(legend_b, details_b, edit_b, neighb_b, hide_b,
                       legend_is_open, details_is_open, edit_is_open,
                       neighb_is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "toggle-hide":
        legend_is_open = False
        details_is_open = False
        edit_is_open = False
        neighb_is_open = False

    if button_id == "toggle-legend":
        legend_is_open = not legend_is_open
        details_is_open = False
        edit_is_open = False
        neighb_is_open = False
    if button_id == "toggle-details" or button_id == "cytoscape":
        legend_is_open = False
        details_is_open = not details_is_open
        edit_is_open = False
        neighb_is_open = False
    if button_id == "toggle-edit":
        legend_is_open = False
        details_is_open = False
        edit_is_open = not edit_is_open
        neighb_is_open = False
    if button_id == "toggle-neighbors":
        legend_is_open = False
        details_is_open = False
        edit_is_open = False
        neighb_is_open = not neighb_is_open

    return [legend_is_open, details_is_open, edit_is_open, neighb_is_open]


@visualization_app._app.callback(
    Output("loading-output", "children"),
    [Input("cytoscape", "elements")]
)
def input_triggers_spinner(value):
    time.sleep(1)
    return []


@visualization_app._app.callback(
    Output("aggregated-entity-stats", "children"),
    [
        Input("show-aggregated-entities", "n_clicks")
    ],
    [
        State('cytoscape', 'tapNode')
    ]
)
def compute_aggregated_stats(button_clicked, tapped_node):
    if button_clicked:
        res = visualization_app._aggregated_entities_callback(
            tapped_node["data"]["id"])
        if res is not None:
            list_group = dbc.ListGroup([
                dbc.ListGroupItem("{}. {} ({})".format(i + 1, k, v))
                for i, (k, v) in enumerate(res.items())
            ])
        else:
            list_group = html.P(
                "Cannot fetch aggregated entities (manually renamed entity?)",
                id="fetch-aggregated-error-message",
                style={"color": "red"})
        entities_card_content = [
            html.H6(
                "Top raw entities linked to '{}' (by paper frequency)".format(
                    tapped_node["data"]["id"]), className="card-title",
                style={"margin-top": "15pt"}),
            list_group
        ]
        return entities_card_content
    else:
        return []
