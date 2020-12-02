import os
import flask

import json
import time
import math
import numpy as np
import networkx as nx

import copy
from operator import ge, gt, lt, le, eq, ne
from collections import OrderedDict

from jupyter_dash import JupyterDash

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State

from bbg_apps.curation_app import DROPDOWN_FILTER_LIST
from bbg_apps.resources import (VISUALIZATION_CONTENT_STYLE,
                                CYTOSCAPE_STYLE_STYLESHEET,
                                DEFAULT_TYPES,
                                MIN_NODE_SIZE,
                                MAX_NODE_SIZE,
                                MIN_FONT_SIZE,
                                MAX_FONT_SIZE,
                                MIN_EDGE_WIDTH,
                                MAX_EDGE_WIDTH,
                                LAYOUT_CONFIGS,
                                COLORS)
import bbg_apps.components as components

from dash.exceptions import PreventUpdate

from kganalytics.paths import (top_n_paths, top_n_tripaths, top_n_nested_paths,
                               minimum_spanning_tree, graph_from_paths)
from kganalytics.utils import (top_n, merge_nodes)
from cord_analytics.utils import (build_cytoscape_data, generate_paper_lookup, CORD_ATTRS_RESOLVER)

from colorsys import rgb_to_hls, hls_to_rgb
from bbg_apps.utils import save_run, merge_cyto_elements


def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def lighten_color(r, g, b, factor=0.1):
    return adjust_color_lightness(r, g, b, 1 + factor)


def filter_nodes_by_attr(graph_object, key, values):
    result = []
    for n in graph_object.nodes():
        if key in graph_object.nodes[n]:
            if graph_object.nodes[n][key] in values:
                result.append(n)
    return result


def subgraph_from_clusters(graph_object, cluster_type, clustersearch, nodes_to_keep):
    if cluster_type and clustersearch:
        graph_object = nx.Graph(graph_object.subgraph(
            nodes=[
                n
                for n in graph_object.nodes()
                if graph_object.nodes[n][cluster_type] in clustersearch or n in nodes_to_keep
            ]))
    return graph_object


def get_top_n_nodes(graph_object, n, node_subset=None, nodes_to_keep=None):
    """Get top N nodes by paper frequency."""
    if nodes_to_keep is None:
        nodes_to_keep = []

    if n is None:
        n = len(graph_object.nodes())
    
    if node_subset is None:
        node_subset = list(graph_object.nodes())
    
    if n <= len(node_subset):
        node_frequencies = {}
        for node in node_subset:
            node_frequencies[node] = len(graph_object.nodes[node]["paper"])
        nodes_to_include = top_n(node_frequencies, n)
    else:
        nodes_to_include = node_subset

    return nodes_to_include + [n for n in nodes_to_keep if n not in nodes_to_include]


def top_n_subgraph(graph_object, n, node_subset=None, nodes_to_keep=None):
    """Build a subgraph with top n nodes."""
    nodes_to_include = get_top_n_nodes(
        graph_object, n, node_subset, nodes_to_keep)

    return nx.Graph(graph_object.subgraph(nodes_to_include))


def top_n_spanning_tree(graph_object, n, node_subset=None, nodes_to_keep=None):
    """Build a spanning tree with default top n nodes."""
    nodes_to_include = get_top_n_nodes(
        graph_object, n, node_subset, nodes_to_keep)

    tree = minimum_spanning_tree(
        graph_object.subgraph(nodes_to_include),
        "distance_npmi")
    return tree


def get_cytoscape_data(graph, positions=None):
     # Generate a cytoscape repr of the input graph
    cyto_repr = build_cytoscape_data(graph, positions=positions)

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
            font_sizes = generate_sizes(min_font_size, max_font_size, all_values)

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


def clear_grouping(elements):
    new_elements = []
    for el in elements:
        if el["data"]["type"] != "cluster":
            new_element = {"data": {}}
            for k, v in el["data"].items():
                if k != "parent":
                    new_element['data'][k] = v
            new_elements.append(new_element)
    return new_elements        


def generate_gml(elements, node_freq_type=None, edge_freq_type=None):
    result = (
"""graph
[
    Creator "bbg_app"
    directed 0"""
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
                    COLORS[el["data"]["entity_type"]])
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
    
    def __init__(self, configs=None):
        self._app =  JupyterDash("allvis")
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

    def _configure_layout(self, configs=None):
        if configs is None:
            configs = {}

        self.cyto, layout, self.dropdown_items, self.cluster_filter =\
            components.generate_layout(self._graphs, configs)

        self._app.layout  = layout


    def _update_weight_data(self, graph_id, cyto_repr,
                           node_freq_type="degree_frequency", edge_freq_type="npmi"):
        
        min_value, max_value, marks, step = recompute_node_range(cyto_repr, node_freq_type)
        self._graphs[graph_id]["min_node_weight"] = min_value
        self._graphs[graph_id]["max_node_weight"] = max_value
        self._graphs[graph_id]["current_node_value"] = [min_value, max_value]
        self._graphs[graph_id]["node_marks"] = marks
        self._graphs[graph_id]["node_step"] = step
        
        min_value, max_value, marks, step = recompute_node_range(cyto_repr, edge_freq_type)
        self._graphs[graph_id]["min_edge_weight"] = min_value
        self._graphs[graph_id]["max_edge_weight"] = max_value
        self._graphs[graph_id]["current_edge_value"] = [min_value, max_value]
        self._graphs[graph_id]["edge_marks"] = marks
        self._graphs[graph_id]["edge_step"] = step

    def _update_cyto_graph(self, graph_id, graph_object, top_n_entities=None, positions=None,
                           node_freq_type="degree_frequency", edge_freq_type="npmi", node_subset=None,
                           nodes_to_keep=None):
        if not self._graphs[graph_id]["full_graph_view"]:
            if top_n_entities is None:
                # compute the spanning tree on all the nodes
                if "full_tree" in self._graphs[graph_id] and (
                        node_subset is None or set(node_subset) == set(graph_object.nodes())):
                    graph_view = self._graphs[graph_id]["full_tree"]
                else:
                    graph_view = top_n_spanning_tree(
                        graph_object, len(graph_object.nodes()),
                        node_subset=node_subset, nodes_to_keep=nodes_to_keep)
            else:
                # compute the spanning tree on n nodes
                graph_view = top_n_spanning_tree(
                    graph_object, top_n_entities,
                    node_subset=node_subset, nodes_to_keep=nodes_to_keep)
        else:
            if top_n_entities is None:
                graph_view = graph_object
            else:
                graph_view = top_n_subgraph(
                    graph_object, top_n_entities,
                    node_subset=node_subset, nodes_to_keep=nodes_to_keep)
    
        if positions is None and top_n_entities is None:
            if "positions" in self._graphs[graph_id]:
                positions = self._graphs[graph_id]["positions"]
            
        cyto_repr = get_cytoscape_data(graph_view, positions=positions)
        
        self._graphs[graph_id]["cytoscape"] = cyto_repr
        self._graphs[graph_id]["top_n"] = top_n_entities
        
        self._update_weight_data(
            graph_id, cyto_repr, node_freq_type=node_freq_type, edge_freq_type=edge_freq_type)
        return cyto_repr
    
    def _update_configs(self, elements, current_graph, nodes_to_keep, top_n_slider_value, node_freq_type, edge_freq_type,
                        nodefreqslider, edgefreqslider, cluster_type, clustersearch,
                        searchpathfrom, searchpathto, searchnodetotraverse, searchpathlimit,
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

    def save_configs(self, path):
        with open(path, "w+") as f:
            json.dump(self._configs, f)
        
    def set_graph(self, graph_id, graph_object, tree_object=None,
                  positions=None, default_top_n=None, full_graph_view=False):
        # Generate a paper lookup table
        paper_lookup = generate_paper_lookup(graph_object)

        if self._graphs is None:
            self._graphs = {}

        self._graphs[graph_id] = {
            "nx_object_backup": graph_object.copy(),
            "nx_object": graph_object,
            "positions": positions,
            "paper_lookup": paper_lookup,
            "full_graph_view": full_graph_view
        }

        if tree_object:
            self._graphs[graph_id]["full_tree"] = tree_object

        if default_top_n:
            default_top_n = len(graph_object.nodes()) if default_top_n > len(graph_object.nodes()) else default_top_n
            self._graphs[graph_id]["default_top_n"] = default_top_n

         # Build a cyto repe with the spanning tree with default top n nodes
        self._update_cyto_graph(
            graph_id, graph_object, default_top_n, positions)   
            
        self.dropdown_items.options = [
            {'label': val.capitalize(), 'value': val} for val in list(self._graphs.keys())
        ]
        
        return

    def set_current_graph(self, graph_id):
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
        save_run(self, port, mode=mode, debug=debug, inline_exceptions=inline_exceptions)

    def set_list_papers_callback(self, func):
        self._list_papers_callback = func

    def set_entity_definitons(self, definition_dict):
        self._entity_definitions = definition_dict

        
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
        for n in visualization_app._graphs[showgraph]["nx_object"].nodes():
            attrs = visualization_app._graphs[showgraph]["nx_object"].nodes[n]
        
            cluster_matches = False
            if cluster_type is not None and cluster_search is not None:
                if (cluster_type in attrs and attrs[cluster_type] in cluster_search) or\
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
                    if (cluster_type in ele_data["data"] and ele_data["data"][cluster_type] in cluster_search) or\
                           el_id in nodes_to_keep:
                        cluster_matches = True
                else:
                    cluster_matches = True

                # Check if the name matches
                name_matches = False

                if (search_value in el_id) or (el_id in search_value) or el_id in value:
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
            (edge_condition(el["data"]) if edge_condition is not None else True)
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


def get_all_clusters(graph_id, cluster_type):
    return list(set([
        visualization_app._graphs[graph_id]["nx_object"].nodes[n][cluster_type]
        for n in visualization_app._graphs[graph_id]["nx_object"].nodes()
        if cluster_type in visualization_app._graphs[graph_id]["nx_object"].nodes[n]
    ]))


def get_all_clusters(graph_id, cluster_type):
    return list(set([
        visualization_app._graphs[graph_id]["nx_object"].nodes[n][cluster_type]
        for n in visualization_app._graphs[graph_id]["nx_object"].nodes()
        if cluster_type in visualization_app._graphs[graph_id]["nx_object"].nodes[n]
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
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
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
                              selected_rename_input, editing_mode, neighborlimit, edge_expansion_limit):
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

    current_top_n = visualization_app._graphs[val]["top_n"]
    total_number_of_entities = len(visualization_app._graphs[val]["nx_object"].nodes())
    
    output_editing_mode = editing_mode
    if button_id in full_graph_events:
        output_editing_mode = 1

    # -------- Filter elements by selected clusters to display -------- 
    if nodes_to_keep is None:
        nodes_to_keep = []

    nodes_with_cluster = filter_nodes_by_attr(
        visualization_app._graphs[val]["nx_object"], cluster_type, clustersearch)
    
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
    
    if clustersearch is not None and cluster_type is not None and button_id != "bt-reset" and button_id not in element_preserving:
        memory["filtered_elements"] = [] 
        memory["removed_elements"] =  {} 
        memory["renamed_elements"] =  {} 
        memory["merged_elements"] =  {}
        if button_id == "clustersearch" and visualization_app._is_initializing:
            # If app is in the initialization stage, then we fix some of the parameters from the loaded configs
            configs = visualization_app._configs
            visualization_app._is_initializing = False
            elements = configs["elements"] if "elements" in configs else []
            if "current_graph" in configs and "nodefreqslider" in configs:
                visualization_app._graphs[configs["current_graph"]]["current_node_value"] = configs["nodefreqslider"]
            if "current_graph" in configs and "edgefreqslider" in configs:
                visualization_app._graphs[configs["current_graph"]]["current_edge_value"] = configs["edgefreqslider"]
        else:
            if len(recompute_spanning_tree) > 0:
                elements = visualization_app._update_cyto_graph(
                    val, visualization_app._graphs[val]["nx_object"],
                    visualization_app._graphs[val]["top_n"],
                    node_freq_type=node_freq_type, edge_freq_type=edge_freq_type,
                    node_subset=nodes_with_cluster, nodes_to_keep=nodes_to_keep)
            else:
                elements, hidden_elements = filter_elements(
                    elements,
                    node_condition=lambda x: (cluster_type in x and x[cluster_type] in clustersearch) or x["id"] in nodes_to_keep 
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
                ele_data["id"]: {"data": ele_data} for ele_data in selected_edge_data
            }
            
        if editing_mode == 1:
            for n in nodes_to_remove:
                visualization_app._graphs[val]["nx_object"].remove_node(n)

            for edge in edges_to_remove.values():
                if (edge["data"]["source"], edge["data"]["target"]) in visualization_app._graphs[val]["nx_object"].edges():
                    visualization_app._graphs[val]["nx_object"].remove_edge(edge["data"]["source"], edge["data"]["target"])

            elements = visualization_app._update_cyto_graph(
                val, visualization_app._graphs[val]["nx_object"], visualization_app._graphs[val]["top_n"],
                node_freq_type=node_freq_type, edge_freq_type=node_freq_type,
                nodes_to_keep=nodes_to_keep)
        else:
            memory["removed_elements"].update(nodes_to_remove)
            memory["removed_elements"].update(edges_to_remove)

        
    # -------- Handle merge selected nodes -------
       
    # Open/close the merge dialog
    if button_id == "merge-button" or button_id == "merge-close" or\
       button_id == "merge-apply":
        open_merge_modal = not open_merge_modal
        
    # Merge elements dict
    
    # Merge underlying nx_object and tree if applicable
    if button_id == "merge-apply" and merge_label_value:
        # Retreive name 
        new_name = merge_label_value
        
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
                visualization_app._graphs[val]["nx_object"], list(selected_nodes),
                new_name, CORD_ATTRS_RESOLVER, copy=False)

            elements = visualization_app._update_cyto_graph(
                val, new_graph, visualization_app._graphs[val]["top_n"],
                node_freq_type=node_freq_type, edge_freq_type=node_freq_type,
                nodes_to_keep=nodes_to_keep)

            visualization_app._graphs[val]["paper_lookup"][new_name] = new_graph.nodes[new_name]["paper"]
        else:
            elements, target_node, merging_data = merge_cyto_elements(elements, selected_nodes, new_name)
            
            for k, v in merging_data["removed_elements"].items():
                if k not in memory["merging_backup"]["added_elements"]:
                    memory["merging_backup"]["removed_elements"][k] = v

            memory["merging_backup"]["added_elements"] += merging_data["added_elements"]
            
            papers = set()
            for n in selected_nodes:    
                memory["paper_backup"][n] = visualization_app._graphs[val]["paper_lookup"][n]
                papers.update(visualization_app._graphs[val]["paper_lookup"][n])
            
            visualization_app._graphs[val]["paper_lookup"][new_name] = list(papers)
            

    if button_id == "rename-apply":
        if selected_rename_input:
            rename_input_value = selected_rename_input 
        
        # Check if the rename input is valid
        if rename_input_value != selected_nodes[0]:
            if editing_mode == 1:
                if rename_input_value in visualization_app._graphs[val]["nx_object"].nodes():
                    rename_invalid = True
                    rename_error_message = "Node with the label '{}' already exists".format(
                        rename_input_value)
            else:
                for el in elements:
                    if rename_input_value == el["data"]["id"]:
                        rename_invalid = True
                        rename_error_message = "Node with the label '{}' already exists".format(
                            rename_input_value)
                        break
            
        if not rename_invalid:
            if editing_mode == 1:
                # Rename node in the graph
                new_graph = nx.relabel_nodes(
                    visualization_app._graphs[val]["nx_object"],
                    {selected_nodes[0]: rename_input_value})

                visualization_app._graphs[val]["nx_object"] = new_graph
                elements = visualization_app._update_cyto_graph(
                    val, new_graph, visualization_app._graphs[val]["top_n"],
                    node_freq_type=node_freq_type, edge_freq_type=node_freq_type,
                    nodes_to_keep=nodes_to_keep)

                visualization_app._graphs[val]["paper_lookup"][rename_input_value] = visualization_app._graphs[
                    val]["paper_lookup"][selected_nodes[0]]
                visualization_app._entity_definitions[rename_input_value] =\
                    visualization_app._entity_definitions[selected_nodes[0]]    
            else:
                memory["renamed_elements"].update({selected_nodes[0]: rename_input_value})

    # Open/close the rename dialog
    if button_id == "rename-button" or button_id == "rename-close" or\
       (button_id == "rename-apply" and not rename_invalid):
        open_rename_modal = not open_rename_modal
                        
    # -------- Handle reset graph elements --------
    if button_id == "reset-elements-button":
        if editing_mode == 1:
            visualization_app._graphs[val]["nx_object"] = visualization_app._graphs[
                val]["nx_object_backup"].copy()
            elements = visualization_app._update_cyto_graph(
                val, visualization_app._graphs[val]["nx_object_backup"], visualization_app._graphs[val]["top_n"],
                node_freq_type=node_freq_type, edge_freq_type=node_freq_type,
                nodes_to_keep=nodes_to_keep)
        else:
            # Remove added elements
            elements = [
                el for el in elements
                if el["data"]["id"] not in memory["merging_backup"]["added_elements"]
            ]
            for el in memory["merging_backup"]["added_elements"]:
                if el in visualization_app._graphs[val]["paper_lookup"]:
                    del visualization_app._graphs[val]["paper_lookup"][el]

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
                
            #Bring back paper lookup
            visualization_app._graphs[val]["paper_lookup"].update(memory["paper_backup"])
            
            memory["removed_elements"] = {}
            memory["merged_elements"] = {}
            memory["renamed_elements"] = {}
            
    # --------- Handle neighbor search -----
    neighbor_card_content = [
        html.H6("No neighbors to display", className="card-title")
    ]
    if button_id == "bt-neighbors":
        graph = visualization_app._graphs[val]["nx_object"]
        weights = {}
        for n in graph.neighbors(selected_nodes[0]):
             weights[n] = graph.edges[selected_nodes[0], n]["npmi"]
        top_neighbors = top_n(weights, neighborlimit)

        list_group = dbc.ListGroup([
            dbc.ListGroupItem("{}. {} (NMPI {:.2f})".format(i + 1, n, weights[n]))
             for i, n in enumerate(top_neighbors)
        ])
        neighbor_card_content = [
            html.H6("Top neighbors of '{}' by NMPI".format(
                selected_nodes[0]), className="card-title"),
            list_group
        ]

    # -------- Handle path search -------
    no_path_message = ""
    if button_id == "bt-path" and pathbt is not None:
        memory["removed_nodes"] = []
        memory["removed_edges"] = []
        memory["filtered_elements"] = []
        
        output_editing_mode = 2
        
        success = False
        if searchpathfrom and searchpathto:
            topN = searchpathlimit if searchpathlimit else 20
            
            source = searchpathfrom
            target = searchpathto
            
            # create a subgraph given the selected clusters 
            
            graph_object = subgraph_from_clusters(
                visualization_app._graphs[val]["nx_object"], 
                cluster_type, clustersearch, nodes_to_keep)
                
            try:
                if nestedpaths and pathdepth:
                    paths = top_n_nested_paths(
                        graph_object, source, target, topN, nested_n=topN,
                        strategy="naive", distance="distance_npmi", depth=pathdepth)
                elif searchnodetotraverse:
                    intersecting = len(searchpathoverlap) == 1
                    a_b_paths, b_c_paths = top_n_tripaths(
                        graph_object, source,
                        searchnodetotraverse, target, topN,
                        strategy="naive", distance="distance_npmi",
                        intersecting=intersecting, pretty_print=False)
                    paths = a_b_paths + b_c_paths
                else:
                    paths = top_n_paths(
                        graph_object, source, target,
                        topN, distance="distance_npmi", strategy="naive", pretty_print=False)
                elements = []

                visited = set()
                path_graph = graph_from_paths(
                    paths, visualization_app._graphs[val]["nx_object"])
                elements = get_cytoscape_data(path_graph)
                visualization_app._current_layout = "klay"
                success = True

            except ValueError as e:
                print(e)
                no_path_message = "No undirect paths from '{}' to '{}' were found (the nodes are either disconnected or connected by a direct edge only)".format(
                    source, target)
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
            visualization_app._graphs[val]["nx_object"], 
            cluster_type, clustersearch, nodes_to_keep)
        
        paths = top_n_paths(
            graph_object, source, target,
            edge_expansion_limit, distance="distance_npmi", strategy="naive", pretty_print=False)

        path_graph = graph_from_paths(
            paths, visualization_app._graphs[val]["nx_object"])
        new_elements = get_cytoscape_data(path_graph)
        
        existing_nodes = [
            el["data"]["id"] for el in elements if "source" not in el["data"]
        ]
        
        existing_edges = [
            (el["data"]["source"], el["data"]["target"]) for el in elements if "source" in el["data"]
        ]
        
        for el in new_elements:
            if "source" not in el["data"]:
                if el["data"]["id"] not in existing_nodes:
                    elements.append(el)
            else:
                if (el["data"]["source"], el["data"]["target"]) not in existing_edges and\
                   (el["data"]["target"], el["data"]["source"]) not in existing_edges:
                    elements.append(el)

        visualization_app._current_layout = "klay"

    # -------- Handle 'display a spanning tree on top N nodes' -------
    message = (
        "Displaying top {} most frequent entities (out of {})".format(
            visualization_app._graphs[val]["top_n"], total_number_of_entities)
        if visualization_app._graphs[val]["top_n"] is not None
        else "Displaying all {} entities".format(total_number_of_entities)
    )
    
    def has_clusters(graph, node):
        if clustersearch is not None and cluster_type is not None:
            if cluster_type in graph.nodes[node] and\
               graph.nodes[node][cluster_type] in clustersearch:
                return True
            else:
                return False
        else:
            return True
    
    if button_id == "top-n-button":
        if current_top_n is None or current_top_n != top_n_slider_value:
            if len(recompute_spanning_tree) > 0:
                elements = visualization_app._update_cyto_graph(
                    val, visualization_app._graphs[val]["nx_object"], top_n_slider_value,
                    node_freq_type=node_freq_type, edge_freq_type=edge_freq_type,
                    node_subset=nodes_with_cluster, nodes_to_keep=nodes_to_keep)
            else:
                top_n_nodes = get_top_n_nodes(
                    visualization_app._graphs[val]["nx_object"], top_n_slider_value,
                )
                
                elements, hidden_elements = filter_elements(
                    elements,
                    lambda x: x["id"] in top_n_nodes or x["id"] in nodes_to_keep, 
                    lambda x: (
                        x["source"] in top_n_nodes or x["source"] in nodes_to_keep
                     ) and (
                        x["target"] in top_n_nodes or x["target"] in nodes_to_keep
                    )
                )
                memory["filtered_elements"] = hidden_elements
                    
        message = "Displaying top {} most frequent entities (out of {})".format(
            top_n_slider_value
            if top_n_slider_value <= total_number_of_entities
            else total_number_of_entities,
            total_number_of_entities)
    elif button_id == "show-all-button":
#         if current_top_n is not None:
        # Top entities are selected, but button is clicked, so show all
        elements = visualization_app._update_cyto_graph(
            val, visualization_app._graphs[val]["nx_object"], None,
            node_freq_type=node_freq_type, edge_freq_type=edge_freq_type,
            node_subset=nodes_with_cluster, nodes_to_keep=nodes_to_keep)

        message = "Displaying all {} entities".format(total_number_of_entities) 
    
    # -------- Handle node/edge weight sliders -------
    
    def node_range_condition(el, start, end):
        if el["id"] in nodes_to_keep:
            return True
        if node_freq_type in el:
            if el[node_freq_type] >= start and\
               el[node_freq_type] <= end:
                return True
        return False
    
    def edge_range_condition(el, start, end):
        if edge_freq_type in el:
            if el[edge_freq_type] >= start and\
               el[edge_freq_type] <= end:
                return True
        return False
    
    if nodefreqslider and button_id == "nodefreqslider_content":
        if nodefreqslider[0] != visualization_app._graphs[val]["current_node_value"][0] or\
           nodefreqslider[1] != visualization_app._graphs[val]["current_node_value"][1]:
            elements, hidden_elements = filter_elements(
                elements + memory["filtered_elements"],
                lambda x: node_range_condition(x, nodefreqslider[0], nodefreqslider[1]),
                lambda x: edge_range_condition(
                    x,
                    visualization_app._graphs[val]["current_edge_value"][0],
                    visualization_app._graphs[val]["current_edge_value"][1]))
            visualization_app._graphs[val]["current_node_value"] = nodefreqslider
            memory["filtered_elements"] = hidden_elements

            new_marks = {}
            for k, v in visualization_app._graphs[val]["node_marks"].items():
                if k == visualization_app._graphs[val]["min_node_weight"] or\
                   k == visualization_app._graphs[val]["max_node_weight"]:
                     new_marks[k] = v
            
            # add a new value mark
            if nodefreqslider[0] != visualization_app._graphs[val]["min_node_weight"]:           
                new_marks[nodefreqslider[0]] = generate_mark(nodefreqslider[0])
            if nodefreqslider[1] != visualization_app._graphs[val]["max_node_weight"]:           
                new_marks[nodefreqslider[1]] = generate_mark(nodefreqslider[1])
                
            visualization_app._graphs[val]["node_marks"] = new_marks

    elif edgefreqslider and button_id == "edgefreqslider_content":
        if edgefreqslider[0] != visualization_app._graphs[val]["current_edge_value"][0] or\
           edgefreqslider[1] != visualization_app._graphs[val]["current_edge_value"][1]:
            elements, hidden_elements = filter_elements(
                elements + memory["filtered_elements"],
                lambda x: node_range_condition(
                    x,
                    visualization_app._graphs[val]["current_node_value"][0],
                    visualization_app._graphs[val]["current_node_value"][1]),
                lambda x: edge_range_condition(
                    x, edgefreqslider[0], edgefreqslider[1]))
            visualization_app._graphs[val]["current_edge_value"] = edgefreqslider

            memory["filtered_elements"] = hidden_elements
                   
            new_marks = {}
            for k, v in visualization_app._graphs[val]["edge_marks"].items():
                if k == visualization_app._graphs[val]["min_edge_weight"] or\
                   k == visualization_app._graphs[val]["max_edge_weight"]:
                     new_marks[k] = v
    
            # add a new value mark
            if edgefreqslider[0] != visualization_app._graphs[val]["min_edge_weight"]:           
                new_marks[edgefreqslider[0]] = generate_mark(edgefreqslider[0])
            if edgefreqslider[1] != visualization_app._graphs[val]["max_edge_weight"]:           
                new_marks[edgefreqslider[1]] = generate_mark(edgefreqslider[1])
                
            visualization_app._graphs[val]["edge_marks"] = new_marks
  
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
        elements, val, nodes_to_keep, top_n_slider_value, node_freq_type, edge_freq_type,
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
def display_tap_node(datanode, dataedge, statedatanode, statedataedge, showgraph, memory):  
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
    papers = []
    res = []
    modal_button = None
    npmi_message =  None
        
    paper_lookup = visualization_app._graphs[visualization_app._current_graph]["paper_lookup"]

    if datanode:
        definition = ""
        if "definition" in datanode['data']:
            definition = str(datanode["data"]["definition"])
        elif datanode["data"]["id"] in visualization_app._entity_definitions:
            definition = visualization_app._entity_definitions[datanode["data"]["id"]]
        label = datanode['data']['name']
        try:
            _type = str(datanode['data']['entity_type'])
        except Exception as e:
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
        modal_button = dbc.Button(label, id="open-body-scroll",color="primary")
        
        papers = paper_lookup[datanode['data']['id']]

    elements = visualization_app._graphs[showgraph]["cytoscape"]
    elements_dict = {
        el["data"]["id"]: el
        for el in visualization_app._graphs[showgraph]["cytoscape"]
    }
        
    if dataedge and statedataedge:
        label = str(dataedge['style']['label'])
        papers = list(set(paper_lookup[dataedge['data']['source']]).intersection(
            set(paper_lookup[dataedge['data']['target']])))
        frequency = str(len(papers))
        mention_label= ''' '%s' mentioned with '%s' in %s papers''' % (
            dataedge['data']['source'], dataedge['data']['target'], frequency) 
        npmi = dataedge["data"]["npmi"]
        label = mention_label if str(dataedge['style']['label']) == "" else str(dataedge['style']['label']) 
        npmi_message = html.P("Normalized pointwise mutual information: {:.2f}".format(npmi), id="edgeDesc")
        modal_button = dbc.Button(
            label, id="open-body-scroll", color="primary")

    papers_in_kg = None
    if len(papers) > 0:
        try:
            papers_in_kg = visualization_app._list_papers_callback(papers)
        except Exception as e:
            print(e)
            error_message = visualization_app._db_error_message
        rows = []
        
        modal_children = []

        if papers_in_kg:
            for paper in papers_in_kg:
                title = paper[0] if paper[0] else ''
                authors = paper[1] if paper[1] else ''
                abstract = paper[2] if paper[2] else ''
                journal = paper[5] if paper[5] else ''
                url = paper[4] if paper[4] else ''
                publish_time = str(paper[8]) if paper[8] else ''

                abstract = (abstract[:500] + '...') if abstract and len(abstract) > 500 else abstract
                
                paper_card = dbc.Card(
                    dbc.CardBody([
                        html.H4(title, className="card-title"),
                        html.H5(authors, className="card-subtitle", style={"margin-bottom": "20pt"}),
                        html.H6(journal + " (" + publish_time + ")", className="card-subtitle"),
                        html.P(abstract, className="card-text" ),
                        dbc.Button("View the paper", href=url,target="_blank",color="primary"),
                    ]))
                rows.append(paper_card)

            cards = dbc.Row(rows)        
        else:
            cards = html.P(error_message, id="paperErrorMessage", style={"color": "red"})
            
        modal = html.Div(
            [
                npmi_message,
                modal_button,
                dbc.Modal([
                        dbc.ModalHeader("{} {}".format(
                            label,
                            "(displaying 200 results)" if int(frequency) > 200 else "")),
                        dbc.ModalBody(cards),            
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close", id="close-body-scroll", className="ml-auto"
                            )
                        ),
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
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    visualization_app._current_layout = layout
    
    if layout in LAYOUT_CONFIGS:
        layout_config = LAYOUT_CONFIGS[layout]
    else:
        layout_config = {
            "name":  layout
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
    
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
    stylesheet = [
        s
        for s in CYTOSCAPE_STYLE_STYLESHEET
        if "selected" not in s["selector"]
    ]
        
    if node_freq_type:
        stylesheet = [
            style
            for style in stylesheet
            if "style" in style and not (style["selector"] == 'node' and 'width' in style["style"])
        ]
        stylesheet.append({
            "selector": 'node',
            'style': {
                "shape": node_shape,
                'width':'data(' + node_freq_type + '_size)',
                'height':'data(' + node_freq_type + '_size)',
                'font-size':'data(' + node_freq_type + '_font_size)'
            }

        })
        
    if cluster_type:
        stylesheet = [
            style
            for style in stylesheet
            if "style" in style and not ('node' in style["selector"] and 'background-color' in style["style"])
        ]
        
        cluster_styles = []
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
                    "background-color": COLORS[t],
                    "opacity": 1
                }
            })
#         stylesheet.append({
#             "selector": '[type = "cluster"]',
#             "style": {
#                 "opacity": 0.2,
#             },
#         })
        
    if edge_freq_type:
        stylesheet = [style for style in stylesheet if not (style["selector"] == 'edge' and 'width' in style["style"])]
        stylesheet.append({
            "selector": 'edge',
            'style': {
                'width':'data(' + edge_freq_type + '_size)'
            }
        })

    highlight_color = "rgba({},{},{},1)".format(
        follower_color['rgb']["r"], follower_color['rgb']["g"], follower_color['rgb']["b"])    

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
        if selected_nodes is None or searchvalue not in [n["id"] for n in selected_nodes]:
            r, g, b = adjust_color_lightness(
                follower_color['rgb']["r"], follower_color['rgb']["g"], follower_color['rgb']["b"], 1.5)
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
            if el['data']['source'] in focused_nodes or el['data']['target'] in focused_nodes:
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
        elements, search_value, value, showgraph, [], cluster_type, cluster_search, nodes_to_keep)


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
        
    if button_id == 'No clicks yet' and "clustersearch" in visualization_app._configs:
        types = visualization_app._configs["clustersearch"]
        visualization_app._is_initializing = True
    else:
        types = get_all_clusters(val, cluster_type)
    
    l = ""
    for l, v in components.cluster_types:
        if v == cluster_type:
            cluster_label = l
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
    types = get_all_clusters(current_graph, cluster_type)
    res = []
    for t in types:
        if search_value:
            if (search_value in t) or (t in search_value) or t in (value or []) :
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
    types = set([
        el['data'][cluster_type] for el in elements if cluster_type in el['data']
    ])

    children = []

    for t in cluster_search:
        children.append(
            dbc.Button([
                    html.Span(
                        className="legend-item",
                        style={
                            "height": "15px",
                            "width": "15px",
                            "border-radius": "50%",
                            "display": "inline-block",
                            "vertical-align": "text-bottom",
                            "margin-right": "5px",
                            "background-color": COLORS[t]
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
#         Input("searchdropdown", "value"),
        Input("cytoscape", "selectedNodeData")
    ])
def update_merge_input(value, selected_nodes):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

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
            id="merge-label-input", placeholder="New entity name...", type="text")


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
        Input("toggle-hide", "n_clicks"),
#         Input('cytoscape', 'tapNode'),
#         Input('cytoscape', 'tapEdge')
    ],
    [
        State("collapse-legend", "is_open"),
        State("collapse-details", "is_open"),
        State("collapse-edit", "is_open"),
        State("collapse-neighbors", "is_open")
    ])
def toggle_bottom_tabs(legend_b, details_b, edit_b, neighb_b, hide_b, 
#                        tap_node, tap_edge,
                       legend_is_open, details_is_open, edit_is_open, neighb_is_open):
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
