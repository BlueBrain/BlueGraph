import os
import flask

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
import dash_daq as daq
from dash_extensions import Download
# from dash_extensions.snippets import download_store

from dash.dependencies import Input, Output, State

import dash_cytoscape as cyto

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
from dash.exceptions import PreventUpdate

from kganalytics.paths import (top_n_paths, top_n_tripaths, top_n_nested_paths,
                               minimum_spanning_tree, graph_from_paths)
from kganalytics.utils import (top_n)
from cord_analytics.utils import (build_cytoscape_data, generate_paper_lookup)


DEFAULT_LAYOUT = "cose-bilkent"


def top_n_spanning_tree(graph_object, n):
    """Build a spanning tree with default top n nodes."""
    if n <= len(graph_object.nodes()):
        node_frequencies = {}
        for node in graph_object.nodes():
            node_frequencies[node] = len(graph_object.nodes[node]["paper"])

        nodes_to_include = top_n(node_frequencies, n)
    else:
        nodes_to_include = graph_object.nodes()

    tree = minimum_spanning_tree(
        graph_object.subgraph(nodes_to_include),
        "distance_npmi")
    return tree


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
                cluster_type: k
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


node_shape_option_list = [
    'ellipse',
    'triangle',
    'rectangle',
    'diamond',
    'pentagon',
    'hexagon',
    'heptagon',
    'octagon',
    'star',
    'polygon'
]

graph_layout_options = {
    'cose-bilkent': "good for trees",
    'circle': "",
    'klay': "good for path search",
    'dagre': "good for path search",
    'random': "",
    'grid': "",
    'preset': "pregenerated",
    'concentric': "",
    'breadthfirst': "",
    'cose': "",
    'cola': "",
    'spread': "",
    'euler': ""
}

node_frequency_type = [
    ("Frequency", "paper_frequency"),
    ("Degree", "degree_frequency"),
    ("PageRank", "pagerank_frequency")
]

cluster_type = [
    ("Entity Type", "entity_type"),
    ("Community by Frequency", "community_frequency"),
    ("Community by Mutual Information", "community_npmi")
]

edge_frequency_type = [
    ("Mutual Information", "npmi"),
    ("Raw Frequency", "frequency"),
]


class VisualizationApp(object):
    
    def __init__(self):
        self._app =  JupyterDash("allvis")
        self._app.add_bootstrap_links = True
        self._app.external_stylesheets = dbc.themes.CYBORG

        self._server = self._app.server

        self._graphs = {}
        self._current_graph = None
        
        # Components
        button_group = dbc.InputGroup([
            dbc.Button("Reset", color="primary", className="mr-1",id='bt-reset', style={"margin": "2pt"}),
            dbc.Tooltip(
                "Reset the display to default values",
                target="bt-reset",
                placement="bottom",
            ),
            dbc.Button("Remove Selected Node", color="primary", className="mr-1",id='remove-button', style={"margin": "2pt"}),
            dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem("png", id="png-menu"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("jpg", id="jpg-menu"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("svg", id="svg-menu"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem(
                        "gml", id="gml-menu", href="/download/graph.gml")
                ] + [Download(id="download-gml")],
                label="Download",
                id='dropdown-download',
                color="primary",
                group=True,
                className="mr-1",
                style={"margin": "2pt"}
            )
        ])

        buttons = dbc.FormGroup([button_group], className="mr-1")

        self.dropdown_items = dcc.Dropdown(
            id="showgraph",
            value=self._current_graph,
            options=[{'label': val.capitalize(), 'value': val} for val in self._graphs.values()],
            style={"width":"100%"})
        
        graph_type_dropdown = dbc.FormGroup(
            [
                dbc.Label("Display", html_for="showgraph"),
                self.dropdown_items
            ]
        )

        input_group = dbc.InputGroup([
            dbc.InputGroupAddon("Search", addon_type="prepend"),
                dcc.Dropdown(
                    id="searchdropdown",
                    multi=True,
                    style={"width":"80%"})],
            className="mb-3"
        )


        search = dbc.FormGroup(
            [
                dbc.Label("Search", html_for="searchdropdown", width=3),
                dbc.Col(dcc.Dropdown(
                    id="searchdropdown",
                    multi=True
                ), width=9)

            ],
            row=True
        )

        dropdown_menu_items = [
            dbc.DropdownMenuItem("Frequency",   id="dropdown-menu-freq-frequency"),
            dbc.DropdownMenuItem("Degree Frequency",   id="dropdown-menu-freq-degree_frequency"),
            dbc.DropdownMenuItem("PageRank Frequency", id="dropdown-menu-freq-pagerank_frequency")
        ]
        
        freq_input_group = dbc.InputGroup(
            [
                dbc.Label("Node Weight", html_for="node_freq_type"),
                dcc.Dropdown(
                    id="node_freq_type",
                    value="degree_frequency",
                    options=[{'label': val[0], 'value': val[1]} for val in node_frequency_type],
                    style={"width":"100%"})
            ],
            className="mb-1"
        )

        node_range_group = dbc.FormGroup(
            [
                dbc.Label("Display Range", html_for="nodefreqslider"),
                html.Div(
                    [dcc.RangeSlider(id="nodefreqslider_content", min=0, max=100000)],
                    id="nodefreqslider"
                )
            ]
        )
        
        edge_input_group = dbc.InputGroup(
            [
                 dbc.Label("Edge Weight", html_for="edge_freq_type"),
                 dcc.Dropdown(
                    id="edge_freq_type",
                    value="npmi",
                    options=[{'label': val[0], 'value': val[1]} for val in edge_frequency_type],
                    style={
                        "width":"100%"
                    }
                )
            ],
            className="mb-1"
        )

        edge_range_group = dbc.FormGroup(
            [
                dbc.Label("Display Range", html_for="edgefreqslider"),
#                 self.edge_freq_slider
                html.Div(
                    [dcc.RangeSlider(id="edgefreqslider_content", min=0, max=100000)],
                    id="edgefreqslider")
            ]
        )
        
        frequencies_form = dbc.FormGroup(
            [
                dbc.Col([freq_input_group, node_range_group], width=6),
                dbc.Col([edge_input_group, edge_range_group], width=6)
            ],
            row=True)

        display_message = html.P(
            "Displaying top 100 most frequent entities",
            id="display-message")
        
        top_n_button = dbc.Button(
            "Show N most frequent",
            color="primary", className="mr-1", id='top-n-button')
        
        top_n_slider = daq.NumericInput(
            id="top-n-slider",
            min=1,  
            max=1000,
            value=100,
            className="mr-1",
            disabled=False
        )
        show_all_button = dbc.Button(
            "Show all entities",
            id="show-all-button",
            color="primary", className="mr-1", style={"float": "right"})
        
        top_n_groups = dbc.InputGroup(
            [top_n_button, top_n_slider, show_all_button],
            style={"margin-bottom": "10pt"})
        
        item_details = dbc.FormGroup([html.Div(id="modal")])

        item_details_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("", className="card-title"),
                    html.H6("", className="card-subtitle"),
                    html.P("",className="card-text"),
                    dbc.Button("", color="primary", id ="see-more-card")
                ],
                id="item-card-body"
            )
        )

        form = dbc.Form([
            button_group,
            html.Hr(),
            graph_type_dropdown,
            display_message,
            top_n_groups,
            search, 
            html.Hr(),
            frequencies_form,
            item_details_card])
        
        # ------ Clustering form ------

        cluster_group = dbc.InputGroup(
            [
                dbc.Label("Cluster by", html_for="cluster_type"),
                dcc.Dropdown(
                    id="cluster_type",
                    value="entity_type",
                    options=[{'label': val[0], 'value': val[1]} for val in cluster_type],
                    style={"width":"100%"})
            ],
            className="mb-1"
        )
        
        legend = dbc.FormGroup([
            dbc.Label("Legend", html_for="cluster_board"),
            html.Div(id="cluster_board", children=[])
        ])
        
#         cluster_layout_button = dbc.InputGroup([
#             dbc.Checklist(
#                 options=[{"value": 1, "disabled": True}],
#                 value=[],
#                 id="groupedLayout",
#                 switch=True,
#             ),
#             dbc.Label("Grouped Layout", html_for="groupedLayout"),
#         ])        

        self.cluster_filter = dcc.Dropdown(
            id="clustersearch",
            multi=True,
            options=[],
            value="All"
        )
        
        filter_by_cluster = dbc.FormGroup(
            [
                dbc.Label("Clusters to display", html_for="clustersearch"),
                self.cluster_filter,
                dbc.Button(
                    "Add all clusters", color="primary", className="mr-1", id='addAllClusters',
                    style={"margin-top": "10pt", "float": "right"})
            ]
        )
        
        nodes_to_keep = dbc.FormGroup(
            [
                dbc.Label("Nodes to keep", html_for="nodestokeep"),
                dcc.Dropdown(
                    id="nodestokeep",
                    multi=True,
                    options=[],
                )
            ],
            style={"margin-top": "25pt"}
        )

        grouping_form = dbc.Form([
            cluster_group,
#             cluster_layout_button,
            html.Hr(),
            legend,
            html.Hr(),
            filter_by_cluster,
            nodes_to_keep
        ])
        
        # ------ Path search form --------
        
        path_from = dbc.FormGroup([
            dbc.Label("From", html_for="searchpathfrom", width=3),
            dbc.Col(dcc.Dropdown(
                id="searchpathfrom"
            ), width=9)],
            row=True)

        path_to = dbc.FormGroup([
            dbc.Label("To", html_for="searchpathto", width=3),
            dbc.Col(dcc.Dropdown(id="searchpathto"), width=9)],
            row=True)
        
        top_n_paths = dbc.FormGroup([
            dbc.Label("Top N", html_for="searchpathlimit", width=3),
            dbc.Col(
                daq.NumericInput(
                    id="searchpathlimit",
                    min=1,  
                    max=50,
                    value=10,
                   className="mr-1"
                ), width=9)], row=True)
    
        path_condition = dbc.FormGroup([
            dbc.Label("Traversal conditions"),
            dbc.FormGroup([
                    dbc.Col([
                        dbc.Label("Entity to traverse", html_for="searchnodetotraverse"),
                        dcc.Dropdown(id="searchnodetotraverse")
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Allow Overlap", html_for="searchpathoverlap"),
                        dbc.Checklist(
                            options=[{"value": 1}],
                            value=[1],
                            id="searchpathoverlap",
                            switch=True,
                        )
                    ], width=6)
                ], row=True)
        ])
            
        nested_path = dbc.FormGroup([
            dbc.Label("Nested path search"),
            dbc.FormGroup([
                dbc.Col(
                    children=[
                        dbc.Label("Nested", html_for="nestedpaths"),
                        dbc.Checklist(
                            options=[{"value": 1}],
                            value=[],
                            id="nestedpaths",
                            switch=True,
                        )
                    ], width=6
                ),
                dbc.Col([
                    dbc.Label("Depth", html_for="pathdepth"),
                    daq.NumericInput(
                        id="pathdepth",
                        min=1,  
                        max=4,
                        value=1,
                        disabled=True,
                        className="mr-1"
                    )
                ], width=6)
            ], row=True)
        ])
        
        
        search_path = dbc.InputGroup(
            [
                html.P("", id="noPathMessage", style={"color": "red", "margin-right": "10pt"}),
                dbc.Button("Find Paths", color="primary",
                           className="mr-1", id='bt-path', style={"float": "right"}),
                dbc.Tooltip(
                    "Find paths between selected entities",
                    target="bt-path",
                    placement="bottom",
                )
            ], style={"float": "right"}
        )

        form_path_finder = dbc.Form([
            path_from,
            path_to,
            top_n_paths,
            html.Hr(),
            path_condition,
            html.Hr(),
            nested_path,
            html.Hr(),
            search_path])

        graph_layout = dbc.FormGroup(
            [
                dbc.Label("Layout", html_for="searchdropdown", width=3),
                dbc.Col(dcc.Dropdown(
                    id ='dropdown-layout',
                    options = [
                        {
                            'label': "{}{}".format(
                                val.capitalize(),
                                " ({})".format(graph_layout_options[val]) if graph_layout_options[val] else ""
                            ),
                            'value': val
                        } for val in graph_layout_options.keys()
                    ],
                    value=DEFAULT_LAYOUT,
                    clearable=False
                ), width=9)
            ],
            row=True
        )

        node_shape = dbc.FormGroup(
            [
                dbc.Label("Node Shape", html_for="dropdown-node-shape", width=3),
                dbc.Col(dcc.Dropdown(
                    id='dropdown-node-shape',
                    value='ellipse',
                    clearable=False,
                    options = [{'label': val.capitalize(), 'value': val} for val in node_shape_option_list]
                ), width=9)

            ],
            row=True
        )

        link_color_picker = dbc.FormGroup(
            [
                dbc.Col(daq.ColorPicker(
                  id='input-follower-color',
                  value=dict(hex='#1375B3'),
                  label="Highlight Color"
                ))    
            ],
            row=True
        )

        conf_form = dbc.Form([graph_layout, node_shape, link_color_picker])
        
        # ---- Create a layout from components ----------------

        self.cyto = cyto.Cytoscape(
            id='cytoscape',
            elements=self._graphs[self._current_graph]["cytoscape"] if self._current_graph is not None else None,
            stylesheet=CYTOSCAPE_STYLE_STYLESHEET,
            style={"width": "100%", "height": "100%"})

        
        self._app.layout  = html.Div([
#             dcc.Store(id='memory', data={"current_layout": DEFAULT_LAYOUT}),
            dbc.Row([]),
            dbc.Row([
                dbc.Col([
                    html.Div(style=VISUALIZATION_CONTENT_STYLE, children=[self.cyto]), 
                    ], width=8),
                dbc.Col(html.Div(children=[
                    dbc.Button(
                        "Controls",
                        id="collapse-button",
                        color="primary",
                        style={
                            "margin": "10pt"
                        }
                    ),
                    dbc.Collapse(dbc.Tabs(id='tabs', children=[
                        dbc.Tab(
                            label='Details', label_style={"color": "#00AEF9", "border-radius":"4px"},
                            children=[dbc.Card(dbc.CardBody([form]))]),
                        dbc.Tab(
                            label="Clusters", label_style={"color": "#00AEF9", "border-radius":"4px"},
                            children=[dbc.Card(dbc.CardBody([grouping_form]))]),
                        dbc.Tab(
                            label='Layout', label_style={"color": "#00AEF9"},
                            children=[dbc.Card(dbc.CardBody([conf_form]))]),
                        dbc.Tab(
                            label='Path Finder', label_style={"color": "#00AEF9"},
                            children=[dbc.Card(dbc.CardBody([form_path_finder]))])]), id="collapse")
                    ]),
                    width=4
                )
            ])])

        self._app.config['suppress_callback_exceptions'] = True
        self._app.height = "800px"
        self._min_node_weight = None
        self._max_node_weight = None
        self._min_edge_weight = None
        self._max_edge_weight = None
        self._removed_nodes = set()

        # Store removed nodes and edges
        self._removed_nodes = set()
        self._removed_edges = set()

        self._current_layout = DEFAULT_LAYOUT


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

    def _update_cyto_graph(self, graph_id, new_cyto_repr, top_n_entities):
        self._graphs[graph_id]["cytoscape"] = new_cyto_repr
        self._graphs[graph_id]["top_n"] = top_n_entities
        
    def set_graph(self, graph_id, graph_object, tree_object=None,
                  positions=None, default_top_n=None):
        # Generate a paper lookup table
        paper_lookup = generate_paper_lookup(graph_object)

        # Build a spanning tree with default top n nodes
        if default_top_n is None:
            if tree_object is not None:
                tree = tree_object
            else:
                tree = top_n_spanning_tree(graph_object, len(graph_object.nodes()))
            top_n_entities = None
        else:
            if len(graph_object.nodes()) <= default_top_n and tree_object is not None:
                tree = tree_object
                top_n_entities = len(graph_object.nodes())
            else:
                tree = top_n_spanning_tree(graph_object, default_top_n)
                top_n_entities = default_top_n
                
        # Generate a cytoscape repr of the input graph
        cyto_repr = get_cytoscape_data(tree, positions=positions)

        if self._graphs is None:
            self._graphs = {}
        self._graphs[graph_id] = {
            "nx_object": graph_object,
            "positions": positions,
            "paper_lookup": paper_lookup,
        }
        
        self._update_cyto_graph(graph_id, cyto_repr, top_n_entities)
        self._update_weight_data(graph_id, cyto_repr)
        
        if tree_object:
            self._graphs[graph_id]["full_tree"] = tree_object
            
        if default_top_n:
            self._graphs[graph_id]["default_top_n"] = default_top_n

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

    def run(self, port):
        self._app.run_server(mode="jupyterlab", debug=True, width="100%", port=port)

    def set_list_papers_callback(self, func):
        self._list_papers_callback = func

    def set_entity_definitons(self, definition_dict):
        self._entity_definitions = definition_dict

        
visualization_app = VisualizationApp()



    
# ############################## CALLBACKS ####################################

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
    
    
def search(search_value,value, showgraph, diffs=[],
           cluster_type=None, cluster_search=None, nodes_to_keep=None,
           global_scope=False):
    res = []
    if global_scope:
        elements = get_cytoscape_data(
            visualization_app._graphs[showgraph]["nx_object"])
    else:
        elements = visualization_app._graphs[showgraph]["cytoscape"]
    
    if nodes_to_keep is None:
        nodes_to_keep = []

    for ele_data in elements:
        if 'name' in ele_data['data']:
            el_id = ele_data["data"]["id"]
            label = ele_data['data']['name']
            
            cluster_matches = False
            if cluster_type is not None and cluster_search is not None:
                if (cluster_type in ele_data["data"] and ele_data["data"][cluster_type] in cluster_search) or\
                       el_id in nodes_to_keep:
                    cluster_matches = True
            else:
                cluster_matches = True
            
            # Check if the name matches
            name_matches = False
            if (search_value in label) or (label in search_value) or el_id in (value or []) :
                if el_id not in diffs:
                    name_matches = True

            if cluster_matches and name_matches:
                res.append({"label": label, "value": el_id})
    return res


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
        min_value: (
            "{}".format(min_value)
            if isinstance(min_value, int) else
            "{:.2f}".format(min_value)
        ),
        max_value: (
            "{}".format(max_value)
            if isinstance(max_value, int) else
            "{:.2f}".format(max_value)
        ),
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
    return [
        el for el in input_elements
        if el["data"]["id"] in nodes_to_keep + edges_to_keep
    ]


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
        Output('dropdown-layout', 'value'),
        Output('cytoscape', 'zoom'),
        Output('cytoscape', 'elements'),
        Output('display-message', 'children'),
        Output("noPathMessage", "children"),
    ],
    [
        Input('bt-reset', 'n_clicks'),
        Input('remove-button', 'n_clicks'),
        Input('showgraph', 'value'),
        Input('nodefreqslider_content', 'value'),
        Input('edgefreqslider_content', 'value'),
        Input("searchdropdown", "value"),
        Input('bt-path', 'n_clicks'),
#         Input('groupedLayout', "value"),
        Input('cluster_type', "value"),
        Input('top-n-button', "n_clicks"),
        Input('show-all-button', "n_clicks"),
        Input("clustersearch", "value"),
        Input("nodestokeep", "value"),
        Input('cytoscape', 'selectedNodeData'),
        Input('cytoscape', 'selectedEdgeData'),
        Input('cytoscape', 'tapNodeData'),
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
        State('dropdown-layout', 'value')
    ]
)
def reset_layout(resetbt, removebt, val, 
                 nodefreqslider, edgefreqslider, 
                 searchvalues, pathbt, 
#                  grouped_layout, 
                 cluster_type, top_n_buttton, show_all_button, clustersearch,
                 nodes_to_keep, data, edge, tappednode, 
                 node_freq_type, edge_freq_type, cytoelements, zoom, searchpathfrom,
                 searchpathto, searchnodetotraverse, searchpathlimit, searchpathoverlap,
                 nestedpaths, pathdepth,  
                 top_n_slider_value, dropdown_layout):
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
        "nodefreqslider_content",
        "edgefreqslider_content",
        "nodestokeep",
        "clustersearch",
    ]
    
    if button_id in full_graph_events:
        elements = visualization_app._graphs[val]["cytoscape"]
#         if visualization_app._graphs[val]["top_n"] is None and\
#            visualization_app._graphs[val]["positions"] is not None:
#             visualization_app._current_layout = "preset"
#         else:
#             visualization_app._current_layout = DEFAULT_LAYOUT
    else:
#         if button_id == "bt-path":
#             visualization_app._current_layout = "klay"
        elements = cytoelements
        
    # Mark selected nodess and edges
    selected_nodes = [
        el["id"] for el in (data if data else [])
    ]
    
    if searchvalues is None:
        searchvalues = []

    selected_elements  = set()

    for searchvalue in set(searchvalues + selected_nodes):
        selected_elements.add(searchvalue)
 
    for el in elements:
        if el["data"]["id"] in selected_elements:        
            el["selected"] = True
    
    if nodes_to_keep is None:
        nodes_to_keep = []
        
    def to_keep(x):
        result = cluster_type in x and x[cluster_type] in clustersearch or x["id"] in nodes_to_keep
        return result
            
    if clustersearch is not None and cluster_type is not None and button_id != "bt-reset":
        elements = filter_elements(
            elements,
            node_condition=to_keep)

    if button_id == "bt-reset":
        visualization_app._removed_nodes = set()
        visualization_app._removed_edges = set()
 
    if button_id == "cluster_type":
        if len(grouped_layout) == 1:
            elements = generate_clusters(elements, cluster_type)

    if button_id == "groupedLayout":
        if len(grouped_layout) == 1:
            elements = generate_clusters(elements, cluster_type)
        else:
            elements = visualization_app._graphs[val]["cytoscape"]

    if button_id == "remove-button" and removebt is not None:
        nodes_to_remove = set()
        edges_to_remove = set()
        if elements and data:
            nodes_to_remove = {ele_data['id'] for ele_data in data}
            edges_to_remove = {
                el["data"]["id"]
                for el in elements
                if "source" in el["data"] and (
                    el["data"]["source"] in nodes_to_remove or
                    el["data"]["target"] in nodes_to_remove
                )
            }
        if elements and edge:
            edges_to_remove = {ele_data['id'] for ele_data in edge}
       
        visualization_app._removed_nodes.update(nodes_to_remove)
        visualization_app._removed_edges.update(edges_to_remove)
    
    no_path_message = ""
    if button_id == "bt-path" and pathbt is not None:
        visualization_app._removed_nodes = set()
        visualization_app._removed_edges = set()
        
        success = False
        if searchpathfrom and searchpathto:
            topN = searchpathlimit if searchpathlimit else 20
            
            source = searchpathfrom
            target = searchpathto
            
            # create a subgraph given the selected clusters 
            graph_object = visualization_app._graphs[val]["nx_object"]

            if cluster_type and clustersearch:
                graph_object = nx.Graph(graph_object.subgraph(
                    nodes=[
                        n
                        for n in graph_object.nodes()
                        if graph_object.nodes[n][cluster_type] in clustersearch or n in nodes_to_keep
                    ]))
                
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
                        strategy="naive", distance="distance_npmi", intersecting=intersecting, pretty_print=False)
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
                visualization_app._current_layout = "cic"
    
    current_top_n = visualization_app._graphs[val]["top_n"]
    total_number_of_entities = len(visualization_app._graphs[val]["nx_object"].nodes())

    message = (
        "Displaying top {} most frequent entities (out of {})".format(
            visualization_app._graphs[val]["top_n"], total_number_of_entities)
        if visualization_app._graphs[val]["top_n"] is not None
        else "Displaying all {} entities".format(total_number_of_entities)
    )
    
    if button_id == "top-n-button":
        if current_top_n is None or current_top_n != top_n_slider_value:
            tree = top_n_spanning_tree(
                visualization_app._graphs[val]["nx_object"],
                top_n_slider_value
            )
            elements = get_cytoscape_data(tree)
            visualization_app._update_cyto_graph(val, elements, top_n_slider_value, node_freq_type, edge_freq_type)
            visualization_app._update_weight_data(
                val, elements, node_freq_type, edge_freq_type)
            
        message = "Displaying top {} most frequent entities (out of {})".format(
            top_n_slider_value
            if top_n_slider_value <= total_number_of_entities
            else total_number_of_entities,
            total_number_of_entities)
    elif button_id == "show-all-button":
        if current_top_n is not None:
            # Top entities are selected, but button is clicked, so show all
            if "full_tree" in visualization_app._graphs[val]:
                tree = visualization_app._graphs[val]["full_tree"]
            else:
                tree = top_n_spanning_tree(
                    visualization_app._graphs[val]["nx_object"],
                    len(visualization_app._graphs[val]["nx_object"].nodes())
                )
            elements = get_cytoscape_data(tree, visualization_app._graphs[val]["positions"])

            visualization_app._update_cyto_graph(val, elements, None, node_freq_type, edge_freq_type)
            visualization_app._update_weight_data(
                val, elements, node_freq_type, edge_freq_type)

        message = "Displaying all {} entities".format(total_number_of_entities) 
                    
    def node_range_condition(el, start, end):
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
            elements = filter_elements(
                elements,
                lambda x: node_range_condition(x, nodefreqslider[0], nodefreqslider[1]),
                lambda x: edge_range_condition(
                    x,
                    visualization_app._graphs[val]["current_edge_value"][0],
                    visualization_app._graphs[val]["current_edge_value"][1]))
            visualization_app._graphs[val]["current_node_value"] = nodefreqslider

    elif edgefreqslider and button_id == "edgefreqslider_content":
        if edgefreqslider[0] != visualization_app._graphs[val]["current_edge_value"][0] or\
           edgefreqslider[1] != visualization_app._graphs[val]["current_edge_value"][1]:
            elements = filter_elements(
                elements,
                lambda x: node_range_condition(
                    x,
                    visualization_app._graphs[val]["current_node_value"][0],
                    visualization_app._graphs[val]["current_node_value"][1]),
                lambda x: edge_range_condition(
                    x, edgefreqslider[0], edgefreqslider[1]))
            visualization_app._graphs[val]["current_edge_value"] = edgefreqslider
  
    elements = [
        el for el in elements
        if el["data"]["id"] not in visualization_app._removed_nodes and\
            el["data"]["id"] not in visualization_app._removed_edges
    ]
    
     if button_id in full_graph_events:
        if visualization_app._graphs[val]["top_n"] is None and\
           visualization_app._graphs[val]["positions"] is not None:
            visualization_app._current_layout = "preset"
            for el in elements:
                if "source" not in el["data"]:
                    el["position"] = visualization_app._graphs[val]["positions"][el["data"]["id"]]
        else:
            visualization_app._current_layout = DEFAULT_LAYOUT
    else:
        if button_id == "bt-path":
            visualization_app._current_layout = "klay"

    return [
        visualization_app._current_layout,
        zoom, elements,
        message,
        no_path_message
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
        State('showgraph', 'value')
    ])
def display_tap_node(datanode, dataedge, statedatanode, statedataedge, showgraph):  
    papers = []
    res = []
    modal_button = None
    npmi_message =  None
    
    paper_lookup = visualization_app._graphs[visualization_app._current_graph]["paper_lookup"]
    
    if datanode and statedatanode:
        definition = ""
        if "definition" in datanode['data']:
            definition = str(datanode["data"]["definition"])
        elif datanode["data"]["id"] in visualization_app._entity_definitions:
            definition = visualization_app._entity_definitions[datanode["data"]["id"]]
        label = str(datanode['style']['label'])
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

        papers = set(paper_lookup[dataedge['data']['source']]).intersection(
            set(paper_lookup[dataedge['data']['target']]))
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
#             print(e)
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
                        html.H5(authors, className="card-subtitle"),
                        html.H6(journal + "( " + publish_time + " )", className="card-subtitle"),
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
        Input("cytoscape", "elements"),
        Input('dropdown-layout', 'value'),
    ],
    [
        State('showgraph', 'value')
    ])
def update_cytoscape_layout(elements, layout, current_graph):
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
    ],
    [
        State('cytoscape', 'stylesheet')
    ])
def generate_stylesheet(elements,
                        follower_color, node_shape,
                        showgraph, node_freq_type, edge_freq_type,
                        cluster_type, original_stylesheet):
    
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

    # Update selection styles to correspond to the follower
    selection_styles = [
        {
            "selector": "node:selected",
            "style": {
                "border-width": "5px",
                "border-color": follower_color['hex'],
                "opacity": 0.8,
                "text-opacity": 1,
                'z-index': 9999
            }
        }, {
            "selector": "edge:selected",
            "style": {
                "line-color": follower_color['hex'],
                "opacity": 0.8,
            }
        }
    ]
    stylesheet += selection_styles
        
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
        stylesheet.append({
            "selector": '[type = "cluster"]',
            "style": {
                "opacity": 0.2,
            },
        })
        
    if edge_freq_type:
        stylesheet = [style for style in stylesheet if not (style["selector"] == 'edge' and 'width' in style["style"])]
        stylesheet.append({
            "selector": 'edge',
            'style': {'width':'data(' + edge_freq_type + '_size)'}
        })

#     for focus_node in focus_nodes:
#         if "edgesData" not in focus_node:
#             print(focus_node)
#         node_style = [
#             {
#                 "selector": "node:selected",
#     #                 "selector": 'node[id = "{}"]'.format(focus_node['data']['id'] if "data" in focus_node else focus_node['id']),
#                 "style": {
#                     "border-width": "5px",
#                     "border-color": follower_color['hex'],
#                     "text-opacity": 1,
#                     'z-index': 9999
#                 }
#             }
#         ]
#         for style in node_style:
#             stylesheet.append(style)
        
        
#         if "edgesData" in focus_node:
#             for edge in focus_node['edgesData']:
#                 if edge['source'] == focus_node['data']['id'] if "data" in focus_node else focus_node['id']:
#                     stylesheet.append({
#                         "selector": 'node[id = "{}"]'.format(edge['target']),
#                         "style": {
#                             'opacity': 0.9
#                         }
#                     })
#                     stylesheet.append({
#                         "selector": 'edge[id= "{}"]'.format(edge['id']),
#                         "style": {
#                             "mid-target-arrow-color": follower_color['hex'],
#                             "line-color": follower_color['hex'],
#                             'opacity': 0.9,
#                             'z-index': 5000
#                         }
#                     })
#                 if edge['target'] == focus_node['data']['id'] if "data" in focus_node else focus_node['id']:
#                     stylesheet.append({
#                         "selector": 'node[id = "{}"]'.format(edge['source']),
#                         "style": {
#                             'opacity': 0.9,
#                             'z-index': 9999
#                         }
#                     })
#                     stylesheet.append({
#                         "selector": 'edge[id= "{}"]'.format(edge['id']),
#                         "style": {
#                             "mid-target-arrow-color": follower_color['hex'],
#                             "line-color": follower_color['hex'],
#                             'opacity': 1,
#                             'z-index': 5000
#                         }
#                     })

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
        search_value, value, showgraph, [], cluster_type, cluster_search, nodes_to_keep)


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
    return search(search_value, value, showgraph, [])


@visualization_app._app.callback(
        Output("clustersearch", "value"),
    [
        Input("showgraph", "value"),
        Input("cluster_type", "value"),
        Input("addAllClusters", "n_clicks"),
        Input("bt-reset", "n_clicks")
    ])
def prepopulate_value(val, cluster_type, add_all_clusters, reset_button):
    types = get_all_clusters(val, cluster_type)
    return types
    

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
        State("searchpathto", "value"),
        State('searchpathfrom', 'value'),
        State('showgraph', 'value'),
        State('cluster_type', "value"),
        State("clustersearch", "value"),
        State("nodestokeep", "value")
    ]
)
def searchpathto(search_value, value, node_from, showgraph,
                 cluster_type, cluster_search, nodes_to_keep):
    if not search_value:
        raise PreventUpdate    

    return search(
        search_value, value, showgraph, [node_from],
        cluster_type, cluster_search, nodes_to_keep, global_scope=True)


@visualization_app._app.callback(
    Output("searchnodetotraverse", "options"),
    [Input("searchnodetotraverse", "search_value")],
    [
        State("searchnodetotraverse", "value"),
        State('searchpathfrom', 'value'),
        State('searchpathto', 'value'),
        State('showgraph', 'value'),
        State('cluster_type', "value"),
        State("clustersearch", "value"),
        State("nodestokeep", "value")
    ]
)
def searchpathtraverse(search_value, value, node_from, to, showgraph,
                       cluster_type, cluster_search, nodes_to_keep):
    if not search_value:
        raise PreventUpdate
    return search(
        search_value, value, showgraph, [node_from, to],
        cluster_type, cluster_search, nodes_to_keep, global_scope=True)


@visualization_app._app.callback(
    Output("searchpathfrom", "options"),
    [Input("searchpathfrom", "search_value")],
    [
        State("searchpathfrom", "value"),
        State('showgraph', 'value'),
        State('cluster_type', "value"),
        State("clustersearch", "value"),
        State("nodestokeep", "value")
    ],
)
def searchpathfrom(search_value, value, showgraph,
                   cluster_type, cluster_search, nodes_to_keep):
    if not search_value:
        raise PreventUpdate
    return search(
        search_value, value, showgraph, [],
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


# @app.callback(Output("download", "data"), [Input("btn", "n_clicks")])
# def generate_xlsx(n_nlicks):

#     def to_xlsx(bytes_io):
#         xslx_writer = pd.ExcelWriter(bytes_io, engine="xlsxwriter")
#         df.to_excel(xslx_writer, index=False, sheet_name="sheet1")
#         xslx_writer.save()

#     return send_bytes(to_xlsx, "some_name.xlsx")


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
