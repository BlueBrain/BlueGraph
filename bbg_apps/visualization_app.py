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
                                COSE_BILKENT_CONFIG,
                                COSE_CONFIG,
                                COLA_CONFIG,
                                CISE_CONFIG,
                                COLORS)
from dash.exceptions import PreventUpdate

from kganalytics.paths import (top_n_paths, top_n_tripaths, top_n_nested_paths)
from kganalytics.utils import (top_n)


DEFAULT_TOP_N = 50

    
def generate_sizes(start, end, weights, func="linear"):
    sorted_indices = np.argsort(weights)
    if func == "linear":
        sizes = np.linspace(start, end, len(weights))
    elif func == "log":
        sizes = np.logspace(start, end, len(weights))
    return [
        int(round(sizes[el]))
        for el in sorted_indices
    ]


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
                cyto_repr[i]["data"]["{}_size".format(weight)] = sizes[j]
                if min_font_size and max_font_size:
                    cyto_repr[i]["data"]["{}_font_size".format(weight)] = font_sizes[j]
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
    

def create_edge(id, from_id, to_id, label=None, label_size=10, label_color="black", thickness=2, edge_color="grey", edge_style="solid",frequency=1,papers=[]):
    if thickness == 0:
        thickness = 2
    return {
        "data": { 
            "id": str(id),
            "source": str(from_id).lower(),
            "target": str(to_id).lower(),
            "frequency": frequency,
            "papers": papers
        },
        "style": {
           "label": label if label else '',
            "width": thickness
        }
    }


def create_node(id, node_type=None,label=None, label_size=10, label_color="black", radius=30, node_color='grey',frequency={}, definition="",papers=[]):
    actualLabel = None
    if label is not None:
        actualLabel = label.lower()
    else:
        actualLabel = str(id).lower().split("/")[-1].split("#")[-1]
    frequency_raw = frequency['frequency'] if 'frequency' in frequency else 1
    return {
        "data": { 
            "id": str(id).lower(),
            "frequency":frequency_raw,
            "degree_frequency":frequency['degree_frequency'] if 'degree_frequency' in frequency else frequency_raw,
            "pagerank_frequency":frequency['pagerank_frequency'] if 'pagerank_frequency' in frequency else frequency_raw,
            "definition":definition,
            "papers":papers,
            "type":node_type
        },
        "style": {
            "label": actualLabel
        }
    }

def create_edge(id, from_id, to_id, label=None, label_size=10, label_color="black", thickness=2, edge_color="grey", 
                edge_style="solid",frequency=1,papers=[]):

        if thickness == 0:
            thickness = 2
        return {
            "data": { 
                "id": str(id),
                "source": str(from_id).lower(),
                "target": str(to_id).lower(),
                "frequency":frequency,
                "papers":papers
            },
            "style": {
               "label": label if label else '',
                "width": thickness
            }
        }


def create_node(id, node_type=None,label=None, label_size=10, label_color="black", radius=30, node_color='grey',frequency={}, definition="",papers=[]):

        actualLabel = None
        if label is not None:
            actualLabel = label.lower()
        else:
            actualLabel = str(id).lower().split("/")[-1].split("#")[-1]
        frequency_raw = frequency['frequency'] if 'frequency' in frequency else 1
        return {
            "data": { 
                "id": str(id).lower(),
                "frequency":frequency_raw,
                "degree_frequency":frequency['degree_frequency'] if 'degree_frequency' in frequency else frequency_raw,
                "pagerank_frequency":frequency['pagerank_frequency'] if 'pagerank_frequency' in frequency else frequency_raw,
                "definition":definition,
                "papers":papers,
                "type":node_type
            },
            "style": {
                "label": actualLabel
            }
        }

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

dropdown_download_option_list = [
    'jpg',
    'png',
    'svg'
]

graph_layout_options = {
    'cose-bilkent': "good for trees",
    'circle': "good for full networks",
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
                "Reset the display to default valuess",
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
                 dbc.DropdownMenuItem("svg", id="svg-menu")
                ],
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
                dcc.RangeSlider(id="nodefreqslider", min=0, max=10, value=[0, 1000000])
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
                dcc.RangeSlider(id="edgefreqslider", min=0, max=10, value=[0, 100000])
            ]
        )
        
        frequencies_form = dbc.FormGroup(
            [
                dbc.Col([freq_input_group, node_range_group], width=6),
                dbc.Col([edge_input_group, edge_range_group], width=6)
            ],
            row=True)

        display_message = html.P(
            "Displaying top {} most frequent entities".format(DEFAULT_TOP_N),
            id="display-message")
        
        top_n_button = dbc.Button(
            "Show all entities",
            color="primary", className="mr-1", id='top-n-button')
        
        top_n_slider = daq.NumericInput(
            id="top-n-slider",
            min=1,  
            max=1000,
            value=DEFAULT_TOP_N,
            className="mr-1",
            disabled=True
        )
        top_n_groups = dbc.InputGroup(
            [top_n_button, top_n_slider],
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
                    value='cose-bilkent',
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
            dcc.Store(id='memory', data={"display_top": DEFAULT_TOP_N}),
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

    
    def set_graph(self, graph_id, cyto_repr, dict_repr, paper_lookup, positioned=False, nx_object=None):
        
        # add some extra attrs to nodes
        weights = ["paper_frequency", "degree_frequency", "pagerank_frequency"]
        set_sizes_from_weights(
            cyto_repr, weights, MIN_NODE_SIZE, MAX_NODE_SIZE,
            MIN_FONT_SIZE, MAX_FONT_SIZE)
        
        # add some extra attrs to nodes
        weights = ["npmi", "ppmi", "frequency"]
        set_sizes_from_weights(
            cyto_repr, weights, MIN_EDGE_WIDTH, MAX_EDGE_WIDTH)
        
        if self._graphs is None:
            self._graphs = {}
        self._graphs[graph_id] = {
            "cytoscape": cyto_repr,
            "dict": dict_repr,
            "positioned": positioned,
            "paper_lookup": paper_lookup
        }

        if nx_object is not None:
            self._graphs[graph_id]["nx_object"] = nx_object
            
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
        self._app.run_server(mode="jupyterlab", width="100%", port=port)

    def set_list_papers_callback(self, func):
        self._list_papers_callback = func

    def set_entity_definitons(self, definition_dict):
        self._entity_definitions = definition_dict

        
visualization_app = VisualizationApp()



    
# ############################## CALLBACKS ####################################

def get_cytoscape_data(factor,graph):
    elements = cytoscape_data(graph[factor])
    elements=elements["elements"]['nodes']+elements["elements"]['edges']
    for element in elements:
        element["data"]["id"] = str(element["data"]["source"]+'_'+element["data"]["target"]).replace(" ","_") if "source" in element["data"] else str(element["data"]["id"]).replace(" ","_")
    elements_dict = {element["data"]["id"]: element for element in elements}
    return elements, elements_dict
    
    
def search(search_value,value, showgraph, diffs=[],
           cluster_type=None, cluster_search=None, nodes_to_keep=None):
    res = []
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


@visualization_app._app.callback(
    [
        Output('nodefreqslider', 'min'),
        Output('nodefreqslider', 'max'),
        Output('nodefreqslider', 'marks'),
        Output('nodefreqslider', 'step'),
        Output('edgefreqslider', 'min'),
        Output('edgefreqslider', 'max'),
        Output('edgefreqslider', 'marks'),
        Output('edgefreqslider', 'step')
    ],
    [
        Input('showgraph', 'value'),
        Input('node_freq_type', 'value'),
        Input('edge_freq_type', 'value'),
    ],
    [
        State('cytoscape', 'elements')
    ])
def adapt_weight_ranges(val, node_freq_type, edge_freq_type, cytoelements):
    elements = cytoelements
    if elements:
        # here set min and max if not set yet
        if node_freq_type or (
                visualization_app._min_node_weight is None and\
                visualization_app._max_node_weight is None):
            min_node_value, max_node_value, node_marks, node_step = recompute_node_range(
                elements, node_freq_type)
            visualization_app._min_node_weight = min_node_value
            visualization_app._max_node_weight = max_node_value
            node_value = [min_node_value, max_node_value]
            
        if edge_freq_type or (
                visualization_app._min_edge_weight is None and\
                visualization_app._max_edge_weight is None):
            min_edge_value, max_edge_value, edge_marks, edge_step = recompute_node_range(
                elements, edge_freq_type)
            
            visualization_app._min_edge_weight = min_edge_value
            visualization_app._max_edge_weight = max_edge_value
            edge_value = [min_edge_value, max_edge_value]
    
        return  [
            min_node_value, max_node_value, node_marks, node_step,
            min_edge_value, max_edge_value, edge_marks, edge_step
        ]    


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
        Output('cytoscape', 'zoom'),
        Output('cytoscape', 'elements'),
        Output('top-n-button', 'children'),
        Output('top-n-slider', 'disabled'),
        Output('memory', 'data'),
        Output('display-message', 'children'),
        Output("noPathMessage", "children")
    ],
    [
        Input('bt-reset', 'n_clicks'),
        Input('remove-button', 'n_clicks'),
        Input('showgraph', 'value'),
        Input('nodefreqslider', 'value'),
        Input('edgefreqslider', 'value'),
        Input("searchdropdown", "value"),
        Input('bt-path', 'n_clicks'),
#         Input('groupedLayout', "value"),
        Input('cluster_type', "value"),
        Input('top-n-button', "n_clicks"),
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
        State('memory', 'data'),
        State('top-n-slider', 'value'),
    ]
)
def reset_layout(resetbt, removebt, val, 
                 nodefreqslider, edgefreqslider, 
                 searchvalues, pathbt, 
#                  grouped_layout, 
                 cluster_type, top_n_buttton, clustersearch,
                 nodes_to_keep, data, edge, tappednode, 
                 node_freq_type, edge_freq_type, cytoelements, zoom, searchpathfrom,
                 searchpathto, searchnodetotraverse, searchpathlimit, searchpathoverlap,
                 nestedpaths, pathdepth, memory_data, top_n_slider_value):
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
        "bt-reset",
        "nodefreqslider",
        "edgefreqslider",
        "nodestokeep",
        "clustersearch"
    ]
    if button_id in full_graph_events:
        elements = visualization_app._graphs[val]["cytoscape"]
    else:
        elements = cytoelements

    elements_dict = visualization_app._graphs[val]["dict"]
    
    if button_id == "selectedNodeData" or button_id == "selectedEdgeData" or button_id == "tapNodeData":
        elements = cytoelements
        elements_dict = visualization_app._graphs[val]["dict"]

    if button_id == 'showgraph':
        visualization_app.set_current_graph(val)
        elements = visualization_app._graphs[val]["cytoscape"]
        elements_dict = visualization_app._graphs[val]["dict"]
        
    # Mark selected nodess and edges
    selected_nodes = [
        el["id"] for el in (data if data else [])
    ]
    
    if searchvalues is None:
        searchvalues = []

    selected_elements  = set()

    for searchvalue in set(searchvalues + selected_nodes):
        selected_elements.add(searchvalue)
#         search_node = elements_dict[searchvalue]
#         for el in elements:
#             if el["data"]["id"] not in selected_elements:
#                 if "source" in el["data"]:
#                     if el["data"]["source"] == searchvalue or\
#                        el["data"]["target"] == searchvalue:
#                         selected_elements.add(el["data"]["id"])
 
    for el in elements:
        if el["data"]["id"] in selected_elements:        
            el["selected"] = True
    
    if nodes_to_keep is None:
        nodes_to_keep = []
            
    if clustersearch is not None and cluster_type is not None and button_id != "bt-reset":
        elements = filter_elements(
            elements,
            node_condition=lambda x: cluster_type in x and x[cluster_type] in clustersearch or x["id"] in nodes_to_keep
        )

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
        
        if searchpathfrom and searchpathto:
            topN = searchpathlimit if searchpathlimit else 20
            searchpathfrom_dict = elements_dict[searchpathfrom]
            searchpathto_dict = elements_dict[searchpathto]
            
            source = searchpathfrom_dict['data']['name']
            target = searchpathto_dict['data']['name']
            
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
                    searchnodetotraverse_dict = elements_dict[searchnodetotraverse]

                    intersecting = len(searchpathoverlap) == 1
                    a_b_paths, b_c_paths = top_n_tripaths(
                        graph_object, source,
                        searchnodetotraverse_dict['data']['name'], target, topN,
                        strategy="naive", distance="distance_npmi", intersecting=intersecting, pretty_print=False)
                    paths = a_b_paths + b_c_paths
                else:
                    paths = top_n_paths(
                        graph_object, source, target,
                        topN, distance="distance_npmi", strategy="naive", pretty_print=False)
                elements = []


                if paths:
                    elements.append(searchpathfrom_dict) 
                    elements.append(searchpathto_dict)

                visited = set()
                for path in paths:
                    path_steps = list(path)
                    searchpathfrom = searchpathfrom_dict["data"]["id"]
                    
                    s = searchpathfrom
                    for index, path_step in enumerate(path_steps):
                        t = path_step
                        
                        path_element = elements_dict[path_step]
                        elements.append(path_element)
                        
                        if s != t and (s, t) not in visited and (t, s) not in visited:
                            edge_from_id = str(s).lower().replace(" ","_") + "_" + str(t).lower()
                            edge_from = create_edge(edge_from_id, s, t)
                            elements.append(edge_from)
                            visited.add((s, t))
                        s = path_step

            except ValueError as e:
                print(e)
                no_path_message = "No undirect paths from '{}' to '{}' were found (the nodes are either disconnected or connected by a direct edge only)".format(
                    source, target)

    result_memory_data = {}  

    if memory_data["display_top"] is not None:
        if button_id == "top-n-button":
            # Top entities are selected, but button is clicked, so show all
            result_memory_data["display_top"] = None
            top_n_button_label = "Show N most frequent entities"
            top_n_button_disabled = False
            message = "Displaying all {} entities".format(
                len([el for el in elements if "source" not in el["data"]]))
        else:
            nodes_to_select = top_n(
                {
                    el["data"]["id"]: el["data"]["paper_frequency"]
                    for el in elements
                    if "paper_frequency" in el["data"]
                },
                memory_data["display_top"])
            # Top entities are selected, button is not clicked, show top
            elements = filter_elements(elements, lambda x: x["id"] in nodes_to_select)
            
            top_n_button_label = "Show all entities"
            top_n_button_disabled = True
            result_memory_data["display_top"] = memory_data["display_top"]
            
            total_number_of_entities = len(
                [el for el in visualization_app._graphs[val]["cytoscape"] if "source" not in el["data"]])
            
            message = "Displaying top {} most frequent entities (out of {})".format(
                memory_data["display_top"], total_number_of_entities)
    else:
        if button_id == "top-n-button":
            # Top entities are NOT selected but the button is clicked, show top
            nodes_to_select = top_n(
                {
                    el["data"]["id"]: el["data"]["paper_frequency"]
                    for el in elements
                    if "paper_frequency" in el["data"]
                },
                top_n_slider_value)
            elements = filter_elements(elements, lambda x: x["id"] in nodes_to_select)
            
            top_n_button_label = "Show all entities"
            top_n_button_disabled = True
            result_memory_data["display_top"] = top_n_slider_value
                  
            total_number_of_entities = len(
                [el for el in visualization_app._graphs[val]["cytoscape"] if "source" not in el["data"]])
            message = "Displaying top {} most frequent entities (out of {})".format(
                top_n_slider_value, total_number_of_entities)
        else:
            # Top entities are NOT selected and buttopn is not clicked
            top_n_button_label = "Show N most frequent entities"
            top_n_button_disabled = False
            result_memory_data["display_top"] = None
            message = "Displaying all {} entities".format(
                len([el for el in elements if "source" not in el["data"]]))
   
                    
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

    if nodefreqslider and button_id == "nodefreqslider":
        if visualization_app._min_node_weight is None or\
           visualization_app._max_node_weight is None:
            visualization_app._min_node_weight = nodefreqslider[0]
            visualization_app._max_node_weight = nodefreqslider[1]
        elif nodefreqslider[0] != visualization_app._min_node_weight or\
             nodefreqslider[1] != visualization_app._max_node_weight:

            visualization_app._min_node_weight = nodefreqslider[0]
            visualization_app._max_node_weight = nodefreqslider[1]

            elements = filter_elements(
                elements,
                lambda x: node_range_condition(x, nodefreqslider[0], nodefreqslider[1]),
                lambda x: edge_range_condition(
                    x, visualization_app._min_edge_weight, visualization_app._max_edge_weight))

    elif edgefreqslider and button_id == "edgefreqslider":
        if visualization_app._min_edge_weight is None or\
           visualization_app._max_edge_weight is None:
            visualization_app._min_edge_weight = edgefreqslider[0]
            visualization_app._max_edge_weight = edgefreqslider[1]
        elif edgefreqslider[0] != visualization_app._min_edge_weight or\
             edgefreqslider[1] != visualization_app._max_edge_weight:
            
            visualization_app._min_edge_weight = edgefreqslider[0]
            visualization_app._max_edge_weight = edgefreqslider[1]
    
            elements = filter_elements(
                elements,
                lambda x: node_range_condition(
                    x, visualization_app._min_node_weight, visualization_app._max_node_weight),
                lambda x: edge_range_condition(
                    x, edgefreqslider[0], edgefreqslider[1]))
  
    elements = [
        el for el in elements
        if el["data"]["id"] not in visualization_app._removed_nodes and\
            el["data"]["id"] not in visualization_app._removed_edges
    ]

    return [
        zoom, elements,
        top_n_button_label, top_n_button_disabled, result_memory_data, message,
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
            print(e)
            print(datanode['data'])

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
    elements_dict = visualization_app._graphs[showgraph]["dict"]
        
    if dataedge and statedataedge:
        label = str(dataedge['style']['label'])
        
        source_node = elements_dict[ dataedge['data']['source']]
        source_label = source_node['data']['name']
        target_node = elements_dict[ dataedge['data']['target']]
        target_label = target_node['data']['name']

        papers = set(paper_lookup[source_node['data']['id']]).intersection(
            set(paper_lookup[target_node['data']['id']]))
        frequency = str(len(papers))
        mention_label= ''' '%s' mentioned with '%s' in %s papers''' % (source_label, target_label, frequency) 
        label = mention_label if str(dataedge['style']['label']) == "" else str(dataedge['style']['label']) 
        modal_button= dbc.Button(label, id="open-body-scroll",color="primary")

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



@visualization_app._app.callback(Output('cytoscape', 'layout'),
    [
        Input('dropdown-layout', 'value'),
        Input('showgraph', 'value'),
        Input("cytoscape", "elements")
    ],
    [
        State("cytoscape", "stylesheet")
    ])
def update_cytoscape_layout(layout, showgraph, elements, styles):
    
    if visualization_app._graphs[showgraph]["positioned"] is True:
        return {'name': 'preset'}
    if layout == "cose":
        layout_config = COSE_CONFIG
    elif layout =="cola":
        layout_config = COLA_CONFIG
    elif layout == "cose-bilkent":
        layout_config = COSE_BILKENT_CONFIG
    else:    
        layout_config = {'showlegend':True}

    layout_config["name"] = layout

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
        
    if node_freq_type or node:
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
    return search(search_value, value, showgraph, [], cluster_type, cluster_search, nodes_to_keep)


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
        Input("cluster_type", "value"),
        Input("addAllClusters", "n_clicks"),
        Input("bt-reset", "n_clicks")
    ],
    [
        State('showgraph', 'value'),
        State("clustersearch", "options"),
    ])
def prepopulate_value(cluster_type, add_all_clusters, reset_button, current_graph, options):
    types = set([
        el["data"][cluster_type]
        for el in visualization_app._graphs[current_graph]["cytoscape"]
        if cluster_type in el["data"]
    ])
    return list(types)
    

@visualization_app._app.callback(
    Output("clustersearch", "options"),
    [
        Input("cluster_type", "value"),
        Input("clustersearch", "search_value")
    ],
    [
        State("clustersearch", "value"),
        State('showgraph', 'value'),
    ],
)
def update_cluster_search(cluster_type, search_value, value, current_graph):
    types = set([
        el["data"][cluster_type]
        for el in visualization_app._graphs[current_graph]["cytoscape"]
        if cluster_type in el["data"]
    ])
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
        cluster_type, cluster_search, nodes_to_keep)


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
        cluster_type, cluster_search, nodes_to_keep)


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
        cluster_type, cluster_search, nodes_to_keep)


# @visualization_app._app.callback(Output('nodefreqslider', 'value'),
#               [Input('bt-reset', 'n_clicks')],[State('nodefreqslider', 'value')])
# def display_freq_node(resetbt, nodefreqslider):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         button_id = 'No clicks yet'
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     if button_id == 'bt-reset':
#         return 1


# @visualization_app._app.callback(Output('edgefreqslider', 'value'),
#               [Input('bt-reset', 'n_clicks')],[State('edgefreqslider', 'value')])
# def display_freq_edge(resetbt, edgefreqslider):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         button_id = 'No clicks yet'
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     if button_id == 'bt-reset':
#         return 1


@visualization_app._app.callback(
    [
        Output('cytoscape', 'generateImage')
    ],
    [
        Input('jpg-menu', 'n_clicks'),
        Input('svg-menu', 'n_clicks'),
        Input('png-menu', 'n_clicks')
    ]
)
def download_image(jpg_menu,svg_menu,png_menu):
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
    [
        Output('cluster_board', "children")
    ],
    [
        Input('cluster_type', "value")
    ],
    [
        State('cytoscape', 'elements')
    ])
def generate_legend(cluster_type, elements):
    types = set([
        el['data'][cluster_type] for el in elements if cluster_type in el['data']
    ])
    
    children = []
    for t in types:
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
