import math
import numpy as np

from operator import ge, gt, lt, le, eq, ne

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
                                MIN_NODE_SIZE,
                                MAX_NODE_SIZE,
                                MIN_FONT_SIZE,
                                MAX_FONT_SIZE,
                                MIN_EDGE_WIDTH,
                                MAX_EDGE_WIDTH)
from dash.exceptions import PreventUpdate

    
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
<<<<<<< HEAD

=======
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616


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

graph_layout_option_list = [
    'preset',
    'random',
    'grid',
    'circle',
    'concentric',
    'breadthfirst',
    'cose',
    'cose-bilkent',
    'dagre',
    'cola',
    'klay',
    'spread',
    'euler'
]

node_frequency_type = [
    ("Frequency", "paper_frequency"),
    ("Degree", "degree_frequency"),
    ("PageRank", "pagerank_frequency")
]

edge_frequency_type = [
    ("Mutual Information", "npmi"),
    ("Raw Frequency", "frequency"),
]

# graph_type_option_list = [
#     'Knowledge Graph',
#     'Co-mention Graph Spanning Tree',
#     'Co-mention Graph Cluster',
#     '3000-cluster','3000-spanning'
# ]


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
            dbc.Button("Reset", color="primary", className="mr-1",id='bt-reset'),
            dbc.Tooltip(
                "Reset the display to default valuess",
                target="bt-reset",
                placement="bottom",
            ),
            dbc.Button("Remove Selected Node", color="primary", className="mr-1",id='remove-button'),
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
                className="mr-1"
            )
        ])

        buttons = dbc.FormGroup([button_group], className="mr-1")

        self.radio_items = dbc.RadioItems(
            id="showgraph",
            value=self._current_graph,
            options=[{'label': val.capitalize(), 'value': val} for val in self._graphs.values()],
            inline=True
        )
        
        graph_type_radio = dbc.FormGroup([
            dbc.Label("Display", html_for="showgraph", width=3),
            dbc.Col(self.radio_items, width=9)], row=True)

        scope_option_list = ['Paper', 'Section', 'Paragraph']

        scope_radio = dbc.FormGroup(
            [
                dbc.Label("Scope", html_for="graphscope", width=3),
                dbc.Col(
                    dbc.RadioItems(
                        id="graphscope",
                        value='Paper',
                        options=[{'label': val.capitalize(), 'value': val} for val in scope_option_list],
                        inline=True
                    ), width=9
                )
            ],
            row=True
        )

        group_option_list = ['None','Type', 'Mutual Information']

        group_radio = dbc.FormGroup(
            [
                dbc.Label("Group By", html_for="graphgroup", width=3),
                dbc.Col(
                    dbc.RadioItems(
                        id="graphgroup",
                        value='None',
                        options=[{'label': val.capitalize(), 'value': val} for val in group_option_list],
                        inline=True
                    ), width=9
                )
            ],
            row=True
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
<<<<<<< HEAD
        
        
        
=======
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
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
<<<<<<< HEAD

=======
        
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
        node_range_group = dbc.FormGroup(
            [
                dbc.Label("Display Range", html_for="nodefreqslider"),
                dcc.RangeSlider(id="nodefreqslider", min=0, max=0, value=[0, 0]),
            ]
        )
        
        node_weight_form = dbc.FormGroup([
            freq_input_group,
            node_range_group
        ])
        
#         node_slider = dbc.InputGroup(
#             [
#                 freq_input_group,
# #                 dcc.Dropdown(
# #                     id='node-freq-filter',
# #                     value="ge",
# #                     clearable=False,
# #                     options=DROPDOWN_FILTER_LIST,
# #                     className="mr-1"
# #                 ),
#     #             daq.NumericInput(
#     #                 id="nodefreqslider",
#     #                 min=1,  
#     #                 max=10000,
#     #                 value=1,
#     #                className="mr-1"
#     #             )],
#                 dcc.RangeSlider(id="nodefreqslider", min=0, max=10, value=[3, 7])
#             ],
#             className="mb-3"
#         )

        
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
<<<<<<< HEAD
        )
        
        edge_range_group = dbc.FormGroup(
            [
                dbc.Label("Display Range", html_for="edgefreqslider"),
                dcc.RangeSlider(id="edgefreqslider", min=0, max=10, value=[0, 10]),
            ]
        )
=======
        )
        
        edge_range_group = dbc.FormGroup(
            [
                dbc.Label("Display Range", html_for="edgefreqslider"),
                dcc.RangeSlider(id="edgefreqslider", min=0, max=10, value=[0, 10]),
            ]
        )
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
    
        edge_weight_form = dbc.FormGroup([
            edge_input_group,
            edge_range_group
        ])
#         edge_slider = dbc.InputGroup([
#             edge_input_group,
#             dcc.Dropdown(
#                 id='edge-freq-filter',
#                 value="ge",
#                 clearable=False,
#                 options=DROPDOWN_FILTER_LIST,
#                 className="mr-1"
#             ),
#             daq.NumericInput(
#                 id="edgefreqslider",
#                 min=1,  
#                 max=10000,
#                 value=1,
#                className="mr-1"
#             )],
#             className="mb-3"
#         )
        
        frequencies_form = dbc.FormGroup(
            [
                dbc.Col(node_weight_form, width=6),
                dbc.Col(edge_weight_form, width=6)
            ],
            row=True)

 
        item_details = dbc.FormGroup([html.Div(id="modal")])

        item_details_card = dbc.Card(
            dbc.CardBody([
                html.H5("", className="card-title"),
                html.H6("", className="card-subtitle"),
                html.P("",className="card-text"),
                dbc.Button("", color="primary", id ="see-more-card")],
                id = "item-card-body")
        )

        form = dbc.Form([
            button_group,
            html.Hr(),
            graph_type_radio,
            search, 
            html.Hr(),
            frequencies_form,
            item_details_card])

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

        path_condition = dbc.FormGroup([
            dbc.Label("Constraints"),
            dbc.FormGroup([
                dbc.Label("Traverse", html_for="searchnodetotraverse", width=3),
                dbc.Col(dcc.Dropdown(
                    id="searchnodetotraverse"
                ), width=9)],
                row=True
            ),
            dbc.FormGroup([
                dbc.Label("Allow Overlap", html_for="searchpathoverlap", width=3),
                dbc.Col(
                    dbc.Checklist(
                        options=[{"value": 1}],
                        value=[1],
                        id="searchpathoverlap",
                        switch=True,
                    ),
                    width=9)
                ], row=True),
            dbc.FormGroup([
                dbc.Label("Top N", html_for="searchpathlimit", width=3),
                dbc.Col(daq.NumericInput(
                    id="searchpathlimit",
                    min=10,  
                    max=20,
                    value=10,
                   className="mr-1"
                ), width=9)
            ], row=True)
        ])

        search_path = dbc.InputGroup(
            [
                dbc.Button("Find Paths", color="primary", className="mr-1",id='bt-path'),
                dbc.Tooltip(
                    "Find paths between selected entities",
                    target="bt-path",
                    placement="bottom",
                )
            ]
        )

        form_path_finder = dbc.Form([path_from,path_to,path_condition,search_path])

        graph_layout = dbc.FormGroup(
            [
                dbc.Label("Layout", html_for="searchdropdown", width=3),
                dbc.Col(dcc.Dropdown(
                    id ='dropdown-layout',
                    options = [{'label': val.capitalize(), 'value': val} for val in graph_layout_option_list],
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
                  value=dict(hex='#a0b3dc'),
                  label="Edge Color"
                ))    
            ],
            row=True
        )

        conf_form = dbc.Form([graph_layout,node_shape,link_color_picker])
        
        # ---- Create a layout from components ----------------
        self.cyto = cyto.Cytoscape(
            id='cytoscape',
            elements=self._graphs[self._current_graph]["cytoscape"] if self._current_graph is not None else None,
            stylesheet=CYTOSCAPE_STYLE_STYLESHEET,
            style={"width": "100%", "height": "100%"})
<<<<<<< HEAD

        
        self._app.layout  = html.Div([
            dcc.Store(id='memory', data={"removed":[]}),
=======
        
        self._app.layout  = html.Div([
            dcc.Store(id='memory',data={"removed":[]}),
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
            dbc.Row([]),
            dbc.Row([
                dbc.Col(dcc.Loading(
                        id="loading-graph",
                        children=[html.Div(
                            style=VISUALIZATION_CONTENT_STYLE, children=[self.cyto])],
                        type="circle",
                        style={"margin-top": "75pt"}),
                    width=8),
                dbc.Col(html.Div(children=[
                    dbc.Button(
                        "Controls",
                        id="collapse-button",
                        color="primary",
                    ),
                    dbc.Collapse(dbc.Tabs(id='tabs', children=[
                        dbc.Tab(
                            label='Details', label_style={"color": "#00AEF9", "border-radius":"4px"},
                            children=[dbc.Card(dbc.CardBody([form]))]),
                        dbc.Tab(
                            label='Graph Layout and Shape', label_style={"color": "#00AEF9"},
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

    
    def set_graph(self, graph_id, cyto_repr, dict_repr, style=None):
        
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
            "dict": dict_repr
        }
        if style is not None:
            self._graphs[graph_id]["style"] = style
        self.radio_items.options = [
            {'label': val.capitalize(), 'value': val} for val in list(self._graphs.keys())
        ]
<<<<<<< HEAD
=======

>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
        return
    
    def set_current_graph(self, graph_id):
        self._current_graph = graph_id
        self.radio_items.value = graph_id
        self.cyto.elements = self._graphs[self._current_graph]["cytoscape"]

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
    elements_dict = {element["data"]["id"]:element for element in elements  }
    return elements, elements_dict
    
    
def search(search_value,value, showgraph, diffs=[]):
    res = []
    elements = visualization_app._graphs[showgraph]["cytoscape"]

    for ele_data in elements:
        if 'name' in ele_data['data']:
            label =ele_data['data']['name']
            if (search_value in label) or (label in search_value) or ele_data['data']['id'] in (value or []) :
                if ele_data['data']['id'] not in diffs:
                    res.append({"label":ele_data['data']['name'],"value":ele_data['data']['id']})
    return res

<<<<<<< HEAD

=======
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
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


@visualization_app._app.callback(
    [
        Output('nodefreqslider', 'min'),
        Output('nodefreqslider', 'max'),
        Output('nodefreqslider', 'marks'),
        Output('nodefreqslider', 'value'),
        Output('nodefreqslider', 'step'),
        Output('edgefreqslider', 'min'),
        Output('edgefreqslider', 'max'),
        Output('edgefreqslider', 'marks'),
        Output('edgefreqslider', 'value'),
        Output('edgefreqslider', 'step')
    ],
    [
        Input('bt-reset', 'n_clicks'),
        Input('remove-button', 'n_clicks'),
        Input('showgraph', 'value'),
        Input('node_freq_type', 'value'),
        Input('edge_freq_type', 'value'),
        Input("searchdropdown", "value"),
        Input('bt-path', 'n_clicks')
    ],
    [
        State('cytoscape', 'elements')
    ])
def adapt_weight_ranges(resetbt, removebt, val, node_freq_type, edge_freq_type,
                        searchvalues, pathbt, cytoelements):
    elements = cytoelements
    if elements:
        # here set min and max if not set yet
        if node_freq_type or (
                visualization_app._min_node_weight is None and\
                visualization_app._max_node_weight):
            min_node_value, max_node_value, node_marks, node_step = recompute_node_range(
                elements, node_freq_type)
            visualization_app._min_node_weight = min_node_value
            visualization_app._max_node_weight = max_node_value
            node_value = [min_node_value, max_node_value]
            
        if edge_freq_type or (
                visualization_app._min_edge_weight is None and\
                visualization_app._max_edge_weight):
            min_edge_value, max_edge_value, edge_marks, edge_step = recompute_node_range(
                elements, edge_freq_type)
            
            visualization_app._min_edge_weight = min_edge_value
            visualization_app._max_edge_weight = max_edge_value
            edge_value = [min_edge_value, max_edge_value]
    
    return  [
        min_node_value, max_node_value, node_marks, node_value, node_step,
        min_edge_value, max_edge_value, edge_marks, edge_value, edge_step
    ]    
<<<<<<< HEAD

=======
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616

 
@visualization_app._app.callback(
    [
        Output('cytoscape', 'zoom'),
        Output('cytoscape', 'elements'),
    ],
    [
        Input('bt-reset', 'n_clicks'),
        Input('remove-button', 'n_clicks'),
        Input('showgraph', 'value'),
        Input('nodefreqslider', 'value'),
        Input('edgefreqslider', 'value'),
        Input("searchdropdown", "value"),
        Input('bt-path', 'n_clicks')
    ],
    [
        State('node_freq_type', 'value'),
        State('edge_freq_type', 'value'),
        State('cytoscape', 'elements'),
        State('cytoscape', 'selectedNodeData'),
        State('cytoscape', 'selectedEdgeData'),
        State('cytoscape', 'tapNodeData'),
        State('cytoscape', 'zoom'),
        State('searchpathfrom', 'value'),
        State('searchpathto', 'value'),
        State('searchnodetotraverse', 'value'),
        State('searchpathlimit', 'value'),
        State('searchpathoverlap', 'value')     
    ]
)
def reset_layout(resetbt, removebt, val, nodefreqslider, edgefreqslider, searchvalues, pathbt, 
                 node_freq_type, edge_freq_type, cytoelements, data, edge,
                 tappednode, zoom, searchpathfrom,
                 searchpathto, searchnodetotraverse, searchpathlimit, searchpathoverlap):
<<<<<<< HEAD
    elements = visualization_app._graphs[val]["cytoscape"]
    elements_dict = visualization_app._graphs[val]["dict"]
=======
    global removed 
    global elements_dict
    elements = visualization_app._graphs[val]["cytoscape"]

>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'showgraph':
        visualization_app.set_current_graph(val)
        elements = visualization_app._graphs[val]["cytoscape"]
        elements_dict = visualization_app._graphs[val]["dict"]

    if searchvalues is not None:
        for searchvalue in searchvalues:
            search_node = elements_dict[searchvalue]
            search_node["selected"] = True

    if button_id == 'remove-button':
        print("Here, remove button clicked")
        if elements and data:
            ids_to_remove = {ele_data['id'] for ele_data in data}
        if elements and edge:
            ids_to_remove = {ele_data['id'] for ele_data in edge}
        
        print(ids_to_remove)
        elements = [ele for ele in elements if ele['data']['id'] not in ids_to_remove]

    if button_id == 'bt-path':
        if searchpathfrom and searchpathto:
            topN = searchpathlimit if searchpathlimit else 20
            searchpathfrom_dict = elements_dict[searchpathfrom]
            searchpathto_dict = elements_dict[searchpathto]
            
            if searchnodetotraverse:
                searchnodetotraverse_dict = elements_dict[searchnodetotraverse]
                
                intersecting = len(searchpathoverlap) == 1
                paths = top_n_tripaths(graphs["paper"], searchpathfrom_dict['data']['name'],
                                       searchnodetotraverse_dict['data']['name'], searchpathto_dict['data']['name'], topN,
                                       strategy="naive", distance="distance_npmi", intersecting=intersecting, pretty_print=False)
                
                paths = [list(OrderedDict.fromkeys(path[0] + path[1])) for path in paths]
                
            else:
                paths = top_n_paths(
                    graphs["paper"], searchpathfrom_dict['data']['id'], searchpathto_dict['data']['id'],
                    topN, distance="distance_npmi", strategy="naive",pretty_print=False)
            elements = []
                
            
            if paths:
                elements.append(searchpathfrom_dict) 
                elements.append(searchpathto_dict)
                
            for path in paths:
                
                path_steps = list(path)
                searchpathfrom = searchpathfrom_dict["data"]["id"]
                for index, path_step in enumerate(path_steps):
                    path_step = str(path_step).replace(" ","_")
                    if path_step in elements_dict:
                        path_element = elements_dict[path_step]
                    else:
                        try:
<<<<<<< HEAD
                            result_df = linked_mention_df.loc[str(path_step).lower()]
=======
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616

                            result_df = linked_mention_df.loc[str(path_step).lower()]
                            if len(result_df) > 0:
                                node = result_df.uid
                                path_element = create_node(id=node, label=result_df.concept, definition=result_df.definition)

                        except Exception as e:
                            
                            continue
                    
                    path_element_id = path_element['data']['id']
                    elements.append(path_element)
                   
                    edge_from_id = str(searchpathfrom).lower().replace(" ","_")+"_"+str(path_element_id).lower()
                    edge_from = create_edge(edge_from_id, searchpathfrom, path_element_id)
                    elements.append(edge_from)
                    
                    searchpathfrom = path_element_id

    def node_range_condition(el, start, end):
        if node_freq_type in el["data"]:
            if el["data"][node_freq_type] >= start and\
               el["data"][node_freq_type] <= end:
                return True
        return False
    
    def edge_range_condition(el, start, end, nodes_to_remove=None):
        if edge_freq_type in el["data"]:
            inrange = False
            if el["data"][edge_freq_type] >= start and\
               el["data"][edge_freq_type] <= end:
                inrange = True
            notdangling = False
            if nodes_to_remove is None or (
                    el["data"]["source"] not in nodes_to_remove and\
                    el["data"]["target"] not in nodes_to_remove):
                notdangling = True
            return inrange and notdangling
            
        return False

    if nodefreqslider and button_id == "nodefreqslider":
        if visualization_app._min_node_weight is None or\
           visualization_app._max_node_weight is None:
            visualization_app._min_node_weight = nodefreqslider[0]
            visualization_app._min_edge_weight = nodefreqslider[1]
        elif nodefreqslider[0] != visualization_app._min_node_weight or\
             nodefreqslider[1] != visualization_app._max_node_weight:

            visualization_app._min_node_weight = nodefreqslider[0]
            visualization_app._max_node_weight = nodefreqslider[1]
        
            nodes_to_remove = [
                el["data"]["id"]
                for el in visualization_app._graphs[val]["cytoscape"]
                if "source" not in el["data"] and not node_range_condition(
                    el, nodefreqslider[0], nodefreqslider[1])
            ]

            edges_to_remove = [
                el["data"]["id"]
                for el in visualization_app._graphs[val]["cytoscape"]
                if "source" in el["data"] and (
                    not edge_range_condition(
                        el, visualization_app._min_edge_weight,
                        visualization_app._max_edge_weight, nodes_to_remove))
            ]

            elements = [
                el
                for el in visualization_app._graphs[val]["cytoscape"]
                if el["data"]["id"] not in nodes_to_remove and el["data"]["id"] not in edges_to_remove
            ]

    elif edgefreqslider and button_id == "edgefreqslider":
        if visualization_app._min_edge_weight is None or\
           visualization_app._max_edge_weight is None:
            visualization_app._min_edge_weight = edgefreqslider[0]
            visualization_app._min_edge_weight = edgefreqslider[1]
        elif edgefreqslider[0] != visualization_app._min_edge_weight or\
             edgefreqslider[1] != visualization_app._max_edge_weight:
            
            visualization_app._min_edge_weight = edgefreqslider[0]
            visualization_app._min_edge_weight = edgefreqslider[1]
    
            nodes_to_remove = [
                el["data"]["id"]
                for el in visualization_app._graphs[val]["cytoscape"]
                if "source" not in el["data"] and not node_range_condition(
                    el, visualization_app._min_node_weight, visualization_app._max_node_weight)
            ]

            edges_to_remove  = [
                el["data"]["id"]
                for el in visualization_app._graphs[val]["cytoscape"]
                if "source" in el["data"] and not edge_range_condition(
                    el, edgefreqslider[0], edgefreqslider[1], nodes_to_remove)
            ]
            elements = [
                el
                for el in visualization_app._graphs[val]["cytoscape"]
                if el["data"]["id"] not in nodes_to_remove and el["data"]["id"] not in edges_to_remove
            ]
  
    return [zoom, elements]


@visualization_app._app.callback([Output('item-card-body', 'children')],
                  [Input('cytoscape', 'tapNode'),
                   Input('cytoscape', 'tapEdge')],
                  [State('cytoscape', 'selectedNodeData'),
                   State('cytoscape', 'selectedEdgeData'),
                   State('showgraph', 'value')])
def display_tap_node(datanode, dataedge, statedatanode, statedataedge, showgraph):  
    papers = []
    res = []
    modal_button = None
    if datanode and statedatanode:
        definition = ""
        if 'definition' in str(datanode['data']):
            definition = str(datanode['data']['definition'])
<<<<<<< HEAD
        
=======

>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
        entity = str(datanode['style']['label'])
        if entity in visualization_app._entity_definitions:
            definition = visualization_app._entity_definitions[entity]

        label = str(datanode['style']['label'])
        _type = str(datanode['data']['entity_type'])

        frequency = str(len(datanode['data']['papers']))
        res.append([
            html.H5(label, className="card-title"),
            html.H6(_type, className="card-subtitle"),
            html.P(
                definition,
                className="card-text"
            )
        ])
<<<<<<< HEAD
=======

>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
        label = "'"+label+"' mentioned in " + frequency + " papers"
        modal_button = dbc.Button(label, id="open-body-scroll",color="primary")
        
        papers= datanode['data']['papers']

    elements = visualization_app._graphs[showgraph]["cytoscape"]
    elements_dict = visualization_app._graphs[showgraph]["dict"]
        
    if dataedge and statedataedge:
        label = str(dataedge['style']['label'])
        
        source_node = elements_dict[ dataedge['data']['source']]
        source_label = source_node['data']['name']
        target_node = elements_dict[ dataedge['data']['target']]
        target_label = target_node['data']['name']
        papers = set(source_node['data']['papers']).intersection(set(target_node['data']['papers']))
        frequency = str(len(papers))
        mention_label= ''' '%s' mentioned in %s papers with '%s' ''' % (source_label, frequency, target_label) 
        label = mention_label if str(dataedge['style']['label']) == "" else str(dataedge['style']['label']) 
        modal_button= dbc.Button(label, id="open-body-scroll",color="primary")

    if len(papers) > 0:
<<<<<<< HEAD
=======

>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
        papers_in_kg = visualization_app._list_papers_callback(papers)

        rows = []
        
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

            modal = html.Div(
            [
                modal_button,
                dbc.Modal([
                        dbc.ModalHeader(label),
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
                      Input('showgraph', 'value')
                  ])
def update_cytoscape_layout(layout,showgraph):
    if "style" in visualization_app._graphs[showgraph]:
        return {'name': 'preset'}

    if layout == "cose":
        return {
            'name': layout,
            'showlegend':True,
            'idealEdgeLength': 100,
            'nodeOverlap': 0,
            'refresh': 20,
            'fit': True,
            'padding': 30,
            'randomize': False,
            'componentSpacing': 100,
            'nodeRepulsion': 400000,
            'edgeElasticity': 100,
            'nestingFactor': 5,
            'gravity': 80,
            'numIter': 1000,
            'initialTemp': 200,
            'coolingFactor': 0.95,
            'minTemp': 1.0
        }
    elif layout =="cola":
        return {
            'name': layout,
            'animate': True,
            'refresh': 1,
#             'infinite': True,
            'maxSimulationTime': 8000,
            'ungrabifyWhileSimulating': False,
            'fit': True, 
            'padding': 30,
#             'nodeDimensionsIncludeLabels': False,
            'randomize': True,
            'avoidOverlap': True,
            'handleDisconnected': True,
            'convergenceThreshold': 0.001,
            'nodeSpacing': 10,
            'edgeLength': 100
        }
    elif layout == "cose-bilkent":
        return {
            'name': layout,
            "quality": 'default',
            "refresh": 30,
            "fit": True,
<<<<<<< HEAD
=======

>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
            "padding": 20,
            "randomize": True,
            "nodeSeparation": 75,
            "nodeRepulsion": 40500,
            "idealEdgeLength": 70,
            "edgeElasticity": 0.45,
            "nestingFactor": 0.1,
            "gravity": 50.25,
            "numIter": 2500,
            "tile": True,
            "tilingPaddingVertical": 50,
            "tilingPaddingHorizontal": 50,
            "gravityRangeCompound": 1.5,
            "gravityCompound": 2.0,
            "gravityRange": 23.8,
            "initialEnergyOnIncremental": 50.5
        }
    
    else:    
        return {
            'name': layout,
            'showlegend':True
        }


@visualization_app._app.callback(
                  Output('cytoscape', 'stylesheet'),
                  [Input('cytoscape', 'tapNode'),
                   Input('cytoscape', 'selectedNodeData'),
                   Input('input-follower-color', 'value'),
                   Input('dropdown-node-shape', 'value'),
                   Input('showgraph', 'value'),
                   Input('node_freq_type', 'value'),
                   Input('edge_freq_type', 'value')],
                   [State('cytoscape', 'stylesheet')])
def generate_stylesheet(node, selectedNodes, follower_color, node_shape, showgraph, node_freq_type, edge_freq_type, original_stylesheet):
    if "style" in visualization_app._graphs[showgraph]:
        return visualization_app._graphs[showgraph]["style"]
    else:
        stylesheet = CYTOSCAPE_STYLE_STYLESHEET
    focus_nodes = []
    
    if selectedNodes:
        focus_nodes = [selectedNode for selectedNode in selectedNodes]

    if node is not None:
        focus_nodes.append(node)
        
    if node_freq_type or node:
        stylesheet = [
            style
            for style in stylesheet
            if "style" in style and not (style["selector"] == 'node' and 'width' in style["style"])
        ]
        stylesheet.append({
            "selector": 'node',
            'style': {
                'shape': node_shape,
                'width':'data(' + node_freq_type + '_size)',
                'height':'data(' + node_freq_type + '_size)',
                'font-size':'data(' + node_freq_type + '_font_size)'
            }

        })
        
    if edge_freq_type:
        stylesheet = [style for style in stylesheet if not (style["selector"] == 'edge' and 'width' in style["style"])]
        stylesheet.append({
            "selector": 'edge',
            'style': {'width':'data(' + edge_freq_type + '_size)'}
        })

    for focus_node in focus_nodes:      
        node_style = [
            {
              "selector": "node:selected",
              "style": {
                "border-width": "5px",
                "border-color": "#AAD8FF",
                "border-opacity": "0.5"
              }
            }, 
            {
                "selector": 'edge',
                "style": {
                    'curve-style': 'bezier',
                    'line-color': '#D5DAE6'
                }
            },{
                    "selector": 'node[id = "{}"]'.format(focus_node['data']['id'] if "data" in focus_node else focus_node['id']),
                    "style": {
                        "border-width": "5px",
                        "border-color": "#AAD8FF",
                        "border-opacity": "0.5",
                        "text-opacity": 1,
                        'z-index': 9999
                    }
                }]
        for style in node_style:
            stylesheet.append(style)
        
        
        if "edgesData" in focus_node:
            for edge in focus_node['edgesData']:
                if edge['source'] == focus_node['data']['id'] if "data" in focus_node else focus_node['id']:
                    stylesheet.append({
                        "selector": 'node[id = "{}"]'.format(edge['target']),
                        "style": {
                            'opacity': 0.9
                        }
                    })
                    stylesheet.append({
                        "selector": 'edge[id= "{}"]'.format(edge['id']),
                        "style": {
                            "mid-target-arrow-color": follower_color['hex'],
                            "line-color": follower_color['hex'],
                            'opacity': 0.9,
                            'z-index': 5000
                        }
                    })
                if edge['target'] == focus_node['data']['id'] if "data" in focus_node else focus_node['id']:
                    stylesheet.append({
                        "selector": 'node[id = "{}"]'.format(edge['source']),
                        "style": {
                            'opacity': 0.9,
                            'z-index': 9999
                        }
                    })
                    stylesheet.append({
                        "selector": 'edge[id= "{}"]'.format(edge['id']),
                        "style": {
                            "mid-target-arrow-color": follower_color['hex'],
                            "line-color": follower_color['hex'],
                            'opacity': 1,
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
    [State("searchdropdown", "value"),
    State('cytoscape', 'elements')],
)
def update_multi_options(search_value, value,elements):
    if not search_value:
        raise PreventUpdate
    res = []
    for ele_data in elements:
        if 'name' in ele_data['data']:
            label =ele_data['data']['name']
            if (search_value in label) or (label in search_value) or ele_data['data']['id'] in (value or []) :
                res.append({"label":ele_data['data']['name'],"value":ele_data['data']['id']})
    return res


@visualization_app._app.callback(
    Output("searchpathto", "options"),
    [Input("searchpathto", "search_value")],
    [State("searchpathto", "value"),
     State('searchpathfrom', 'value'),
     State('showgraph', 'value')],
)
def searchpathto(search_value, value,_from, showgraph):
    if not search_value:
        raise PreventUpdate
    return search(search_value, value, showgraph,[_from])


@visualization_app._app.callback(
    Output("searchnodetotraverse", "options"),
    [Input("searchnodetotraverse", "search_value")],
    [State("searchnodetotraverse", "value"),
     State('searchpathfrom', 'value'),
     State('searchpathto', 'value'),
     State('showgraph', 'value')],
)
def searchpathtraverse(search_value, value,_from,to, showgraph):
    if not search_value:
        raise PreventUpdate
    return search(search_value, value, showgraph, [_from,to])


@visualization_app._app.callback(
    Output("searchpathfrom", "options"),
    [Input("searchpathfrom", "search_value")],
    [State("searchpathfrom", "value"),
     State('showgraph', 'value')],
)
def searchpathfrom(search_value, value, showgraph ):
    if not search_value:
        raise PreventUpdate
    return search(search_value, value,showgraph)


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
<<<<<<< HEAD
=======

removed = set()
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
