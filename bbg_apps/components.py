import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

from dash_extensions import Download

import dash_cytoscape as cyto
from bbg_apps.resources import (VISUALIZATION_CONTENT_STYLE,
                                CYTOSCAPE_STYLE_STYLESHEET)

DEFAULT_LAYOUT = "cose-bilkent"

node_frequency_type = [
    ("Frequency", "paper_frequency"),
    ("Degree", "degree_frequency"),
    ("PageRank", "pagerank_frequency")
]

edge_frequency_type = [
    ("Mutual Information", "npmi"),
    ("Raw Frequency", "frequency"),
]

cluster_type = [
    ("Entity Type", "entity_type"),
    ("Community by Frequency", "community_frequency"),
    ("Community by Mutual Information", "community_npmi")
]

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


global_button_group = dbc.FormGroup([
    dbc.Col([
        dbc.Button("Reset", color="primary", className="mr-1", id='bt-reset', style={"margin": "2pt"}),
        dbc.Tooltip(
            "Reset the display to default values",
            target="bt-reset",
            placement="bottom",
        ),
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
        )], width=6, style={"padding-left": "0pt"}),
    dbc.Col([
        dbc.Label("Recompute spanning tree", html_for="recompute-spanning-tree"),
        dbc.Checklist(
            options=[{"value": 1}],
            value=[1],
            id="recompute-spanning-tree",
            switch=True,
            style={"margin-left": "5pt"}
        ),
        dbc.Tooltip(
            "If enabled, the minimum spanning tree will be recomputed on the nodes selected in the current graph view (does not apply to filtering)",
            target="recompute-spanning-tree",
            placement="bottom",
        )
    ], width=6)   
], row=True, style={"margin-left": "5pt"})
            
# recompute_tree = dbc.FormGroup([
#     dbc.Label("Recompute spanning tree", html_for="recompute-spanning-tree"),
#     dbc.Checklist(
#         options=[{"value": 1}],
#         value=[1],
#         id="recompute-spanning-tree",
#         switch=True,
#         style={"margin-left": "5pt"}
#     ),
#     dbc.Tooltip(
#         "If enabled, the minimum spanning tree will be recomputed on the nodes selected in the current graph view (does not apply to filtering)",
#         target="recompute-spanning-tree",
#         placement="bottom",
#     )
# ], style={"margin-left": "10pt"}, row=True)


edit_button_group = dbc.InputGroup([
    dbc.Button(
        "Remove selection",
        color="primary",
        className="mr-1",
        id='remove-button', 
        disabled=True,
        style={"margin": "2pt"}),
    dbc.Button(
        "Merge selected nodes", color="primary", className="mr-1", id='merge-button', style={"margin": "2pt"}, disabled=True),
    dbc.Modal(
        [
            dbc.ModalHeader("Merged label"),
            dbc.ModalBody([
                dbc.FormGroup(
                    [
                        dbc.Label("Would you like to"),
                        dbc.RadioItems(
                            options=[
                                {"label": "merge into one of the entities", "value": 1},
                                {"label": "merge as a new entity", "value": 2}
                            ],
                            value=1,
                            id="merge-options",
                        ),
                    ]
                ),
                dbc.FormGroup([
                        dbc.Input(id="merge-label-input")
                    ],
                    id="merge-input"
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Apply", id="merge-apply", color="primary", className="ml-auto"),
                dbc.Button("Close", id="merge-close",  color="default", className="ml-auto")
            ]),
        ],
        id="merge-modal",
    ),
])

edit_button_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6("Edit graph elements", className="card-title"),
            edit_button_group
        ]
    ), id="edit-button-card", style={"margin-bottom": "10pt"}
)


dropdown_items = dcc.Dropdown(
    id="showgraph",
    value="",
    options=[],
    style={"width":"100%"}
)

graph_type_dropdown = dbc.FormGroup([
    dbc.Label("Graph to display", html_for="showgraph"),
    dropdown_items
])

search = dbc.FormGroup(
    [
        dbc.Label("Search node", html_for="searchdropdown", width=3),
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
    style={"margin-bottom": "0pt"},
    row=True)

# filter_card = dbc.Card(
#     dbc.CardBody([
#         html.H6("Filters", className="card-title"),
#         frequencies_form
#     ]),
#     id="filter-card"
# )

display_message = html.P(
    "Displaying top 100 most frequent entities",
    id="display-message")

top_n_button = dbc.Button(
    "Show N most frequent",
    color="primary", className="mr-1", id='top-n-button',
    style={"margin": "5pt"})

top_n_slider = daq.NumericInput(
    id="top-n-slider",
    min=1,  
    max=1000,
    value=100,
    className="mr-1",
    disabled=False,
    style={"margin": "5pt"}
)
show_all_button = dbc.Button(
    "Show all entities",
    id="show-all-button",
    color="primary", className="mr-1",
    style={"float": "right", "margin": "5pt"})

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

view_selection_card = dbc.Card(
    dbc.CardBody([
        html.H6("View selection", className="card-title"),
        graph_type_dropdown,
        display_message,
        top_n_groups
    ]),
    id="view-selection-card"
#     style={"margin-bottom": "10pt"}
)

cluster_group = dbc.InputGroup(
    [
        dbc.Label("Group by", html_for="cluster_type"),
        dcc.Dropdown(
            id="cluster_type",
            value="entity_type",
            options=[{'label': val[0], 'value': val[1]} for val in cluster_type],
            style={"width":"100%"})
    ],
    className="mb-1"
)


cluster_filter = dcc.Dropdown(
    id="clustersearch",
    multi=True,
    options=[],
    value="All"
)

filter_by_cluster = dbc.FormGroup(
    [
        dbc.Label("Groups to display", html_for="clustersearch"),
        cluster_filter,
        dbc.Button(
            "Add all groups", color="primary", className="mr-1", id='addAllClusters',
            style={"margin-top": "10pt"})
    ], style={"margin-top": "10pt"}
)


cluster_selection_card = dbc.Card(
    dbc.CardBody([
        html.H6("Grouping", className="card-title"),
        cluster_group,
        filter_by_cluster,  
    ]),
    style={"margin-bottom": "10pt"}
)

nodes_to_keep = dbc.FormGroup(
    [
        dbc.Label("Nodes to keep", html_for="nodestokeep"),
        dcc.Dropdown(
            id="nodestokeep",
            multi=True,
            options=[],
        ),
    ],
)

form = dbc.Form([
    global_button_group,
    nodes_to_keep,
    view_selection_card,
    cluster_selection_card,
#     filter_card,
])
        
element_form = dbc.Form([
    edit_button_card,
    search,
    item_details_card
], id="element-form")


legend = dbc.FormGroup([
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

cyto = cyto.Cytoscape(
    id='cytoscape',
    elements=None,
    stylesheet=CYTOSCAPE_STYLE_STYLESHEET,
    style={"width": "100%", "height": "100%"})

layout  = html.Div([
   dcc.Store(id='memory', data={
       "removed_nodes": [],
       "removed_edges": [],
       "added_nodes": [],
       "added_edges": [],
       "merged_nodes": {},
       "hidden_elements": []
   }),
    dbc.Row([]),
    dbc.Row([
        dbc.Col([
            html.Div(style=VISUALIZATION_CONTENT_STYLE, children=[cyto]), 
            html.Div([
                dbc.Card([
                    html.H6(
                        "Legend (colored by Entity Type)", className="card-title",
                        id="legend-title"
                    ),
                    legend
                ], body=True)
            ], className="fixed-bottom", style={"width": "35%", "height": "150pt"})
        ], width=8),
        dbc.Col(html.Div(children=[
            dbc.Button(
                "Controls",
                id="collapse-button",
                color="primary",
                style={
                    "margin": "10pt",
                    "margin-left": "80%"
                }
            ),
            dbc.Collapse(dbc.Tabs(id='tabs', children=[
                dbc.Tab(
                    label='Element edit/view', label_style={"color": "#00AEF9", "border-radius":"4px"},
                    children=[dbc.Card(dbc.CardBody([element_form]))]),
                dbc.Tab(
                    label='Graph view', label_style={"color": "#00AEF9", "border-radius":"4px"},
                    children=[dbc.Card(dbc.CardBody([form]))]),
                dbc.Tab(
                    label="Filters", label_style={"color": "#00AEF9", "border-radius":"4px"},
                    children=[dbc.Card(dbc.CardBody([frequencies_form]))]),
                dbc.Tab(
                    label='Layout', label_style={"color": "#00AEF9"},
                    children=[dbc.Card(dbc.CardBody([conf_form]))]),
                dbc.Tab(
                    label='Path finder', label_style={"color": "#00AEF9"},
                    children=[dbc.Card(dbc.CardBody([form_path_finder]))])
            ]), id="collapse"),
            ]),
            width=4
        )
    ])
])