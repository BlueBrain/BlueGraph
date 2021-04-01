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

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

from dash_extensions import Download

import dash_cytoscape as cyto_module
from cord19kg.apps.resources import (VISUALIZATION_CONTENT_STYLE,
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

cluster_types = [
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


def generate_layout(graphs, configs):
    if configs is None:
        configs = {}

    # Read-out graph view configs
    elements = configs[
        "elements"] if "elements" in configs else None
    current_graph = configs[
        "current_graph"] if "current_graph" in configs else ""
    nodes_to_keep = configs[
        "nodestokeep"] if "nodestokeep" in configs else []
    top_n_slider_value = configs[
        "top_n"] if "top_n" in configs else None
    node_freq_type = configs[
        "node_weight"] if "node_weight" in configs else "degree_frequency"
    edge_freq_type = configs[
        "edge_weight"] if "edge_weight" in configs else "npmi"
    nodefreqslider = configs[
        "nodefreqslider"] if "nodefreqslider" in configs else [0, 100000]
    edgefreqslider = configs[
        "edgefreqslider"] if "edgefreqslider" in configs else [0, 100000]
    cluster_type = configs[
        "cluster_type"] if "cluster_type" in configs else "entity_type"
    clustersearch = configs[
        "clustersearch"] if "clustersearch" in configs else []
    current_layout = configs[
        "current_layout"] if "current_layout" in configs else DEFAULT_LAYOUT

    # Read-out old path search configs
    searchpathfrom = configs[
        "searchpathfrom"] if "searchpathfrom" in configs else None
    searchpathto = configs[
        "searchpathto"] if "searchpathto" in configs else None
    searchnodetotraverse = configs[
        "searchnodetotraverse"] if "searchnodetotraverse" in configs else None
    searchpathlimit = configs[
        "searchpathlimit"] if "searchpathlimit" in configs else 10
    searchpathoverlap = configs[
        "searchpathoverlap"] if "searchpathoverlap" in configs else [1]
    nestedpaths = configs["nestedpaths"] if "nestedpaths" in configs else []
    pathdepth = configs["pathdepth"] if "pathdepth" in configs else 2

    global_button_group = dbc.FormGroup([
        dbc.Col([
            dbc.Button(
                html.Span([
                    html.I(className="fas fa-redo"), " Reset view"
                ]),
                color="secondary", className="mr-1", id='bt-reset',
                style={"margin": "2pt"}),
            dbc.Tooltip(
                "Reset the display to default values",
                target="bt-reset",
                placement="bottom",
            )
        ], width=6, style={"padding-left": "0pt"}),
        dbc.Col([
            dbc.Label(
                html.Span("Recompute spanning tree", id="recomp-label"),
                html_for="recompute-spanning-tree"),
            dbc.Checklist(
                options=[{"value": 1}],
                value=[1],
                id="recompute-spanning-tree",
                switch=True,
                style={"margin-left": "5pt"}
            ),
            dbc.Tooltip(
                "If enabled, the minimum spanning tree will be "
                "recomputed on the nodes selected in the current "
                "graph view (does not apply to filtering)",
                target="recomp-label",
                placement="bottom",
            )
        ], width=6)
    ], row=True, style={"margin-left": "5pt"})

    editing_mode_radio = dbc.FormGroup(
        [
            dbc.Label("Choose the editing mode"),
            dbc.RadioItems(
                options=[
                    {
                        "label": "Editing (edit the graph object)",
                        "value": 1
                    },
                    {
                        "label": "Masking (edit the current view)",
                        "value": 2
                    },
                ],
                value=1,
                id="edit-mode",
            ),
        ]
    )

    edit_button_group = dbc.InputGroup([
        dbc.Button(
            html.Span([
                html.I(className="fas fa-minus"),
                " Remove"
            ]),
            color="primary",
            className="mr-1",
            id='remove-button',
            disabled=True,
            style={"margin": "2pt"}),
        dbc.Button(
            html.Span([
                html.I(className="fas fa-compress-alt"),
                " Merge"
            ]), color="primary", className="mr-1", id='merge-button',
            style={"margin": "2pt"}, disabled=True),
        dbc.Button(
            html.Span([
                html.I(className="fas fa-edit"),
                " Rename"
            ]), color="primary", className="mr-1", id='rename-button',
            style={"margin": "2pt"}, disabled=True),
        dbc.Button(
            html.Span([
                html.I(className="fas fa-redo"),
                " Reset"
            ]),
            color="secondary", className="mr-1", id='reset-elements-button',
            style={"float": "right", "margin": "2pt"},
            disabled=False),
        dbc.Modal(
            [
                dbc.ModalHeader("Merged label"),
                dbc.ModalBody([
                    dbc.FormGroup(
                        [
                            dbc.Label("Would you like to"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "merge into one of the entities",
                                     "value": 1},
                                    {"label": "merge as a new entity",
                                     "value": 2}
                                ],
                                value=1,
                                id="merge-options",
                            ),
                        ]
                    ),
                    dbc.FormGroup(
                        [
                            dbc.Input(id="merge-label-input")
                        ],
                        id="merge-input"
                    )
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Apply", id="merge-apply",
                        color="primary", className="ml-auto"),
                    dbc.Button(
                        "Close", id="merge-close",
                        color="default", className="ml-auto")
                ]),
            ],
            id="merge-modal",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Rename the selected node"),
                dbc.ModalBody([
                    html.P(id="rename-error-message", style={"color": "red"}),
                    dbc.FormGroup(
                        [
                            dbc.Col(dbc.Label("New label"), width=3),
                            dbc.Col(dbc.Input(id="rename-input"), width=9)
                        ], row=True
                    ),
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Apply", id="rename-apply",
                        color="primary", className="ml-auto"),
                    dbc.Button(
                        "Close", id="rename-close",
                        color="default", className="ml-auto")
                ]),
            ],
            id="rename-modal",
        ),
    ])

    dropdown_items = dcc.Dropdown(
        id="showgraph",
        value=current_graph,
        options=[{'label': val.capitalize(), 'value': val} for val in graphs],
        style={"width":"100%"}
    )

    graph_type_dropdown = dbc.FormGroup([
        dbc.Label(
            html.Span("Graph to display", id="showgraph-label"),
            html_for="showgraph"),
        dbc.Tooltip(
            "Switch between different loaded graphs to display.",
            target="showgraph-label",
            placement="top",
        ),
        dropdown_items
    ])

    search = dbc.FormGroup(
        [
            dbc.Label(
                "Search node", html_for="searchdropdown", width=3,
                style={"text-align": "right", "padding-right": "0pt"}
            ),
            dbc.Col(dcc.Dropdown(
                id="searchdropdown",
                multi=False
            ), width=6),
            dbc.Col([
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
                    in_navbar=True
                ),
            ], width=3)
        ],
        row=True, style={"margin": "3pt"}
    )

    freq_input_group = dbc.InputGroup(
        [
            dbc.Label(
                html.Span("Node Weight", id="node_freq_type-label"),
                html_for="node_freq_type"),
            dbc.Tooltip(
                "Select a metric to use as the node weight "
                "(node sizes are proportional to the selected weight)",
                target="node_freq_type-label",
                placement="top",
            ),
            dcc.Dropdown(
                id="node_freq_type",
                value=node_freq_type,
                options=[
                    {'label': val[0], 'value': val[1]}
                    for val in node_frequency_type],
                style={"width": "100%"}),
        ],
        className="mb-1"
    )

    node_range_group = dbc.FormGroup(
        [
            dbc.Label(
                html.Span("Display Range", id="nodefreqslider-label"),
                html_for="nodefreqslider"),
            html.Div(
                [dcc.RangeSlider(
                    id="nodefreqslider_content", min=0,
                    max=100000, value=nodefreqslider)],
                id="nodefreqslider"
            ),
            dbc.Tooltip(
                "Adjust the node weight range (only the nodes having "
                "the weight in the selected range will be displayed)",
                target="nodefreqslider-label",
                placement="bottom",
            ),
        ]
    )

    edge_input_group = dbc.InputGroup(
        [
            dbc.Label(
                html.Span("Edge Weight", id="edge_freq_type-label"),
                html_for="edge_freq_type"),
            dbc.Tooltip(
                "Select a metric to use as the edge weight (edge thickness "
                "is proportional to the selected weight)",
                target="edge_freq_type-label",
                placement="top",
            ),
            dcc.Dropdown(
                id="edge_freq_type",
                value=edge_freq_type,
                options=[{
                    'label': val[0], 'value': val[1]}
                    for val in edge_frequency_type],
                style={
                    "width": "100%"
                }
            )
        ],
        className="mb-1"
    )

    edge_range_group = dbc.FormGroup(
        [
            dbc.Label(
                html.Span("Display Range", id="edgefreqslider-label"),
                html_for="edgefreqslider"),
            html.Div(
                [dcc.RangeSlider(
                    id="edgefreqslider_content",
                    min=0, max=100000, value=edgefreqslider)],
                id="edgefreqslider"),
            dbc.Tooltip(
                "Adjust the edge weight range (only the edge having the "
                "weight in the selected range will be displayed)",
                target="edgefreqslider-label",
                placement="bottom",
            ),
        ]
    )

    frequencies_form = dbc.FormGroup(
        [
            dbc.Col([freq_input_group, node_range_group], width=6),
            dbc.Col([edge_input_group, edge_range_group], width=6)
        ],
        style={"margin-bottom": "0pt"},
        row=True)

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
        value=top_n_slider_value if top_n_slider_value else 100,
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
        [
            top_n_button,
            dbc.Tooltip(
                "Display N nodes with the highest paper occurrence frequency, "
                "where N is defined by the slider on the right.",
                target="top-n-button",
                placement="top",
            ),
            top_n_slider,
            show_all_button,
            dbc.Tooltip(
                "Display all the nodes of the current graph.",
                target="show-all-button",
                placement="top",
            )],
        style={"margin-bottom": "10pt"})

    item_details_card = dbc.Card(
        dbc.CardBody(
            [
                html.H5("", className="card-title"),
                html.H6("", className="card-subtitle"),
                html.P("", className="card-text"),
                dbc.Button("", color="primary", id="see-more-card")
            ],
            id="item-card-body"
        )
    )

    view_selection_card = dbc.Card(
        dbc.CardBody([
            html.H6("View selection", className="card-title"),
            graph_type_dropdown,
            display_message,
            top_n_groups,
            frequencies_form
        ]),
        id="view-selection-card",
        style={"margin-bottom": "10pt"}
    )

    cluster_group = dbc.InputGroup(
        [
            dbc.Label(html.Span(
                "Group by", id="group-by-label"), html_for="cluster_type"),
            dbc.Tooltip(
                "Select a grouping factor for the nodes.",
                target="group-by-label",
                placement="top",
            ),
            dcc.Dropdown(
                id="cluster_type",
                value=cluster_type,
                options=[{
                    'label': val[0], 'value': val[1]}
                    for val in cluster_types],
                style={"width": "100%"})
        ],
        className="mb-1"
    )

    cluster_filter = dcc.Dropdown(
        id="clustersearch",
        multi=True,
        options=[{"label": el, "value": el} for el in clustersearch],
        value=clustersearch
    )

    filter_by_cluster = dbc.FormGroup(
        [
            dbc.Label(
                html.Span("Groups to display", id="clustersearch-label"),
                html_for="clustersearch"),
            dbc.Tooltip(
                "Only the nodes beloning to the groups selected in the field "
                "below will be displayed. You click or start typing to add new "
                "groups to display, or click on the cross icon to remove a group.",
                target="clustersearch-label",
                placement="top",
            ),
            cluster_filter,
            dbc.Button(
                "Add all groups", color="primary", className="mr-1",
                id='addAllClusters',
                style={"margin-top": "10pt"}),
            dbc.Tooltip(
                "Add all available node groups to display.",
                target="addAllClusters",
                placement="bottom",
            )
        ], style={"margin-top": "10pt"}
    )

    cluster_layout_button = dbc.InputGroup([
        dbc.Checklist(
            options=[{"value": 1, "disabled": False}],
            value=[],
            id="groupedLayout",
            switch=True,
        ),
        dbc.Label("Grouped Layout", html_for="groupedLayout"),
    ])

    cluster_selection_card = dbc.Card(
        dbc.CardBody([
            html.H6("Grouping", className="card-title"),
            cluster_group,
            filter_by_cluster,
            cluster_layout_button
        ]),
        style={"margin-bottom": "10pt"}
    )

    nodes_to_keep = dbc.FormGroup(
        [
            dbc.Label(
                html.Span("Nodes to keep", id="nodestokeep-label"),
                html_for="nodestokeep"),
            dcc.Dropdown(
                id="nodestokeep",
                multi=True,
                options=[{"label": n, "value": n} for n in nodes_to_keep],
                value=nodes_to_keep,
                placeholder="Nodes to fix in the view..."
            ),
            dbc.Tooltip(
                "The selected nodes will be fixed in the graph view "
                "and will not be filtered by 'top N' or group filters "
                "(start typing to obtain the nodes to select from).",
                target="nodestokeep-label",
                placement="top",
            )
        ],
    )

    form = dbc.Form([
        global_button_group,
        nodes_to_keep,
        view_selection_card,
        cluster_selection_card,
    ])

    legend = dbc.FormGroup([
        html.Div(id="cluster_board", children=[])
    ])

    # ------ Path search form --------

    path_from = dbc.FormGroup([
        dbc.Label(
            html.Span("From", id="searchpathfrom-label"),
            html_for="searchpathfrom", width=3
        ),
        dbc.Tooltip(
            "Select a node to use as the source in the path search",
            target="searchpathfrom-label",
            placement="top",
        ),
        dbc.Col(dcc.Dropdown(
            id="searchpathfrom",
            value=searchpathfrom,
            options=(
                [{"label": searchpathfrom, "value": searchpathfrom}]
                if searchpathfrom else [])
        ), width=9)],
        row=True)

    path_to = dbc.FormGroup([
        dbc.Label(
            html.Span("To", id="searchpathto-label"),
            html_for="searchpathto", width=3),
        dbc.Tooltip(
            "Select a node to use as the target in the path search",
            target="searchpathto-label",
            placement="top",
        ),
        dbc.Col(dcc.Dropdown(
            id="searchpathto",
            options=(
                [{"label": searchpathto, "value": searchpathto}]
                if searchpathto else []
            ),
            value=searchpathto
        ), width=9)],
        row=True)

    neighbor_control_group = html.Div([
        dbc.FormGroup([
            dbc.Col(
                dbc.Button(
                    "Find top N neighbors", color="primary", className="mr-1",
                    id='bt-neighbors', style={"float": "right"}),
                width=10
            ),
            dbc.Tooltip(
                "Search for the neighbors with the highest mutual information "
                "score.",
                target="bt-neighbors",
                placement="top",
            ),
            dbc.Col(
                daq.NumericInput(
                    id="neighborlimit",
                    min=1,
                    max=100,
                    value=10,
                    className="mr-1"
                ), width=2),

        ], row=True)
    ])

    neighbor_view_card = dbc.Card(
        dbc.CardBody(
            [], id="neighbors-card-body"
        ),
        style={"overflow-y": "scroll", "height": "150pt"}
    )

    top_n_paths = dbc.FormGroup([
        dbc.Label(
            html.Span("Top N", id="searchpathlimit-label"),
            html_for="searchpathlimit", width=3
        ),
        dbc.Tooltip(
            "Set a number of best paths to search for (the best paths are the "
            "ones that maximize the mutual information)",
            target="searchpathlimit-label",
            placement="top",
        ),
        dbc.Col(
            daq.NumericInput(
                id="searchpathlimit",
                min=1,
                max=50,
                value=searchpathlimit,
                className="mr-1"
            ), width=9)], row=True)

    path_condition = dbc.FormGroup([
        dbc.Label("Traversal conditions"),
        dbc.FormGroup(
            [
                dbc.Col([
                    dbc.Label(
                        html.Span(
                            "Entity to traverse",
                            id="searchnodetotraverse-label"),
                        html_for="searchnodetotraverse"),
                    dbc.Tooltip(
                        "Select an entity to traverse in the path search (if "
                        "selected, the search is performed in two steps: "
                        "from the source to the selected entity, and from "
                        "the selected entity to the target)",
                        target="searchnodetotraverse-label",
                        placement="top",
                    ),
                    dcc.Dropdown(
                        id="searchnodetotraverse",
                        value=searchnodetotraverse,
                        options=(
                            [{
                                "label": searchnodetotraverse,
                                "value": searchnodetotraverse
                            }]
                            if searchnodetotraverse else []))
                ], width=6),
                dbc.Col([
                    dbc.Label(
                        html.Span(
                            "Allow Overlap",
                            id="searchpathoverlap-label"),
                        html_for="searchpathoverlap"),
                    dbc.Tooltip(
                        "If the overlap is allowed, then the the paths "
                        "from the source to the intermediate entity can go "
                        "through the same entities as the paths from the "
                        "intermediary to the target. Otherwise the paths "
                        "should go through distinct entities",
                        target="searchpathoverlap-label",
                        placement="top",
                    ),
                    dbc.Checklist(
                        options=[{"value": 1}],
                        value=searchpathoverlap,
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
                    dbc.Label(
                        html.Span("Nested", id="nestedpaths-label"),
                        html_for="nestedpaths"),
                    dbc.Tooltip(
                        "If enabled, the nested paths are found, i.e. for "
                        "every edge found in a path we search for other N "
                        "best paths from the source to the target of this "
                        "edge for <depth> times",
                        target="nestedpaths-label",
                        placement="top",
                    ),
                    dbc.Checklist(
                        options=[{"value": 1}],
                        value=nestedpaths,
                        id="nestedpaths",
                        switch=True,
                    )
                ], width=6
            ),
            dbc.Col([
                dbc.Label(
                    html.Span(
                        "Depth", id="pathdepth-label"),
                    html_for="pathdepth"),
                dbc.Tooltip(
                    "Select the depth of nesting indicating for how many "
                    "iterations we expand the encountered edges into the "
                    "best paths.",
                    target="pathdepth-label",
                    placement="top",
                ),
                daq.NumericInput(
                    id="pathdepth",
                    min=1,
                    max=4,
                    value=pathdepth,
                    disabled=True,
                    className="mr-1"
                )
            ], width=6)
        ], row=True)
    ])

    search_path = dbc.InputGroup(
        [
            html.P(
                "", id="noPathMessage",
                style={"color": "red", "margin-right": "10pt"}),
            dbc.Button(
                html.Span([
                    html.I(className="fas fa-route"),
                    " Find Paths"
                ]), color="primary",
                className="mr-1", id='bt-path', style={"float": "right"}),
            dbc.Tooltip(
                "Find paths between selected entities",
                target="bt-path",
                placement="bottom",
            )
        ], style={"float": "right"}
    )

    expand_edge = dbc.FormGroup([
        dbc.Label("Edge expansion"),
        dbc.FormGroup([
            dbc.Col(
                dbc.Button(
                    "Expand edge", color="primary",
                    className="mr-1", id='bt-expand-edge',
                    disabled=True), width=6),
            dbc.Col(dbc.Label(
                "N best paths", html_for="expand-edge-n",
                style={"margin-top": "5pt"}), width=3),
            dbc.Col(
                daq.NumericInput(
                    id="expand-edge-n",
                    min=1,
                    max=20,
                    value=5,
                    className="mr-1"
                ), width=3),
        ], row=True)
    ])

    form_path_finder = dbc.Form([
        path_from,
        path_to,
        top_n_paths,
        html.Hr(),
        path_condition,
        html.Hr(),
        nested_path,
        html.Hr(),
        expand_edge,
        html.Hr(),
        search_path
    ])

    graph_layout = dbc.FormGroup(
        [
            dbc.Label("Layout", html_for="searchdropdown", width=3),
            dbc.Col(dcc.Dropdown(
                id='dropdown-layout',
                options=[
                    {
                        'label': "{}{}".format(
                            val.capitalize(),
                            " ({})".format(graph_layout_options[val])
                            if graph_layout_options[val] else ""
                        ),
                        'value': val
                    } for val in graph_layout_options.keys()
                ],
                value=current_layout,
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
                options=([
                    {'label': val.capitalize(), 'value': val}
                    for val in node_shape_option_list
                ])
            ), width=9)

        ],
        row=True
    )

    link_color_picker = dbc.FormGroup(
        [
            dbc.Col(daq.ColorPicker(
                id='input-follower-color',
                value=dict(rgb=dict(r=190, g=36, b=37, a=0)),
                label="Highlight Color"
            ))
        ],
        row=True
    )

    conf_form = dbc.Form([graph_layout, node_shape, link_color_picker])

    # ---- Create a layout from components ----------------

    cyto = cyto_module.Cytoscape(
        id='cytoscape',
        elements=elements,
        stylesheet=CYTOSCAPE_STYLE_STYLESHEET,
        style={"width": "100%", "height": "100%"},
    )

    layout = html.Div([
        dcc.Store(id='memory', data={
            "removed_nodes": [],
            "removed_edges": [],
            "added_nodes": [],
            "added_edges": [],
            "filtered_elements": [],
            "removed_elements": {},
            "renamed_elements": {},
            "merging_backup": {
                "added_elements": [],
                "removed_elements": {}
            },
            "paper_backup": {}
        }),
        dbc.Row([]),
        dbc.Row([
            dbc.Col([
                html.Div(
                    style=VISUALIZATION_CONTENT_STYLE,
                    children=[cyto], id="cyto-container"),
                html.Div(
                    [
                        dcc.Loading(
                            id="loading",
                            children=[html.Div(id="loading-output")],
                            type="default"),
                    ],
                    id="loader-container",
                    className="fixed-top",
                    style={"width": "60pt", "height": "60pt", "margin": "20pt"}
                ),
                html.Div(
                    [
                        search,
                    ],
                    id="search-container",
                    className="fixed-top",
                    style={"width": "30%", "margin-left": "80pt"}
                ),
                html.Div([
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-binoculars"), " Legend"
                        ]), id="toggle-legend", color="primary",
                        className="mr-1"
                    ),
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-search-plus"), " Details"
                        ]), id="toggle-details", color="primary",
                        className="mr-1"
                    ),
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-pen"), " Editing"
                        ]), id="toggle-edit", color="primary", className="mr-1"
                    ),
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-star-of-life"),
                            " Neighbors"
                        ]), id="toggle-neighbors", color="primary",
                        className="mr-1"
                    ),
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-angle-down"), ""
                        ]), id="toggle-hide", color="light", className="mr-1"
                    ),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6(
                                    "Legend (colored by Entity Type)",
                                    className="card-title",
                                    style={"margin-bottom": "0pt"},
                                    id="legend-title"
                                )
                            ], style={"margin-botton": "0pt"}),
                            dbc.CardBody([legend]),
                        ], style={"height": "100%"}),
                        style={"height": "150pt"},
                        id="collapse-legend"
                    ),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6(
                                    "Details",
                                    className="card-title",
                                    style={"margin-bottom": "0pt"}
                                )
                            ], style={"margin-botton": "0pt"}),
                            dbc.CardBody(
                                [
                                    item_details_card
                                ], style={"overflow-y": "scroll"})
                        ], style={"height": "100%"}),
                        style={"height": "250pt"},
                        id="collapse-details"
                    ),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6(
                                    "Edit graph",
                                    className="card-title",
                                    style={"margin-bottom": "0pt"}
                                )
                            ], style={"margin-botton": "0pt"}),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([editing_mode_radio], width=5),
                                    dbc.Col([edit_button_group], width=7)
                                ])
                            ])
                        ], style={"height": "100%"}),
                        style={"height": "150pt"},
                        id="collapse-edit"
                    ), dbc.Collapse(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H6(
                                    "Neighbors view",
                                    className="card-title",
                                    style={"margin-bottom": "0pt"}
                                )
                            ], style={"margin-botton": "0pt"}),
                            dbc.CardBody([
                                neighbor_control_group,
                                neighbor_view_card
                            ])
                        ], style={"height": "100%"}),
                        style={"height": "250pt"},
                        id="collapse-neighbors"
                    ),
                ], className="fixed-bottom", style={
                    "width": "55%",
                })
            ], width=8),
            dbc.Col(html.Div(children=[
                dbc.Button(
                    html.Span([
                        html.I(className="fas fa-cog"), " Controls"
                    ]),
                    id="collapse-button",
                    color="primary",
                    style={
                        "margin": "10pt",
                        "margin-left": "60%"
                    }
                ),
                dbc.Collapse(dbc.Tabs(id='tabs', children=[
                    dbc.Tab(
                        id="graph-view-tab",
                        label='Graph view',
                        label_style={
                            "color": "#00AEF9", "border-radius": "4px",
                            "background-color": "white"
                        },
                        children=[dbc.Card(dbc.CardBody([form]))]),
                    dbc.Tab(
                        id="layout-tab",
                        label='Layout', label_style={
                            "color": "#00AEF9", "border-radius": "4px",
                            "background-color": "white"
                        },
                        children=[dbc.Card(dbc.CardBody([conf_form]))]),
                    dbc.Tab(
                        id="path-finder-tab",
                        label='Path finder', label_style={
                            "color": "#00AEF9", "border-radius": "4px",
                            "background-color": "white"
                        },
                        children=[dbc.Card(dbc.CardBody([form_path_finder]))])
                ]), id="collapse"),
            ]), width=4)
        ])
    ], style={"overflow-x": "hidden"})
    return cyto, layout, dropdown_items, cluster_filter
