#
# Blue Brain Graph is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blue Brain Graph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Blue Brain Graph. If not, see <https://choosealicense.com/licenses/lgpl-3.0/>.

TWO_LETTER_ENTITIES = [
    "ph", "ca", "hg", "o2", "na", "mg"
]

MIN_NODE_SIZE = 10
MAX_NODE_SIZE = 40

MIN_FONT_SIZE = 6
MAX_FONT_SIZE = 24

MIN_EDGE_WIDTH = 3
MAX_EDGE_WIDTH = 10


DEFAULT_TYPES = [
    "CHEMICAL",
    "PROTEIN",
    "DISEASE",
    "CELL_TYPE",
    "PATHWAY",
    "CELL_COMPARTMENT",
    "DRUG",
    "Biomarkers",
    "Condition",
    "ORGAN",
    "ORGANISM",
    "GENE"
]


VISUALIZATION_CONTENT_STYLE = {
    "width": "100%",
    "top": "0px",
    "left":"0px",
    "bottom": "0px",
    "position": "fixed",
}

COLORS = {
    "DISEASE": "#c3c94d",
    "ORGANISM": "#9c83e8",
    "ORGAN": "#6dc960",
    "PROTEIN": "#de6dcb",
    "CHEMICAL": "#64cca3",
    "PATHWAY": "#e77158",
    "CELL_TYPE": "#4cc9d8",
    "DRUG": "#cf9749",
    "GENE": "#7fa0de",
    "CELL_COMPARTMENT": "#df7fa5",
    "Biomarkers": "#7cccee",
    "Condition": "#91c79f",
    "0": "#8cb900",
    "1": "#d97dd8",
    "2": "#00c7ff",
    "3": "#ff7045",
    "4": "#23966F",
    "5": "#deb1e0",
    "6": "#dbcd9d",
    "7": "#7cccee",
    "8": "#91c79f",
    "9": "#adbce9",
    "10": "#b3edd5",
    "11": "#8dc3b8",
#     "CHEMICAL": "#72d5de",
#     "PROTEIN": "#ebaba7",
#     "DISEASE": "#79ddcb",
#     "CELL_TYPE": "#deb1e0",
#     "PATHWAY": "#bcdfa4",
#     "CELL_COMPARTMENT": "#74aff3",
#     "DRUG": "#dbcd9d",
#     "Biomarkers": "#7cccee",
#     "Condition": "#91c79f",
#     "ORGAN": "#adbce9",
#     "ORGANISM": "#b3edd5",
#     "GENE": "#8dc3b8",
#     0: "#74aff3",
#     1: "#ebaba7",
#     2: "#bcdfa4",
#     3: "#72d5de",
#     4: "#79ddcb",
#     5: "#deb1e0",
#     6: "#dbcd9d",
#     7: "#7cccee",
#     8: "#91c79f",
#     9: "#adbce9",
#     10: "#b3edd5",
#     11: "#8dc3b8",
}


CYTOSCAPE_STYLE_STYLESHEET = [
#     {
#         "selector":'cy',
#         "style": {
#             "width": "100%",
#             "height": "100%"
#         }
#     },  
    {
        "selector": 'node',
        'style': {
            "opacity": 1,
            "text-valign": "center",
            "text-halign": "center",
            "label":"data(name)",
            "overlay-padding": "6px",
            "z-index": "10",
        }
    }, {
        "selector": "edge",
        "style": {
            'curve-style': 'bezier',
            'line-color': '#D5DAE6',

        }
    }, {
        "selector": "node",
        "style": {
            "width": 10,
            "height": 10,
        }
    }, {
        "selector": "edge",
        "style": {
            "width": 2,
        }
    }
]


# Layout configs

# COSE_BILKENT_CONFIG = {
#     "quality": 'default',
#     "refresh": 30,
#     "fit": True,
#     "padding": 20,
#     "randomize": True,
#     "nodeSeparation": 75,
#     "nodeRepulsion": 40500,
#     "idealEdgeLength": 70,
#     "edgeElasticity": 0.45,
#     "nestingFactor": 0.1,
#     "gravity": 50.25,
#     "numIter": 2500,
#     "tile": True,
#     "tilingPaddingVertical": 50,
#     "tilingPaddingHorizontal": 50,
#     "gravityRangeCompound": 1.5,
#     "gravityCompound": 2.0,
#     "gravityRange": 23.8,
#     "initialEnergyOnIncremental": 50.5
# }

COSE_BILKENT_CONFIG = {
    "name": "cose-bilkent",
    "quality": 'default',
    # Whether to include labels in node dimensions. Useful for avoiding label overlap
    "nodeDimensionsIncludeLabels": False,
    # number of ticks per frame; higher is faster but more jerky
    "refresh": 30,
    # Whether to fit the network view after when done
    "fit": True,
    # Padding on fit
    "padding": 10,
    # Whether to enable incremental mode
    "randomize": True,
    # Node repulsion (non overlapping) multiplier
    "nodeRepulsion": 4500,
    # Ideal (intra-graph) edge length
    "idealEdgeLength": 70,
    # Divisor to compute edge forces
    "edgeElasticity": 0.45,
    # Nesting factor (multiplier) to compute ideal edge length for inter-graph edges
    "nestingFactor": 0.1,
    # Gravity force (constant)
    "gravity": 50.25,
    # Maximum number of iterations to perform
    "numIter": 2500,
    # Whether to tile disconnected nodes
    "tile": True,
    # Type of layout animation. The option set is {'during', 'end', false}
    "animate": False,
    # Duration for animate:end
    "animationDuration": 500,
    # Amount of vertical space to put between degree zero nodes during tiling (can also be a function)
    "tilingPaddingVertical": 10,
    # Amount of horizontal space to put between degree zero nodes during tiling (can also be a function)
    "tilingPaddingHorizontal": 10,
    # Gravity range (constant) for compounds
    "gravityRangeCompound": 1.5,
    # Gravity force (constant) for compounds
    "gravityCompound": 2.0,
    # Gravity range (constant)
    "gravityRange": 30,
    # Initial cooling factor for incremental layout
    "initialEnergyOnIncremental": 0.5
}

COLA_CONFIG = {
    'name': 'cola',
    'animate': True,
    'refresh': 1,
    'maxSimulationTime': 8000,
    'ungrabifyWhileSimulating': False,
    'fit': True,
    'padding': 30,
    'randomize': True,
    'avoidOverlap': True,
    'handleDisconnected': True,
    'convergenceThreshold': 0.001,
    'nodeSpacing': 10,
    'edgeLength': 100
}


COSE_CONFIG = {
    'name': "cose",
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


LAYOUT_CONFIGS = {
    "preset": {
        "name": "preset"
    },
    "cose": COSE_CONFIG,
    "cose-bilkent": COSE_BILKENT_CONFIG,
    "cola": COLA_CONFIG
}

CORD19_PROP_TYPES = {
    "nodes": {
        '@type': 'category',
        'paper': 'category',
        'paper_frequency': 'numeric',
        'entity_type': 'category',
        'degree_frequency': 'numeric',
        'pagerank_frequency': 'numeric',
        'paragraph_frequency': 'numeric',
        'community_frequency': 'numeric',
        'community_npmi': 'numeric'
    },
    "edges": {
        'frequency': 'numeric',
        'ppmi': 'numeric',
        'npmi': 'numeric',
        'distance_npmi': 'numeric'
    }
}