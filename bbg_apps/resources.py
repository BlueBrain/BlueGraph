MIN_NODE_SIZE = 20
MAX_NODE_SIZE = 65

MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 32

MIN_EDGE_WIDTH = 4
MAX_EDGE_WIDTH = 12


VISUALIZATION_CONTENT_STYLE = {
    "width": "100%",
    "top": "0px",
    "left":"0px",
    "bottom": "0px",
    "position": "fixed",
}

COLORS = {
    "CHEMICAL": "#72d5de",
    "PROTEIN": "#ebaba7",
    "DISEASE": "#79ddcb",
    "CELL_TYPE": "#deb1e0",
    "PATHWAY": "#bcdfa4",
    "CELL_COMPARTMENT": "#74aff3",
    "DRUG": "#dbcd9d",
    "Biomarkers": "#7cccee",
    "Condition": "#91c79f",
    "ORGAN": "#adbce9",
    "ORGANISM": "#b3edd5",
    "GENE": "#8dc3b8",
}


CYTOSCAPE_STYLE_STYLESHEET = [
    {
        "selector": 'edge',
        'style': {
            "width": 5
        }
    }, {
        "selector": 'node',
        'style': {
            "font-size": 70,
            "text-valign": "center",
            "text-halign": "center",
            "label":"data(name)",
            "overlay-padding": "6px",
            "z-index": "10",
#             'color': 'data(color)',
        }
    }, {
        "selector" : "node[community_npmi = 0.0]",
        "css" : {"background-color" : "rgb(137,208,245)"}
    }, {
        "selector" : "node[community_npmi = 1.0]",
        "css" : {
          "background-color" : "rgb(255,102,51)"
        }
    }, {
        "selector" : "node[community_npmi = 2.0]",
        "css" : {
          "background-color" : "rgb(0,102,0)"
        }
    }, {
        "selector":'cytoscape',
        "style": {
            "width": "100%",
            "height": "100%"
        }
    }, {
        "selector": 'edge',
        "style": {
            'curve-style': 'bezier',
            'line-color': '#D5DAE6',
        }
    }, {
        "selector": 'node[entity_type = "CHEMICAL"]',
        "style": {"background-color": COLORS["CHEMICAL"]},
    }, {
        "selector": 'node[entity_type = "PROTEIN"]',
        "style": {"background-color": COLORS["PROTEIN"]},
    }, {
        "selector": 'node[entity_type = "DISEASE"]',
        "style": {"background-color": COLORS["DISEASE"]},
    }, {
        "selector": 'node[entity_type = "CELL_TYPE"]',
        "style": {"background-color": COLORS["CELL_TYPE"]},
    }, {
        "selector": 'node[entity_type = "PATHWAY"]',
        "style": {"background-color": COLORS["PATHWAY"]},
    }, {
        "selector": 'node[entity_type = "CELL_COMPARTMENT"]',
        "style": {"background-color": COLORS["CELL_COMPARTMENT"]},
    }, {
        "selector": 'node[entity_type = "DRUG"]',
        "style": {"background-color": COLORS["DRUG"]},
    }, {
        "selector": 'node[entity_type = "Biomarkers"]',
        "style": {"background-color": COLORS["Biomarkers"]},
    }, {
        "selector": 'node[entity_type = "Condition"]',
        "style": {"background-color": COLORS["Condition"]},
    }, {
        "selector": 'node[entity_type = "ORGAN"]',
        "style": {"background-color": COLORS["ORGAN"]},
    }, {
        "selector": 'node[entity_type = "ORGANISM"]',
        "style": {"background-color": COLORS["ORGANISM"]},
    }, {
        "selector": 'node[entity_type = "GENE"]',
        "style": {"background-color": COLORS["GENE"]},
    }
]


# Layout configs

COSE_BILKENT_CONFIG = {
    "quality": 'default',
    "refresh": 30,
    "fit": True,
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

COLA_CONFIG = {
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


COSE_CONFIG = {
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


CISE_CONFIG = {
    "animate": True,
    "refresh": 10, 
    "fit": True,
    "padding": 30,
    # separation amount between nodes in a cluster
    # note: increasing this amount will also increase the simulation time 
    "nodeSeparation": 12.5,
    # Inter-cluster edge length factor 
    # (2.0 means inter-cluster edges should be twice as long as intra-cluster edges)
    "idealInterClusterEdgeLengthCoefficient": 1.4,
    # Whether to pull on-circle nodes inside of the circle
    "allowNodesInsideCircle": False,
    # Max percentage of the nodes in a circle that can move inside the circle
    "maxRatioOfNodesInsideCircle": 0.1,
    # - Lower values give looser springs
    # - Higher values give tighter springs
    "springCoeff": 0.45,
    # Node repulsion (non overlapping) multiplier
    "nodeRepulsion": 4500,
    # Gravity force (constant)
    "gravity": 0.25,
    # Gravity range (constant)
    "gravityRange": 3.8,
}