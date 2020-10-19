<<<<<<< HEAD
=======

>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
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
<<<<<<< HEAD
        }
=======
>>>>>>> b8df3964948c29a3f1c82edd5f2bee8ead889616
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
