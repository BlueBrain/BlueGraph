import math
import traceback
import pandas as pd

from operator import ge, gt, lt, le, eq, ne

import dash
from jupyter_dash import JupyterDash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from dash.exceptions import PreventUpdate

import plotly.express as px


import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


OPERATORS = [
    ['ge ', '>='],
    ['le ', '<='],
    ['lt ', '<'],
    ['gt ', '>'],
    ['ne ', '!='],
    ['eq ', '='],
    ['contains '],
    ['datestartswith ']
]


DROPDOWN_FILTER_LIST = [
    {"label": ">", "value": "gt"},
    {"label": ">=", "value": "ge"},
    {"label": "<", "value": "lt"},
    {"label": "<=", "value": "le"},
    {"label": "=", "value": "eq"},
    {"label": "!=", "value": "ne"}
]


SUPPORTED_JUPYTER_DASH_MODE = ["jupyterlab", "inline","external"]

DEFAULT_ENTITY_FREQUENCY = 1

# -------------- Utils ------------------

def split_filter_part(filter_part):
    for operator_type in OPERATORS:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                return name, operator_type[0].strip(), value
    return [None] * 3


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))


# -------------- Curation app ------------------

class CurationApp(object):
    
    def __init__(self):
        self._app = JupyterDash('Extracted Entities Curation App')
        self._default_terms_to_include = None
        self._table_columns = None
        self._original_table = None
        self._curated_table = None
        self._server = self._app.server
        
        # Components
        button_group = dbc.ButtonGroup(
            [
                dcc.Upload(
                        id='datatable-upload',
                        children=html.Div([
                            dbc.Button("Load a CSV File", color="primary", className="mr-1",id="load_file"),
                            dbc.Tooltip(
                                "Load extracted entities in CSV format",
                                target="load_file",
                                placement="bottom",
                            )
                        ]),
                    className="mr-1"
                )
            ],
            className="mr-1"
        )

        buttons = dbc.FormGroup([button_group])

        dropdown = dbc.FormGroup(
            [
                dbc.InputGroupAddon(
                    dbc.Button("Entity Frequency", color="primary", id="entity_frequency"),
                    addon_type="prepend",
                    className="mr-1"
                ),
                dbc.Tooltip(
                    "Select an operator and a frequency threshold",
                    target="entity_frequency",
                    placement="bottom",
                ),
                dcc.Dropdown(
                    id='dropdown-freq-filter',
                    value="ge",
                    clearable=False,
                    options=DROPDOWN_FILTER_LIST,

                    className="mr-1"
                ),
                daq.NumericInput(
                    id="entityfreqslider",
                    min=DEFAULT_ENTITY_FREQUENCY,  
                    max=1000,
                    value=DEFAULT_ENTITY_FREQUENCY,
                    className="mr-1"
                )
            ],
            className="mr-1"
        )


        reset = dbc.FormGroup(
            [
                dbc.Button("Reset", color="primary", className="mr-1",id='table-reset'),
                dbc.Tooltip(
                    "Reset table and graph to original extracted entities and default filters",
                    target="table-reset",
                    placement="bottom",
                )
            ]
        )

        link_ontology = dbc.FormGroup(
            [
                dbc.Button("Link to NCIT ontology", color="primary", className="mr-1",id='link_ontology'),
                dbc.Tooltip(
                    "Click to apply ontology linking",
                    target="link_ontology",
                    placement="bottom",
                )
            ]
        )

        top_n_frequent_entity = dbc.FormGroup(
            [
                dbc.InputGroupAddon(
                    children="Generate Graphs from top n frequent entities",
                    id="top_n_frequent_entity",
                    addon_type="prepend",
                    className="mr-1"
                ),
                dbc.Tooltip(
                    "The Co-mention graphs will be generate from the top n most frequent entities",
                    target="top_n_frequent_entity",
                    placement="bottom",
                ),
                daq.NumericInput(
                    id="topnentityslider",
                    max=500,
                    value=500,
                    className="mr-1"
                )
            ],
            className="mr-1"
        )

        form_table = dbc.Form(
            [
                buttons,
                dropdown,
                reset,
                link_ontology,
                top_n_frequent_entity
            ],
            inline=True)

        self.dropdown = dcc.Dropdown(
                id="term_filters",
                multi=True,
                value=self._default_terms_to_include,
                style={
                     "width":"80%"
                },
                placeholder="Search for entities to keep"
            )
        term_filters = dbc.InputGroup(
            [
                dbc.InputGroupAddon("Keep", addon_type="prepend"),
                self.dropdown
            ],
            className="mb-1"
        )
        
        self._app.layout = html.Div([
            dbc.Row(dbc.Col(form_table)),
            dbc.Row(dbc.Col(term_filters)),
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id='datatable-upload-container',
                        data=pd.DataFrame().to_dict('records'),
                        columns=self._table_columns,
                        style_cell={
                            'whiteSpace': 'normal'
                        },
                        style_data_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }],
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        css=[{
                            'selector': 'dash-fixed-content',
                            'rule': 'height: 100%;'
                        }],
                        sort_action="custom", #native
                        sort_mode="multi",
                        column_selectable="multi",
                        filter_action="custom",
                        filter_query='',
                        selected_columns=[],
                        page_action="custom", #native
                        export_format='csv',
                        export_headers='display',
                        merge_duplicate_headers=True,
                        selected_rows=[],
                        page_current=0,
                        page_size=10,
                        sort_by=[],
                    )
                )
            ),
            dbc.Row(dbc.Col(dcc.Graph(id='datatable-upload-Scatter')))
        ])


    def set_default_terms_to_include(self, terms):
        self._default_terms_to_include = terms
        self.dropdown.value=self._default_terms_to_include

    def set_table(self, table):
        self._original_table = table
        self._curated_table = table.copy()
        columns = [
            {
                "name": i,
                "id": i,
                "clearable": True,
                "selectable": True,
                "renamable": False,
                "hideable": True,
                "deletable": False
            } 
            for i in ["entity", "entity_type", "paper_frequency"]
#             for i in self._curated_table.columns
#             if i not in ["paper", "section", "paragraph", "raw_entity_types"]
        ]
        self._table_columns = columns

    def set_ontology_linking_callback(self, func):
        self._ontology_linking_callback = func
        
    def run(self, port, mode="jupyterlab"):
        if mode not in SUPPORTED_JUPYTER_DASH_MODE:
            raise Exception("Please provide one of the following mode value: "+str(SUPPORTED_JUPYTER_DASH_MODE))
        try:
            self._app.run_server(mode=mode, width="100%", port=port)
        except OSError as ose:
            print(f"Opening port number {port} failed: {str(ose)}. Trying port number {port+1} ...")
            try:
                self._app.run_server(mode=mode, width="100%", port=port+1)
            except Exception as e:
                print(e)

    def get_curated_table(self):
        table = self._curated_table.copy()
        table = table.set_index("entity")
        return table


curation_app = CurationApp()

# Callbacks
@curation_app._app.callback(
    Output('datatable-upload-container', 'style_data_conditional'),
    [Input('datatable-upload-container', 'selected_columns')]
)
def update_styles(selected_columns):
    return [{
        'if': {'column_id': i},
        'background_color': '#D2F3FF'
    } for i in selected_columns]


@curation_app._app.callback(
    Output("term_filters", "options"),
    [Input("term_filters", "search_value"), Input('link_ontology', 'n_clicks')],
    [State("term_filters", "value")],
)
def update_filter(search_value, click_link_ontology, values):
    
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                        
    if not search_value and values is None:
        raise PreventUpdate
    
    res = []

    if values is not None:
        if button_id == "link_ontology":
            for value in values:
                try:
                    vals = linked_mention_df_unique.loc[str(value).lower()]
                    vals = vals.concept.lower()
                    res.append({"label": vals, "value": vals})
                except Exception as e:
                    res.append({"label": value, "value": value})
        else:
            for value in values:      
                res.append( {"label": value, "value": value})
    
    if search_value is not None:
        result_df = curation_app._original_table[
            curation_app._original_table["entity"].str.contains(str(search_value))]
        result_df = result_df["entity"].unique()
        if result_df is not None:
            for result in result_df:
                res.append( {"label": result, "value": result})
    return res
    

@curation_app._app.callback([Output('entityfreqslider', 'value'),
                        Output('dropdown-freq-filter', 'value')],
                       [Input('table-reset', 'n_clicks')],
                       [State('entityfreqslider', 'value'),
                        State('dropdown-freq-filter', 'value')])
def reset(reset, entityfreq,freqoperator):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
    if button_id == "table-reset" or button_id == "No clicks yet":
        curation_app._curated_table = curation_app._original_table.copy()
        return [DEFAULT_ENTITY_FREQUENCY, "ge"]

def get_freq(row, operator, filter_value, term_filters):
    return eval(operator)(row.paper_frequency, int(filter_value)) or str(row['entity']).lower() in term_filters


@curation_app._app.callback([
               Output('datatable-upload-container', 'data'),
               Output('datatable-upload-container', 'columns'),
               Output('datatable-upload-container', 'editable'),
               Output('datatable-upload-container', 'row_deletable'),
               Output('datatable-upload-container', 'page_count')],
              [Input('datatable-upload-container', 'page_size'),
               Input('datatable-upload-container', 'page_current'),
               Input('datatable-upload-container','data_timestamp'),
               Input('datatable-upload', 'contents'),
               Input('entityfreqslider', 'value'),
               Input('dropdown-freq-filter', 'value'),
               Input('datatable-upload-container', 'sort_by'),
               Input('datatable-upload-container', 'filter_query'),
               Input('link_ontology', 'n_clicks')],
              [State("datatable-upload-container", "data"),
               State("datatable-upload-container", "columns"),
               State('datatable-upload', 'filename'),
               State('datatable-upload-container', 'derived_viewport_data'),
               State("term_filters", "value")
              ])
def update_output(page_size, page_current, ts, upload, entityfreq,
                  freqoperator, sort_by, filter_query, click_link_ontology, data,
                  columns, filename, derived_viewport_data, 
                  term_filters):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]       
            
        if term_filters is not None:
            term_filters = [str(term_filter_value).lower() for term_filter_value in term_filters ]
        else:
            term_filters = []

        if upload is not None:
            curation_app._curated_table = parse_contents(upload, filename).copy()            
        elif button_id == "table-reset":
            curation_app._curated_table = curation_app._original_table.copy()
        elif derived_viewport_data:
            removed = [row for row in derived_viewport_data if row not in data and str(row["entity"]).lower() not in term_filters]
            for row in removed:
                curation_app._curated_table = curation_app._curated_table[curation_app._curated_table.entity.str.lower() != str(row["entity"]).lower()]
#                 curation_app._original_table = curation_app._original_table[curation_app._original_table.entity.str.lower() != str(row["entity"]).lower()]

        if button_id == "link_ontology":
            curation_app._curated_table = curation_app._ontology_linking_callback(curation_app._curated_table)

        if (button_id == "entityfreqslider" or button_id=="dropdown-freq-filter")  and 'paper_frequency' in curation_app._curated_table:
            row_filtered = []  
            curation_app._curated_table = curation_app._original_table[
                curation_app._original_table.apply(lambda row: get_freq(row, freqoperator, entityfreq, term_filters), axis=1)]
    
        result = curation_app._curated_table
        
        # Filter by properties
        dff = result
        if filter_query:
            filtering_expressions = filter_query.split(' && ')
            for filter_part in filtering_expressions:
                col_name, operator, filter_value = split_filter_part(filter_part)

                if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                    
                    dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
                elif operator == 'contains':
                    dff = dff.loc[dff[col_name].str.contains(filter_value)]
                elif operator == 'datestartswith':
                    dff = dff.loc[dff[col_name].str.startswith(filter_value)]
            
        # Sorting by properties
        if sort_by and len(sort_by):
            result_sorted = dff.sort_values(
                [col['column_id'] for col in sort_by],
                ascending=[
                    col['direction'] == 'asc'
                    for col in sort_by
                ],
                inplace=False
            )
        else:
            result_sorted = dff
            
        result_paginated = result_sorted.iloc[
            page_current * page_size:(page_current + 1)*page_size
        ]
                
        page_count = len(result_sorted) // page_size
        
        return result_paginated.to_dict('records'), curation_app._table_columns, True, True, page_count
    except Exception as e:
        traceback.print_exc()

@curation_app._app.callback(
    [Output('datatable-upload-Scatter', 'figure')],
    [Input('datatable-upload-container', 'data_timestamp'),
     Input('datatable-upload-container', 'data')],)
def display_graph(dts, rows):
    df = curation_app._curated_table.copy()
    if (df.empty or len(df.columns) < 1):
        scatter = {'data': [{'x': [], 'y': []}]}
    else:
        scatter = px.scatter(
            df, x=df.entity, y=df.paper_frequency, color=df.entity_type.apply(lambda x: ",".join(x) if isinstance(x,list) else x))
    return [scatter]

@curation_app._app.callback(
    Output("top_n_frequent_entity", "children"),
    [Input("topnentityslider", "value")]
)
def set_n_most_frequent(topnentityslider_value):
    curation_app.n_most_frequent = topnentityslider_value
    return f"Generate Graphs from top {topnentityslider_value} frequent entities"
