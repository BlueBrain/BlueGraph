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

import io
import os
import base64
import traceback
import pandas as pd
import numpy as np

from operator import (lt, le, eq, ne, ge, gt)

import dash
from jupyter_dash import JupyterDash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from dash.exceptions import PreventUpdate

import plotly.express as px

import dash_daq as daq

import cord19kg
from cord19kg.utils import assign_raw_type
from cord19kg.apps.app_utils import save_run
from cord19kg.apps.resources import TWO_LETTER_ENTITIES


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


SUPPORTED_JUPYTER_DASH_MODE = ["jupyterlab", "inline", "external"]

DEFAULT_ENTITY_FREQUENCY = 1


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
    """JupyterDash-based interactive entity occurrence curation app."""

    def __init__(self):
        self._app = JupyterDash(
            'Extracted Entities Curation App',
            assets_folder=os.path.join(cord19kg.__path__[0], "apps/assets"))
        self._terms_to_include = None
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
                        dbc.Button(
                            "Load a CSV File",
                            color="primary", className="mr-1",
                            id="load_file"),
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
                    dbc.Button(
                        "Entity Frequency", color="primary",
                        id="entity_frequency"),
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
                dbc.Button(
                    "Reset", color="primary",
                    className="mr-1", id='table-reset'),
                dbc.Tooltip(
                    "Reset table and graph to original extracted "
                    "entities and default filters",
                    target="table-reset",
                    placement="bottom",
                )
            ]
        )

        link_ontology = dbc.FormGroup(
            [
                dbc.Button(
                    "Link to NCIT ontology", color="primary",
                    className="mr-1", id='link_ontology'),
                dbc.Tooltip(
                    "Click to apply ontology linking",
                    target="link_ontology",
                    placement="bottom",
                )
            ]
        )

        remove_short_entities = dbc.FormGroup(
            [
                dbc.Button(
                    "Remove 2-letter entities", color="primary",
                    className="mr-1", id='remove_2_letter'),
                dbc.Tooltip(
                    "Click to remove entities that have less than two letters "
                    "(entities 'pH', 'Ca', 'Hg', 'O2', 'Na', 'Mg' are preserved)",
                    target="remove_2_letter",
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
                    "The Co-mention graphs will be generate from the top n "
                    "most frequent entities",
                    target="top_n_frequent_entity",
                    placement="bottom",
                ),
                daq.NumericInput(
                    id="topnentityslider",
                    max=2000,
                    value=500,
                    className="mr-1"
                )
            ],
            className="mr-1"
        )

        form_table = dbc.Form(
            [
                link_ontology,
                remove_short_entities,
                dropdown,
                buttons,
                top_n_frequent_entity,
                reset,
            ],
            inline=True)

        self.dropdown = dcc.Dropdown(
            id="term_filters",
            multi=True,
            value=self._terms_to_include,
            style={
                "width": "80%"
            },
            placeholder="Search for entities to keep"
        )
        term_filters = dbc.InputGroup(
            [
                dbc.InputGroupAddon("Keep", addon_type="prepend"),
                self.dropdown
            ],
            className="mb-1",
            style={
                "margin-top": "10pt",
                "margin-bottom": "10pt !important"
            }
        )

        self._app.layout = html.Div([
            dbc.Row(dbc.Col(form_table)),
            dbc.Row(dbc.Col(term_filters)),
            dbc.Row(
                dbc.Col(dcc.Loading(
                    id="loading", type="default",
                    children=[
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
                            style_table={
                                'padding-left': '11pt',
                                'padding-right': '20pt',
                                'padding-top': '5pt',
                            },
                            css=[{
                                'selector': 'dash-fixed-content',
                                'rule': 'height: 100%;'
                            }],
                            sort_action="custom",  # native
                            sort_mode="multi",
                            column_selectable="multi",
                            filter_action="custom",
                            filter_query='',
                            selected_columns=[],
                            page_action="custom",  # native
                            export_format='csv',
                            export_headers='display',
                            merge_duplicate_headers=True,
                            selected_rows=[],
                            page_current=0,
                            page_size=10,
                            sort_by=[],
                            editable=True
                        )
                    ]
                ))
            ),
            dbc.Row(dbc.Col(dcc.Graph(id='datatable-upload-Scatter')))
        ])

    def set_default_terms_to_include(self, terms):
        """Set default terms to be fixed in the table.

        The application allows to set fixed terms that
        are included in the table at all times (even when
        don't satisfy the conditions of current filters).

        Parameters
        ----------
        terms : list of str
            List of terms to fix in the data table
        """
        self._terms_to_include = terms
        self.dropdown.value = self._terms_to_include

    def set_table(self, table):
        self._linked = False
        self._original_table = table
        self._original_linked_table = None
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
        ]
        self._table_columns = columns

    def set_ontology_linking_callback(self, func):
        """Set the ontology linking callback.

        This function will be called when the `Link ontology`
        button is clicked. The curation app will pass the
        current curation table object to this function.
        """
        self._ontology_linking_callback = func

    def run(self, port, mode="jupyterlab", debug=False,
            inline_exceptions=False):
        """Run the curation app.

        Parameters
        ----------
        port : int
            Port number to launch the app (`localhost:<port>`).
        mode : str, optional
            Mode in which the app should be launched. Possible values:
            `inline` inside the current Jupyter notebook, `external`
            given an external link that can be opened in a browser,
            `jupyterlab` as a new tab of JupyterLab.
        debug : bool, optional
            Flag indicating whether the app is launched in the debug mode
            (traceback will be printed).
        inline_exceptions : bool, optional
            Flag indicating whether app exceptions should be printed in
            the current active notebook cell.
        """
        save_run(
            self, port, mode=mode, debug=debug,
            inline_exceptions=inline_exceptions)

    def get_curated_table(self):
        """Get the current curated data table.

        Returns
        -------
        table : pd.DataFrame
            Current curation data table. The table is indexed by unique
            entities and contains the following columns:
            `paper`/`section`/`paragraph` sets of papers/sections/paragraphs
            where the entity occurs, `aggregated_entities` set of unique raw
            entities associated with the entity (by ontology linking, if
            performed), `uid` ontology concept id,  `definition` ontology
            concept definition, `paper_frequency` number of unique papers
            where occurs, `entity_type` type of the entity computed either
            from the raw NER types or the ontology concept type.
        """
        table = self._curated_table.copy()
        table = table.set_index("entity")

        table["paper"] = table["paper"].apply(set)
        table["paragraph"] = table["paragraph"].apply(set)
        table["section"] = table["section"].apply(set)
        # Check if the table contains multiple entity types
        # (this means that the data was not linked and that
        # the types need to be resolved here).
        multi_type = table.entity_type.apply(
            lambda x: len(x.split(",")) > 1)

        if multi_type.any():
            table["entity_type"] = table["entity_type"].apply(
                lambda x: assign_raw_type(x.split(",")))
        if "aggregated_entities" not in table.columns:
            table["aggregated_entities"] = np.nan
        if "uid" not in table.columns:
            table["uid"] = np.nan
        if "definition" not in table.columns:
            table["definition"] = np.nan

        return table[[
            "paper", "section", "paragraph",
            "aggregated_entities", "uid", "definition",
            "paper_frequency", "entity_type"]]

    def get_terms_to_include(self):
        return self._terms_to_include


curation_app = CurationApp()


# ------------------------ Callbacks --------------------
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
    [
        Input("term_filters", "search_value"),
        Input('link_ontology', 'n_clicks')
    ],
    [
        State("term_filters", "value")
    ],
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
                    # TODO: fixe here
                    vals = curation_app._original_linked_table.loc[
                        str(value).lower()]
                    vals = vals.concept.lower()
                    res.append({"label": vals, "value": vals})
                except Exception:
                    res.append({"label": value, "value": value})
        else:
            for value in values:
                res.append({"label": value, "value": value})

    if search_value is not None:
        if curation_app._original_linked_table is None:
            result_df = curation_app._original_table[
                curation_app._original_table["entity"].str.contains(
                    str(search_value))]
        else:
            result_df = curation_app._original_linked_table[
                curation_app._original_linked_table["entity"].str.contains(
                    str(search_value))]
        result_df = result_df["entity"].unique()
        if result_df is not None:
            for result in result_df:
                res.append({"label": result, "value": result})
    return res


@curation_app._app.callback(
    [
        Output('entityfreqslider', 'value'),
        Output('dropdown-freq-filter', 'value')
    ],
    [
        Input('table-reset', 'n_clicks'),
        Input('link_ontology', 'n_clicks'),
    ],
    [
        State('entityfreqslider', 'value'),
        State('dropdown-freq-filter', 'value')
    ])
def reset(reset, link, entityfreq, freqoperator):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "table-reset" or button_id == "No clicks yet":
        curation_app._curated_table = curation_app._original_table.copy()
        curation_app._original_linked_table = None
        curation_app._linked = False
    elif button_id == "link_ontology":
        curation_app._linked = True

    return [DEFAULT_ENTITY_FREQUENCY, "ge"]


def get_freq(row, operator, filter_value, term_filters):
    return eval(operator)(
        row.paper_frequency,
        int(filter_value)) or str(row['entity']).lower() in term_filters


@curation_app._app.callback(
    [
        Output('datatable-upload-container', 'data'),
        Output('datatable-upload-container', 'columns'),
        Output('datatable-upload-container', 'editable'),
        Output('datatable-upload-container', 'row_deletable'),
        Output('datatable-upload-container', 'page_count')
    ],
    [
        Input('datatable-upload-container', 'page_size'),
        Input('datatable-upload-container', 'page_current'),
        Input('datatable-upload-container','data_timestamp'),
        Input('datatable-upload', 'contents'),
        Input('entityfreqslider', 'value'),
        Input('dropdown-freq-filter', 'value'),
        Input('datatable-upload-container', 'sort_by'),
        Input('datatable-upload-container', 'filter_query'),
        Input("term_filters", "value"),
        Input("remove_2_letter", "n_clicks")
    ],
    [
        State("datatable-upload-container", "data"),
        State("datatable-upload-container", "columns"),
        State('datatable-upload', 'filename'),
        State('datatable-upload-container', 'derived_viewport_data'),
    ])
def update_output(page_size, page_current, ts, upload, entityfreq,
                  freqoperator, sort_by, filter_query, term_filters,
                  remove_2_letter, data, columns, filename,
                  derived_viewport_data):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if term_filters is not None:
            term_filters = [
                str(term_filter_value).lower()
                for term_filter_value in term_filters
            ]
            if button_id == "term_filters":

                terms_to_add = [
                    t for t in term_filters
                    if t not in curation_app._terms_to_include and
                    t not in curation_app._curated_table.entity
                ]

                terms_to_remove = [
                    t for t in curation_app._terms_to_include
                    if t not in term_filters
                ]

                curation_app._terms_to_include = term_filters

                if curation_app._original_linked_table is None:
                    new_terms = curation_app._original_table[
                        curation_app._original_table.entity.isin(terms_to_add)]
                else:
                    new_terms = curation_app._original_linked_table[
                        curation_app._original_linked_table.entity.isin(
                            terms_to_add)]
                curation_app._curated_table = pd.concat([
                    curation_app._curated_table[
                        ~curation_app._curated_table.entity.isin(
                            terms_to_remove)],
                    new_terms
                ])

        else:
            term_filters = []

        if upload is not None:
            curation_app._original_linked_table = None
            curation_app._curated_table = parse_contents(
                upload, filename).copy()
        elif button_id == "table-reset":
            curation_app._original_linked_table = None
            curation_app._curated_table = curation_app._original_table.copy()
        elif derived_viewport_data:
            named_data_rows = {
                row["entity"]: row for row in data
            }
            # Removed
            removed = []
            renamed = {}
            retyped = {}
            for row in derived_viewport_data:
                if row["entity"] not in named_data_rows.keys() and str(
                        row["entity"]).lower() not in term_filters:
                    # Was it renamed? find a record with the same
                    # aggregated_entities
                    found = False
                    for e, data_row in named_data_rows.items():
                        if data_row["aggregated_entities"] ==\
                           row["aggregated_entities"]:
                            renamed[row["entity"]] = e
                            if row["entity_type"] != data_row["entity_type"]:
                                retyped[e] = data_row["entity_type"]
                            found = True
                            break
                    if not found:
                        removed.append(row["entity"])
                else:
                    if row["entity_type"] !=\
                       named_data_rows[row["entity"]]["entity_type"]:
                        retyped[row["entity"]] = named_data_rows[
                            row["entity"]]["entity_type"]
            # Apply removals
            for el in removed:
                curation_app._curated_table = curation_app._curated_table[
                    curation_app._curated_table.entity.str.lower() !=
                    str(el).lower()
                ]
            # Apply relabeling
            for k, v in renamed.items():
                curation_app._curated_table.loc[curation_app._curated_table[
                    "entity"] == k, "entity"] = v

            # Apply retyping
            for k, v in retyped.items():
                curation_app._curated_table.loc[curation_app._curated_table[
                    "entity"] == k, "entity_type"] = v

        if (button_id == "entityfreqslider" or button_id == "dropdown-freq-filter") and\
           'paper_frequency' in curation_app._curated_table:
            if curation_app._original_linked_table is None:
                if curation_app._linked:
                    curation_app._original_linked_table =\
                        curation_app._ontology_linking_callback(
                            curation_app._original_table)
                    curation_app._curated_table =\
                        curation_app._original_linked_table.copy()
                else:
                    curation_app._curated_table = curation_app._original_table[
                        curation_app._original_table.apply(
                            lambda row: get_freq(
                                row, freqoperator, entityfreq, term_filters),
                            axis=1)]
            else:
                curation_app._curated_table = curation_app._original_linked_table[
                    curation_app._original_linked_table.apply(
                        lambda row: get_freq(
                            row, freqoperator, entityfreq, term_filters),
                        axis=1)]

        if button_id == "remove_2_letter":
            if curation_app._original_linked_table is None:
                curation_app._original_table = curation_app._original_table[
                    curation_app._original_table.entity.apply(
                        lambda x: len(x) > 2 or x in TWO_LETTER_ENTITIES)]
                curation_app._curated_table =\
                    curation_app._original_table.copy()
            else:
                curation_app._original_linked_table =\
                    curation_app._original_linked_table[
                        curation_app._original_linked_table.entity.apply(
                            lambda x: len(x) > 2 or x in TWO_LETTER_ENTITIES)]
                curation_app._curated_table =\
                    curation_app._original_linked_table.copy()

        result = curation_app._curated_table

        # Filter by properties
        dff = result
        if filter_query:
            filtering_expressions = filter_query.split(' && ')
            for filter_part in filtering_expressions:
                col_name, operator, filter_value = split_filter_part(
                    filter_part)

                if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):

                    dff = dff.loc[getattr(dff[col_name], operator)(
                        filter_value)]
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
            page_current * page_size:(page_current + 1) * page_size
        ]

        page_count = len(result_sorted) // page_size

        return (
            result_paginated.to_dict('records'),
            curation_app._table_columns,
            True, True, page_count
        )
    except Exception:
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
            df, x=df.entity, y=df.paper_frequency,
            color=df.entity_type.apply(
                lambda x: ",".join(x) if isinstance(x, list) else x))
    return [scatter]


@curation_app._app.callback(
    Output("top_n_frequent_entity", "children"),
    [Input("topnentityslider", "value")]
)
def set_n_most_frequent(topnentityslider_value):
    curation_app.n_most_frequent = topnentityslider_value
    return "Generate Graphs from top {} frequent entities".format(
        topnentityslider_value)
