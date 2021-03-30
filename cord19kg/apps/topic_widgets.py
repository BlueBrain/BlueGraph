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

import ast
import jwt
import os

import ipywidgets
from IPython.display import display

import pandas as pd
import json
import pickle

import time
import tempfile

from kgforge.specializations.resources import Dataset
from bluegraph.core.io import PandasPGFrame

import math


def convert_size(size_bytes):
    # taken from the answer
    # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class TopicWidget(object):
    """Widget for loading and selecting topics from Nexus."""

    def __init__(self, forge, token):
        """Initialize a topic widget.

        Parameters
        ----------
        forge : kgforge.forge
            Nexus forge object
        token : str
            Nexus authorization token
        """
        self.forge = forge
        self.agent_username = jwt.decode(
            token, verify=False)['preferred_username']
        self.realm = jwt.decode(token, verify=False)["iss"].split("/")[-1]

        self.table_extractions = None
        self.curated_table_extractions = None
        self.curation_meta_data = None
        self.loaded_graphs = None
        self.visualization_configs = None
        self.edit_history = None

        self._topics_df = None
        self._topic_resource = None
        self._kg_resources = None

        # initialize the layout of the widget
        self._widget_elements = [
            ipywidgets.Button(
                description='🔬 List all your topics',
                button_style='',
                layout=ipywidgets.Layout(width='300px', height='30px'),
                disabled=False),
            ipywidgets.Button(
                description="📃 Show datasets for selected topic",
                button_style='',
                layout=ipywidgets.Layout(width='300px', height='30px'),
                disabled=False),
            ipywidgets.RadioButtons(
                description='Select:',
                disabled=False),
            ipywidgets.Button(
                description='📈 Reuse selected dataset',
                button_style='',
                layout=ipywidgets.Layout(width='300px', height='30px'),
                disabled=False),
            ipywidgets.Button(
                description='✏️ Update topic',
                button_style='',
                layout=ipywidgets.Layout(width='300px', height='30px'),
                disabled=False),
            ipywidgets.Text(
                description='Field:',
                disabled=False),
            ipywidgets.Textarea(
                description='Description:',
                disabled=False),
            ipywidgets.Textarea(
                description='Keywords:',
                disabled=False),
            ipywidgets.Text(
                disabled=False),
            ipywidgets.Text(
                disabled=False),
            ipywidgets.Text(
                disabled=False),
            ipywidgets.Text(
                disabled=False)
        ]
        self.sq = ipywidgets.VBox(children=self._widget_elements[8:12])

        self._widget_element_group = ipywidgets.VBox(
            children=self._widget_elements[5:8] + [
                ipywidgets.Label('Questions:'),
                self.sq, self._widget_elements[4]
            ])

        self._topic_form_elements = [
            ipywidgets.Dropdown(
                description='Select topic:',
                disabled=False),
            ipywidgets.Text(
                placeholder='e.g. COVID-19',
                description='Topic name:',
                disabled=False),
            ipywidgets.Text(
                placeholder='e.g. Neuroscience',
                description='Field:',
                disabled=False),
            ipywidgets.Textarea(
                placeholder='Add a description of your topic',
                description='Description:',
                disabled=False),
            ipywidgets.Textarea(
                placeholder=(
                    'e.g. Coronavirus; COVID-19; SARS; risk factor; ' +
                    'glycosylation; sugar; carbohydrates'
                ),
                description='Keywords:',
                disabled=False),
            ipywidgets.Text(
                placeholder='Add a question about your research topic',
                disabled=False),
            ipywidgets.Text(
                placeholder='Add a question about your research topic',
                disabled=False),
            ipywidgets.Text(
                placeholder='Add a question about your research topic',
                disabled=False),
            ipywidgets.Text(
                placeholder='Add a question about your research topic',
                disabled=False),
            ipywidgets.Button(
                description='Create',
                button_style='',
                tooltip='Create new topic',
                disabled=False)
        ]

        self._select_topic_output = ipywidgets.Output()
        self._new_topic_output = ipywidgets.Output()
        self._load_dataset_output = ipywidgets.Output()

        buttons = ipywidgets.HBox(children=self._widget_elements[:2])
        outputs = ipywidgets.HBox(
            children=[self._select_topic_output, self._load_dataset_output])
        tab1 = ipywidgets.VBox(children=[buttons, outputs])
        tab2 = ipywidgets.VBox(
            children=self._topic_form_elements[1:5] + [
                ipywidgets.Label(
                    'Please express your research topic in a few questions:')
            ] + self._topic_form_elements[5:10] + [self._new_topic_output]
        )

        self._widget = ipywidgets.Tab(children=[tab1, tab2])
        self._widget.set_title(0, 'Select topic')
        self._widget.set_title(1, 'Create topic')

        self._topic_form_elements[9].on_click(self.save_topic)
        self._widget_elements[0].on_click(self.get_topics)
        self._widget_elements[1].on_click(self.get_datasets)
        self._widget_elements[3].on_click(self.download_dataset)
        self._widget_elements[4].on_click(self.update_topic)

    def save_topic(self, b):
        self._select_topic_output.clear_output()
        self._new_topic_output.clear_output()
        self._load_dataset_output.clear_output()
        topic_to_save = {
            'id': str(self._widget.children[1].children[0].value).replace(
                ' ', '_'),
            'type': 'Topic',
            'name': self._widget.children[1].children[0].value,
            'field': self._widget.children[1].children[1].value,
            'description': self._widget.children[1].children[2].value,
            'keywords': self._widget.children[1].children[3].value,
            'question': [
                self._widget.children[1].children[i].value
                for i in range(5, 9)]
        }
        self._topic_resource = self.forge.from_json(topic_to_save)
        self.forge.register(self._topic_resource)
        with self._new_topic_output:
            if self._topic_form_elements[1].value == "":
                print("Please provide a topic name")
            else:
                print("Topic saved!")
                for i in range(1, 9):
                    self._topic_form_elements[i].value = ""

    def get_topics(self, b):
        self._select_topic_output.clear_output()
        self._new_topic_output.clear_output()
        self._load_dataset_output.clear_output()
        query = f"""
        SELECT ?id ?name ?description ?keywords ?field ?question ?createdAt
        WHERE {{
            ?id a Topic ;
                name ?name .
            OPTIONAL {{ ?id description ?description }}
            OPTIONAL {{ ?id keywords ?keywords }}
            OPTIONAL {{ ?id field ?field }}
            OPTIONAL {{ ?id question ?question }}
            ?id <https://bluebrain.github.io/nexus/vocabulary/deprecated> false ;
                <https://bluebrain.github.io/nexus/vocabulary/createdAt> ?createdAt ;
                <https://bluebrain.github.io/nexus/vocabulary/createdBy> <{self.forge._store.endpoint}/realms/{self.realm.lower()}/users/{self.agent_username}> .
        }}
        """
        resources = self.forge.sparql(query, limit=100)
        if len(resources) >= 1:
            self._topics_df = self.forge.as_dataframe(resources)
            self._select_topic_output.clear_output()
            with self._select_topic_output:
                topics_list = list(set(self._topics_df.name))
                topics_list.sort()
                self._topic_form_elements[0].options = [""] + topics_list
                self._topic_form_elements[0].value = ""
                self._topic_form_elements[0].placeholder = "Select topic"
                self._topic_form_elements[0].observe(self.topics_change, names='value')
                display(self._topic_form_elements[0])
                display(self._widget_element_group)
        else:
            with self._select_topic_output:
                print("No topics found!")

    def topics_change(self, change):
        self._load_dataset_output.clear_output()
        with self._select_topic_output:
            if len(self._select_topic_output.outputs) >= 1:
                self._select_topic_output.outputs = (
                    self._select_topic_output.outputs[0],)
            self._widget_elements[5].value = ""
            self._widget_elements[6].value = ""
            self._widget_elements[7].value = ""
            self._widget_elements[8].value = ""
            self._widget_elements[9].value = ""
            self._widget_elements[10].value = ""
            self._widget_elements[11].value = ""
            if change['new'] != "":
                self._topic_resource = self.forge.retrieve(
                    list(set(self._topics_df[
                        self._topics_df.name == change['new']].id))[0])
                self._widget_elements[5].value = self._topic_resource.field
                self._widget_elements[6].value =\
                    self._topic_resource.description
                self._widget_elements[7].value = self._topic_resource.keywords
                question = self._topic_resource.question
                if isinstance(question, str):
                    question = [question]
                if isinstance(question, list):
                    for i in range(len(question)):
                        self.sq.children[i].value = question[i]
            display(self._widget_element_group)

    def update_topic(self, b):
        self._new_topic_output.clear_output()
        if self._topic_form_elements[0].value != "":
            self._topic_resource.id = self.forge.as_jsonld(
                self._topic_resource, form="expanded")['@id']
            self._topic_resource.field = self._widget_elements[5].value
            self._topic_resource.description = self._widget_elements[6].value
            self._topic_resource.keywords = self._widget_elements[7].value
            self._topic_resource.question = [
                self.sq.children[i].value for i in range(0, 4)]
            self.forge.update(self._topic_resource)
            with self._select_topic_output:
                print("topic updated!")

    def get_datasets(self, b):
        self._load_dataset_output.clear_output()
        if self._topic_form_elements[0].value != "":
            topic_resource_id = self.forge.as_jsonld(
                self._topic_resource, form="expanded")['@id']
            query = f"""
                SELECT ?id ?name ?description ?keywords ?field ?question ?createdAt
                WHERE {{
                    ?id a Dataset ;
                        name ?name ;
                        about <{topic_resource_id}> ;
                        <https://bluebrain.github.io/nexus/vocabulary/deprecated> false ;
                        <https://bluebrain.github.io/nexus/vocabulary/createdAt> ?createdAt ;
                        <https://bluebrain.github.io/nexus/vocabulary/createdBy> <{self.forge._store.endpoint}/realms/bbp/users/{self.agent_username}> .
                }}
                """
            self._kg_resources = self.forge.sparql(query, limit=100)
            if len(self._kg_resources) >= 1:
                with self._load_dataset_output:
                    display(self._widget_elements[2])
                    self._widget_elements[2].options = [
                        r.name for r in self._kg_resources]
                    display(self._widget_elements[3])
            else:
                with self._load_dataset_output:
                    print("No datasets found!")

    def download_dataset(self, b):
        resource_id = [
            r.id
            for r in self._kg_resources
            if r.name == self._widget_elements[2].value][0]
        self._kg_resource = self.forge.retrieve(resource_id)
        self.forge.download(
            self._kg_resource, "distribution.contentUrl", "/tmp/",
            overwrite=True)
        for r in self._kg_resource.distribution:
            message = ""

            # Read extracted and curated table
            if "table_extractions" in r.name:
                if "curated" in r.name:
                    self.curated_table_extractions = pd.read_csv(
                        f"/tmp/{r.name}", index_col="entity")
                    collection_columns = [
                        "aggregated_entities", "raw_entity_types", "paragraph",
                        "paper", "section", "taxonomy"
                    ]
                    for c in collection_columns:
                        try:
                            self.curated_table_extractions[c] =\
                                self.curated_table_extractions[c].apply(
                                    ast.literal_eval)
                        except:
                            print(c)

                    message += (
                        f"Loaded curated table '{r.name}' ("
                        f"{len(self.curated_table_extractions)} entities)\n"
                    )
                else:
                    self.table_extractions = pd.read_csv(f"/tmp/{r.name}")
                    message += (
                        f"Loaded raw mentions table '{r.name}' "
                        f"({len(self.table_extractions)} rows)\n"
                    )

            # Read the graph objects
            if "graphs" in r.name:
                with open(f"/tmp/{r.name}", "r") as f:
                    json_data = json.load(f)
                    self.loaded_graphs = {}
                    for g, data in json_data.items():
                        self.loaded_graphs[g] = {}
                        self.loaded_graphs[g]["graph"] = PandasPGFrame.from_json(
                            data["graph"])
                        self.loaded_graphs[g]["tree"] = PandasPGFrame.from_json(
                            data["tree"])
                    message += (
                        f"Loaded graph objects '{r.name}' "
                        f"({list(self.loaded_graphs.keys())})\n"
                    )
            # Read the visualization app configs
            if "visualization_session" in r.name:
                with open(f"/tmp/{r.name}", "r") as f:
                    self.visualization_configs = json.load(f)
                    message += f"Loaded visualization session '{r.name}'\n"

            # Read the list of nodes to keep
            if "curation_meta_data" in r.name:
                with open(f"/tmp/{r.name}", "r") as f:
                    self.curation_meta_data = json.load(f)
                    message += f"Loaded curation meta-data '{r.name}''\n"

            with self._load_dataset_output:
                print(message)

    def get_table_extractions(self):
        return self.table_extractions

    def curated_table_extractions(self):
        return self.curated_table_extractions

    def get_loaded_graphs(self):
        return self.loaded_graphs

    def get_visualization_configs(self):
        return self.visualization_configs

    def get_curation_meta_data(self):
        return self.curation_meta_data

    def get_topic_resource_id(self):
        return self._topic_resource.id

    def get_all(self):
        return (
            self.table_extractions,
            self.curated_table_extractions,
            self.curation_meta_data,
            self.loaded_graphs,
            self.visualization_configs,
            self._topic_resource.id if self._topic_resource else None
        )

    def display(self):
        """Launch the topic widget."""
        display(self._widget)


class DataSaverWidget(object):
    """Widget for saving analysis data under topics stored in Nexus."""

    def __init__(self, forge, token, topic_resource_id,
                 table_extractions=None, curated_table_extractions=None,
                 curation_meta_data=None, exported_graphs=None,
                 visualization_configs=None, edit_history=None,
                 temp_prefix=None):
        """Initialize a data saver widget.

        Parameters
        ----------
        forge : kgforge.forge
            Nexus forge object
        token : str
            Nexus authorization token
        topic_resource_id : str
            Identifier of the selected topic resource in the Nexus store
        table_extractions : pd.DataFrame, optional
            Raw entity occurrence data table to save
        curated_table_extractions : pd.DataFrame, optional
            Curated entity occurrence data table to save
        curation_meta_data : dict, optional
            Curation meta-data
        exported_graphs : dict, optional
            Graphs exported from the visualization app
        visualization_configs : dict, optional
            Graph visualization app configs
        edit_history : dict, optional
            History of graph edits from the visualization app
        temp_prefix : str, optional
            Prefix to use for saving temp files
        """
        if topic_resource_id is None:
            raise ValueError(
                "Topic resource ID is undefined, make sure "
                "you have selected a topic."
            )

        self.forge = forge

        agent = jwt.decode(token, verify=False)
        self.agent = self.forge.reshape(
            self.forge.from_json(agent), keep=[
                "name", "email", "sub", "preferred_username"])
        self.agent.id = self.agent.sub
        self.agent.type = "Person"

        timestr = time.strftime("%Y%m%d-%H%M%S")

        self.tempdir = tempfile.TemporaryDirectory(prefix=temp_prefix)
        self.temp_prefix = self.tempdir.name

        self.dataset = Dataset(
            self.forge, name="A dataset", about=topic_resource_id)

        # Temporally save the extracted entities csv file locally
        if table_extractions is not None:
            table_extractions_filename =\
                "{}/table_extractions_{}.csv".format(self.temp_prefix, timestr)
            table_extractions.to_csv(table_extractions_filename)
            print("Extracted table file: ", convert_size(os.path.getsize(
                table_extractions_filename)))
            self.dataset.add_distribution(
                table_extractions_filename,
                content_type="application/csv")

        # Temporally save the curated list of extracted entities csv
        # file locally
        if curated_table_extractions is not None:
            curated_table_extractions_filename =\
                "{}/curated_table_extractions_{}.csv".format(
                    self.temp_prefix, timestr)
            curated_table_extractions.to_csv(
                curated_table_extractions_filename)
            print("Curated table file: ", convert_size(os.path.getsize(
                curated_table_extractions_filename)))
            self.dataset.add_distribution(
                curated_table_extractions_filename,
                content_type="application/csv")

            # Temporally save the curated list of extracted entities csv
            # file locally
            curation_meta_data_filename = "{}/curation_meta_data_{}.json".format(
                self.temp_prefix, timestr)
            with open(curation_meta_data_filename, "w") as f:
                json.dump(curation_meta_data, f)
            print("Curation meta-data filename file: ", convert_size(
                os.path.getsize(curation_meta_data_filename)))
            self.dataset.add_distribution(
                curation_meta_data_filename, content_type="application/json")

        if exported_graphs is not None:
            # Save graph objects produced by the app
            graphs_filename = "{}/graphs_{}.json".format(
                self.temp_prefix, timestr)
            with open(graphs_filename, "w") as f:
                json.dump(exported_graphs, f)
            print("Graph file: ", convert_size(
                os.path.getsize(graphs_filename)))
            self.dataset.add_distribution(
                graphs_filename, content_type="application/json")

            # Save app configs
            config_filename = "{}/visualization_session_{}.json".format(
                self.temp_prefix, timestr)
            with open(config_filename, "w+") as f:
                json.dump(visualization_configs, f)
            print("Visualization session file: ", convert_size(
                os.path.getsize(config_filename)))

            # Save manual edits
            edit_history_filename = "{}/edit_history_{}.json".format(
                self.temp_prefix, timestr)
            with open(edit_history_filename, "w") as f:
                json.dump(edit_history, f)

            self.dataset.add_distribution(
                config_filename, content_type="application/json")
            self.dataset.add_distribution(
                edit_history_filename, content_type="application/json")

        self.dataset.add_contribution(self.agent, versioned=False)
        self.dataset.contribution.hadRole = "Scientists"

        self.version = self.agent.preferred_username + "_" + timestr
        # print(self.forge.as_jsonld(self.dataset))

        # initialize the layout of the widget
        self.register_output = ipywidgets.Output()
        self.tag_output = ipywidgets.Output()

        register_button = ipywidgets.Button(
            description='💾  Register Dataset',
            button_style='',
            layout=ipywidgets.Layout(width='300px', height='30px'),
            disabled=False)

        tag_button = ipywidgets.Button(
            description='🔖 Tag Dataset',
            button_style='',
            layout=ipywidgets.Layout(width='300px', height='30px'),
            disabled=False)

        self.name_element = ipywidgets.Text(
            placeholder='Add a name for your dataset',
            description='Name:',
            disabled=False)

        self.description_element = ipywidgets.Textarea(
            placeholder='Add a description of your dataset',
            description='Description:',
            disabled=False)

        self.version_element = ipywidgets.Text(
            description='Tag:',
            value=self.version,
            disabled=False)

        register_button.on_click(self.register_dataset)
        tag_button.on_click(self.version_dataset)

        self._widget = ipywidgets.VBox(children=[
            self.name_element, self.description_element, register_button,
            self.register_output, self.version_element, tag_button,
            self.tag_output
        ])

    def register_dataset(self, b):
        self.register_output.clear_output()
        self.tag_output.clear_output()

        self.dataset.name = self.name_element.value
        self.dataset.description = (
            self.description_element.value
            if self.description_element.value else "No description provided"
        )

        self.forge.register(self.dataset)

        self.tempdir.cleanup()

        if self.dataset._last_action.succeeded is True:
            with self.register_output:
                print("Dataset registered!")
        else:
            with self.register_output:
                print(self.dataset._last_action.message)

    def version_dataset(self, b):
        self.tag_output.clear_output()
        version = self.version_element.value
        self.forge.tag(self.dataset, version)
        if self.dataset._last_action.succeeded is True:
            with self.tag_output:
                print(f"Tagged with: {str(version)}")
        else:
            with self.register_output:
                print(self.dataset._last_action.message)

    def display(self):
        """Launch the data saver widget."""
        display(self._widget)
