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
"""Collection of data structures for representing property graphs as data frames."""
from abc import ABC, abstractmethod

import json
import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from bluegraph.core.utils import (_aggregate_values,
                                  element_has_type,
                                  str_to_set)
from bluegraph.exceptions import BlueGraphException


class PGFrame(ABC):
    """Class for storing typed property graphs as a collection of frames."""

    def __init__(self, nodes=None, edges=None, default_prop_type='category'):

        self._nodes = self._create_frame(["@id"])
        self._edges = self._create_frame(["@source_id", "@target_id"])

        self._node_prop_types = {}
        self._edge_prop_types = {}

        self.default_prop_type = default_prop_type

        if nodes is not None:
            self.add_nodes(nodes)

        if edges is not None:
            self.add_edges(edges)

    # ----------------- Abstract methods -----------------

    @staticmethod
    @abstractmethod
    def _create_frame(columns):
        pass

    @abstractmethod
    def add_nodes(self, node_ids):
        pass

    @abstractmethod
    def add_edges(self, edge_ids):
        pass

    @abstractmethod
    def add_node_properties(self, prop_column, prop_type=None):
        pass

    @abstractmethod
    def remove_node_properties(self, prop_column):
        pass

    @abstractmethod
    def add_edge_properties(self, prop_column, prop_type=None):
        pass

    @abstractmethod
    def remove_edge_properties(self, prop_column):
        pass

    @abstractmethod
    def remove_nodes(self, nodes_to_remove):
        pass

    @abstractmethod
    def rename_nodes(self, mapping):
        pass

    @abstractmethod
    def remove_edges(self, edges_to_remove):
        pass

    @staticmethod
    @abstractmethod
    def _is_numeric_column(frame, column):
        pass

    @staticmethod
    @abstractmethod
    def _is_string_column(frame, column):
        pass

    @abstractmethod
    def node_types(self, flatten=False):
        """Return a set of node types."""
        pass

    @abstractmethod
    def edge_types(self, flatten=False):
        """Return a set of edges types."""
        pass

    @abstractmethod
    def has_node_types(self):
        pass

    @abstractmethod
    def has_edge_types(self):
        pass

    @abstractmethod
    def nodes(self, typed_by=None, raw_frame=False):
        """Return a list of nodes."""
        pass

    @abstractmethod
    def edges(self, typed_by=None, raw_frame=False):
        """Return a list of edges."""
        pass

    @abstractmethod
    def node_properties(self, node_type=None, include_type=None):
        """Return a list of node properties"""
        pass

    @abstractmethod
    def edge_properties(self, edge_type=None, include_type=None):
        """Return a list of edge properties."""
        pass

    @abstractmethod
    def number_of_nodes(self, node_type=None):
        """Return a number of nodes."""
        pass

    @abstractmethod
    def number_of_edges(self, edge_type=None):
        """Return a number of nodes."""
        pass

    @abstractmethod
    def _write_node(self, node_id, node_type, context, attrs=None):
        pass

    @abstractmethod
    def _write_edge(self, source_id, target_id, edge_type, attrs=None):
        pass

    @abstractmethod
    def _aggregate_nodes(self):
        pass

    @abstractmethod
    def _aggregate_edges(self):
        pass

    @abstractmethod
    def get_node_typing(self):
        pass

    @abstractmethod
    def get_edge_typing(self):
        pass

    @staticmethod
    @abstractmethod
    def aggregate_properties(frame, func, into="aggregation_result"):
        pass

    @staticmethod
    @abstractmethod
    def _export_csv(frame, path):
        pass

    @staticmethod
    @abstractmethod
    def _load_csv(path, index_col=None):
        pass

    @abstractmethod
    def to_triples(self, predicate_prop="@type", include_type=True,
                   include_literals=True):
        pass

    @abstractmethod
    def isolated_nodes(self):
        pass

    @abstractmethod
    def remove_isolated_nodes(self):
        pass

    @abstractmethod
    def to_json(self):
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, json_data):
        pass

    @abstractmethod
    def export_json(self, path):
        pass

    @classmethod
    @abstractmethod
    def load_json(cls, path):
        pass

    @classmethod
    @abstractmethod
    def from_frames(self, nodes, edges):
        pass

    @abstractmethod
    def copy(self):
        """Copy the PGFrame."""
        pass

    # -------------------- Concrete methods --------------------

    def _valid_node_prop_type(self, prop, prop_type):
        if prop_type == "text":
            return self._is_string_column(self._nodes, prop)
        elif prop_type == "numeric":
            return self._is_numeric_column(self._nodes, prop)
        return True

    def _valid_edge_prop_type(self, prop, prop_type):
        if prop_type == "text":
            return self._is_string_column(self._edges, prop)
        elif prop_type == "numeric":
            return self._is_numeric_column(self._edges, prop)
        return True

    def _set_default_prop_types(self):
        self._node_prop_types["@type"] = "category"

        for prop in self.node_properties():
            if self._is_numeric_column(self._nodes, prop):
                self._node_prop_types[prop] = "numeric"
            else:
                self._node_prop_types[prop] = "category"

        for prop in self.edge_properties():
            if self._is_numeric_column(self._edges, prop):
                self._edge_prop_types[prop] = "numeric"
            else:
                self._edge_prop_types[prop] = "category"

    def _set_node_prop_type(self, prop, prop_type):
        if self._valid_node_prop_type(prop, prop_type):
            self._node_prop_types[prop] = prop_type
        else:
            raise ValueError(
                "Cannot cast the values of the node property "
                f"'{prop}' to '{prop_type}'"
            )

    def _set_edge_prop_type(self, prop, prop_type):
        if self._valid_edge_prop_type(prop, prop_type):
            self._edge_prop_types[prop] = prop_type
        else:
            raise ValueError(
                "Cannot cast the values of the edge property "
                f" '{prop}' to '{prop_type}'"
            )

    def node_prop_as_category(self, prop):
        self._set_node_prop_type(prop, "category")

    def edge_prop_as_category(self, prop):
        self._set_edge_prop_type(prop, "category")

    def node_prop_as_text(self, prop):
        self._set_node_prop_type(prop, "text")

    def edge_prop_as_text(self, prop):
        self._set_edge_prop_type(prop, "text")

    def node_prop_as_numeric(self, prop):
        self._set_node_prop_type(prop, "numeric")

    def edge_prop_as_numeric(self, prop):
        self._set_edge_prop_type(prop, "numeric")

    def is_categorical_node_prop(self, prop):
        if prop in self._node_prop_types:
            return self._node_prop_types[prop] == "category"
        else:
            return False

    def is_categorical_edge_prop(self, prop):
        if prop in self._edge_prop_types:
            return self._edge_prop_types[prop] == "category"
        else:
            return False

    def is_text_node_prop(self, prop):
        if prop in self._node_prop_types:
            return self._node_prop_types[prop] == "text"
        else:
            return False

    def is_text_edge_prop(self, prop):
        if prop in self._edge_prop_types:
            return self._edge_prop_types[prop] == "text"
        else:
            return False

    def is_numeric_node_prop(self, prop):
        if prop in self._node_prop_types:
            return self._node_prop_types[prop] == "numeric"
        else:
            return False

    def is_numeric_edge_prop(self, prop):
        if prop in self._edge_prop_types:
            return self._edge_prop_types[prop] == "numeric"
        else:
            return False

    def nodes_of_type(self, node_type):
        return self.nodes(typed_by=node_type, raw_frame=True)

    def edges_of_type(self, edge_type):
        return self.edges(typed_by=edge_type, raw_frame=True)

    def aggregate_node_properties(self, func, into="aggregation_result"):
        self._nodes = self.aggregate_properties(self._nodes, func, into)

    def aggregate_edge_properties(self, func, into="aggregation_result"):
        self._edges = self.aggregate_properties(self._edges, func, into)

    def assign_node_types(self, node_types):
        self.add_node_properties(node_types)

    def assign_edge_types(self, edge_types):
        self.add_edge_properties(edge_types)

    def _nodes_edges_from_dict(self, source, relation, source_attrs, record,
                               include_context=True, type_handler=None,
                               types_from_relations=True,
                               exclude=[], only_props=False):
        """Retreive nodes and edges from a resource."""
        node_id = None
        if "@id" in record:
            node_id = record["@id"]
        attrs = {}
        node_type = None
        context = None
        neighbours = {}

        for k, v in record.items():
            if k != "@id":
                key = k.replace("@", "")
                if key == "type":
                    if type_handler:
                        node_type = type_handler(v)
                    else:
                        node_type = v
                elif key == "context":
                    if include_context:
                        context = v
                elif key not in exclude:
                    if not isinstance(v, dict):
                        if isinstance(v, list):
                            list_property = False
                            for el in v:
                                if not isinstance(el, dict):
                                    list_property = True
                                    break
                                else:
                                    if key not in neighbours:
                                        neighbours[key] = []
                                    if not only_props:
                                        neighbours[key].append(
                                            self._nodes_edges_from_dict(
                                                node_id if node_id is not None
                                                else source,
                                                key
                                                if node_id is not None
                                                else relation + "." + key,
                                                attrs if node_id is not None
                                                else source_attrs,
                                                el,
                                                include_context,
                                                type_handler,
                                                types_from_relations,
                                                exclude)
                                        )
                            if list_property:
                                attrs[key] = set(v)
                        else:
                            attrs[key] = v
                    elif not only_props:
                        neighbours[key] = self._nodes_edges_from_dict(
                            node_id if node_id is not None else source,
                            key if node_id is not None else relation + "." + key,
                            attrs if node_id is not None else source_attrs,
                            v,
                            include_context,
                            type_handler,
                            types_from_relations,
                            exclude)

        if node_type is None and types_from_relations:
            node_type = [relation]

        if node_id is not None:
            self._write_node(node_id, node_type, context, attrs)
            for t, ns in neighbours.items():
                edge_type = t
                elements = ns if isinstance(ns, list) else [ns]
                for el in elements:
                    if el is not None:
                        edge_attrs = {}
                        self._write_edge(node_id, el, edge_type, edge_attrs)
        else:
            if len(neighbours) == 0 and source_attrs:
                for k, v in attrs.items():
                    source_attrs[relation + "." + k] = v
            else:
                for t, ns in neighbours.items():
                    edge_type = t
                    elements = ns if isinstance(ns, list) else [ns]
                    for el in elements:
                        self._write_edge(source, el, edge_type, attrs)
        return node_id

    def from_jsonld(self, resources, include_context=True, type_handler=None,
                    types_from_relations=True, exclude=None, only_props=False):
        """Create a PGFrame from jsonld.

        Parameters
        ----------
        resources : iterable of dict
            Collection of input resources in JSON-LD format
        include_context : bool, optional
            Flag indicating if the context should be included as a property. Default is True.
        type_handler : func, optional
            Function to apply to the value of type (e.g. '@type')
        types_from_relations : bool, optional
            Flag indicating if resources with unkown types should be assigned with types
            from the incoming relations. Default is True
        exclude : list of str, optional
            Collection of property names to exclude. Default is empty.
        only_props : bool, optional
            Flag indicating if the procedure should extract only
            properties from the given resources.
        """
        if exclude is None:
            exclude = []
        for r in resources:
            self._nodes_edges_from_dict(
                None, None, None, r,
                include_context,
                type_handler,
                types_from_relations,
                exclude,
                only_props)
        self._aggregate_nodes()
        self._aggregate_edges()

        # Set default node and edge types
        self._set_default_prop_types()

    def to_jsonld(self, edges_key="edges"):
        """Create a JSON-LD representation of the PGFrame."""
        def _normalize_to_set(x):
            return [x] if isinstance(x, str) else x

        def aggregate_nodes(x):
            node = {
                "@id": str(x.name)
            }
            if x["@type"]:
                node["@type"] = _normalize_to_set(x["@type"])

            for k in x.keys():
                if k != "@type":
                    node[k] = x[k]
            try:
                incident_edges = self._edges.xs(
                    x.name, level=0, axis=0).to_dict("index")
                edges = []
                for target, edge_props in incident_edges.items():
                    edge_type = edge_props["@type"]
                    del edge_props["@type"]
                    edge_props[edge_type] = {
                        "@id": str(target)
                    }
                    edges.append(edge_props)
                node[edges_key] = edges
            except KeyError:
                pass
            return node

        nodes = self._nodes.apply(aggregate_nodes, axis=1).to_list()
        return nodes

    def to_csv(self, node_path, edge_path):
        self._export_csv(self._nodes, node_path)
        self._export_csv(self._edges, edge_path)

    @classmethod
    def from_csv(cls, node_path, edge_path=None,
                 node_property_types=None, edge_property_types=None):
        graph = cls()
        graph._nodes = graph._load_csv(
            node_path, index_col="@id")

        graph._edges = graph._load_csv(
            edge_path, index_col=["@source_id", "@target_id"])
        # Set default node and edge types
        graph._set_default_prop_types()
        if node_property_types:
            graph._node_prop_types.update(node_property_types)
        if edge_property_types:
            graph._edge_prop_types.update(edge_property_types)
        return graph

    def export_to_gephi(self, prefix, node_attr_mapping,
                        edge_attr_mapping, edge_filter=None):
        """Save the graph for Gephi import.

        Saves the graph as two `.csv` files one with nodes
        (`<prefix>_nodes.csv`) and one with edges (
        `<prefix>_edges.csv`). Node IDs are replaced by
        interger identifiers (Gephi asks for node IDs to be numerical) and
        entity names are added as the node property 'Label'.
        """
        ordered_edge_attrs = list(edge_attr_mapping.keys())
        edge_header = "Source;Target;{}\n".format(
            ";".join([
                edge_attr_mapping[attr]
                for attr in ordered_edge_attrs
            ]))

        def generate_edge_repr(edge_props):
            return ";".join([
                str(edge_props[attr])
                for attr in ordered_edge_attrs])

        edge_repr = "\n".join([
            "{};{};{}".format(
                self._nodes.index.get_loc(u) + 1,
                self._nodes.index.get_loc(v) + 1,
                generate_edge_repr(self.get_edge(u, v)))
            for u, v in self.edges()
            if edge_filter is None or edge_filter(
                u, v, self.get_edge(u, v))
        ])

        with open("{}_edges.csv".format(prefix), "w+") as f:
            f.write(edge_header + edge_repr)

        ordered_node_attrs = list(node_attr_mapping.keys())
        node_header = "Id;Label;{}\n".format(
            ";".join([
                node_attr_mapping[attr]
                for attr in ordered_node_attrs
            ]))

        def generate_node_repr(node_props):
            return ";".join([
                str(node_props[attr])
                for attr in ordered_node_attrs])

        node_repr = "\n".join([
            "{};{};{}".format(
                self._nodes.index.get_loc(n) + 1,
                n,
                generate_node_repr(
                    self.get_node(n))
            )
            for n in self.nodes()
        ])

        with open("{}_nodes.csv".format(prefix), "w+") as f:
            f.write(node_header + node_repr)

    def density(self, directed=True):
        total_edges = self.number_of_nodes() * (self.number_of_nodes() - 1)
        if directed is False:
            total_edges = total_edges / 2

        return self.number_of_edges() / total_edges

    class PGFrameException(BlueGraphException):
        pass


class PandasPGFrame(PGFrame):
    """Class for storing typed PGs as a collection of pandas DataFrames."""

    # ------------- Implementation of abstract methods --------------

    @staticmethod
    def _create_frame(columns):
        df = pd.DataFrame(columns=columns)
        df = df.set_index(columns)
        return df

    def add_nodes(self, node_ids):
        """Add node ids to the PG frame."""
        new_df = pd.DataFrame({"@id": node_ids})
        new_df = new_df.set_index("@id")
        self._nodes = self._nodes.append(new_df)

    def add_edges(self, edges):
        sources = [e[0] for e in edges]
        targets = [e[1] for e in edges]

        new_df = pd.DataFrame({"@source_id": sources, "@target_id": targets})
        new_df = new_df.set_index(["@source_id", "@target_id"])
        self._edges = self._edges.append(new_df)

    def add_node_types(self, type_dict):
        type_df = pd.DataFrame(
            type_dict.items(), columns=["@id", "@type"])
        type_df = type_df.set_index("@id")
        self.add_node_properties(type_df)

    def add_edge_types(self, type_dict):
        type_df = pd.DataFrame(type_dict.items(), columns=["_index", "@type"])
        type_df["@source_id"] = type_df["_index"].apply(lambda x: x[0])
        type_df["@target_id"] = type_df["_index"].apply(lambda x: x[1])
        type_df = type_df[["@source_id", "@target_id", "@type"]].set_index(
            ["@source_id", "@target_id"])
        self.add_edge_properties(type_df)

    def add_node_properties(self, prop_column, prop_type=None):
        if not isinstance(prop_column, pd.DataFrame):
            prop_column = pd.DataFrame(prop_column)

        # Make sure that the prop column is indexed by '@id'
        if not prop_column.index.name and "@id" in prop_column.columns:
            prop_column = prop_column.set_index("@id")

        prop_name = prop_column.columns[0]

        if prop_name in self._nodes.columns:
            self._nodes.loc[prop_column.index, prop_name] = prop_column[
                prop_name]
        else:
            if self.number_of_nodes() == 0:
                self._nodes = prop_column
            else:
                self._nodes = self._nodes\
                    .join(prop_column, rsuffix="_right")
                if "@id_right" in self._nodes.columns:
                    self._nodes = self._nodes\
                        .drop("@id_right", axis=1)\
                        .set_index("@id")

        if prop_type is None:
            prop_type = "category"

        if prop_type not in ["text", "numeric", "category"]:
            raise PGFrame.PGFrameException(
                f"Invalid property data type '{prop_type}', "
                "allowed types 'text', 'numeric', 'category'")
        self._set_node_prop_type(prop_name, prop_type)

    def remove_node_properties(self, prop_column):
        self._nodes = self._nodes.drop(columns=[prop_column])

    def rename_nodes(self, mapping):
        reset_nodes = self._nodes.reset_index()
        reset_nodes["@id"] = [
            mapping[el] if el in mapping else el
            for el in self._nodes.index
        ]
        self._nodes = reset_nodes.set_index(
            ["@id"])

        reset_edges = self._edges.reset_index()
        reset_edges["@source_id"] = [
            mapping[s] if s in mapping else s
            for s, _ in self._edges.index]
        reset_edges["@target_id"] = [
            mapping[t] if t in mapping else t
            for _, t in self._edges.index]
        self._edges = reset_edges.set_index(
            ["@source_id", "@target_id"])

    def add_edge_properties(self, prop_column, prop_type=None):
        if not isinstance(prop_column, pd.DataFrame):
            prop_column = pd.DataFrame(prop_column)

        if prop_column.index.names != ["@source_id", "@target_id"] and\
           "@source_id" in prop_column.columns and\
           "@target_id" in prop_column.columns:
            prop_column = prop_column.set_index(
                ["@source_id", "@target_id"])

        prop_name = prop_column.columns[0]

        if prop_name in self._edges.columns:
            self._edges.loc[prop_column.index, prop_name] = prop_column[
                prop_name]
        else:
            if self.number_of_edges() == 0:
                self._edges = prop_column
            else:
                self._edges = self._edges\
                    .join(prop_column, rsuffix="_right")
                if "@source_id_right" in self._edges.columns and\
                   "@target_id_right" in self._edges.columns:
                    self._edges = self._edges\
                        .drop(["@source_id_right", "@target_id_right"], axis=1)\
                        .set_index(["@source_id", "@target_id"])

        if prop_type is None:
            prop_type = "category"

        if prop_type not in ["text", "numeric", "category"]:
            raise PGFrame.PGFrameException(
                f"Invalid property data type '{prop_type}', "
                "allowed types 'text', 'numeric', 'category'")
        self._set_edge_prop_type(prop_name, prop_type)

    def remove_edge_properties(self, prop_column):
        self._edges = self._edges.drop(columns=[prop_column])

    def remove_nodes(self, nodes_to_remove):
        # Remove nodes
        self._nodes = self._nodes.loc[
            ~self._nodes.index.isin(nodes_to_remove)]
        # Detach edges
        self.remove_edges(
            self._edges.index[self._edges.index.map(
                lambda x: x[0] in nodes_to_remove or x[1] in nodes_to_remove)])

    def remove_edges(self, edges_to_remove):
        self._edges = self._edges.loc[
            ~self._edges.index.isin(edges_to_remove)]

    @staticmethod
    def _is_numeric_column(frame, prop):
        if not is_numeric_dtype(frame[prop]):
            try:
                frame[prop] = frame[prop].apply(float)
                return True
            except Exception as e:
                return False
        else:
            return True

    @staticmethod
    def _is_string_column(frame, prop):
        return is_string_dtype(frame[prop])

    def has_node_types(self):
        return "@type" in self._nodes.columns

    def has_edge_types(self):
        return "@type" in self._edges.columns

    def node_types(self, flatten=False):
        """Return a list of node types."""
        if flatten:
            types = _aggregate_values(self._nodes["@type"])
        else:
            types = []
            for el in self._nodes["@type"]:
                if el not in types:
                    types.append(el)
        return types

    def edge_types(self, flatten=False):
        """Return a list of edges types."""
        if flatten:
            types = _aggregate_values(self._edges["@type"])
        else:
            types = []
            for el in self._edges["@type"]:
                if el not in types:
                    types.append(el)
        return types

    def nodes(self, typed_by=None, raw_frame=False, include_index=False,
              filter_props=None, rename_cols=None):
        """Return a list of nodes."""
        df = self._nodes

        if typed_by is not None:
            if "@type" not in self._nodes:
                return []
            df = self._nodes[
                self._nodes["@type"].apply(
                    lambda x: element_has_type(x, typed_by))]

        if raw_frame:
            if filter_props:
                df = df.filter(
                    items=[c for c in df.columns if filter_props(c)],
                    axis=1)
            if include_index:
                df = df.reset_index()
            if rename_cols:
                df = df.rename(columns=rename_cols)
            return df
        return df.index.to_list()

    def edges(self, typed_by=None, raw_frame=False, include_index=False,
              filter_props=None, rename_cols=None):
        """Return a list of edges."""
        df = self._edges

        if typed_by is not None:
            if "@type" not in self._edges:
                return []
            df = self._edges[
                self._edges["@type"].apply(
                    lambda x: element_has_type(x, typed_by))]
        if raw_frame:
            if filter_props:
                df = df.filter(
                    items=[c for c in df.columns if filter_props(c)],
                    axis=1) 
            if include_index:
                df = df.reset_index()
            if rename_cols:
                df = df.rename(columns=rename_cols)
            return df
        return df.index.to_list()

    def node_properties(self, node_type=None, include_type=False):
        """Return a list of node properties"""
        if node_type:
            nodes_of_type = self.nodes_of_type(node_type)
            columns = nodes_of_type.columns[nodes_of_type.notna().any()].to_list()
        else:
            columns = self._nodes.columns.to_list()
        if not include_type and "@type" in columns:
            columns.remove('@type')
        return columns

    def edge_properties(self, edge_type=None, include_type=False):
        """Return a list of edge properties."""
        if edge_type:
            edges_of_type = self.edges_of_type(edge_type)
            columns = edges_of_type.columns[edges_of_type.notna().any()].to_list()
        else:
            columns = self._edges.columns.to_list()
        if not include_type and '@type' in columns:
            columns.remove('@type')
        return columns

    def get_node_property_values(self, prop, nodes=None, typed_by=None):
        if nodes is None:
            nodes = self.nodes()
        df = self._nodes.loc[nodes]
        if typed_by is not None:
            if "@type" not in self._nodes:
                return []
            df = self._nodes[
                self._nodes["@type"].apply(
                    lambda x: element_has_type(x, typed_by))]
        return df[prop]

    def get_edge_property_values(self, prop, edges=None, typed_by=None):
        if edges is None:
            edges = self.edges()
        df = self._edges.loc[edges]
        if typed_by is not None:
            if "@type" not in self._edges:
                return []
            df = self._edges[
                self._edges["@type"].apply(
                    lambda x: element_has_type(x, typed_by))]
        return df[prop]

    def get_node(self, n):
        """Get node properties."""
        return self._nodes.loc[n].to_dict()

    def get_edge(self, s, t):
        """Get edge properties."""
        return self._edges.loc[(s, t)].to_dict()

    def number_of_nodes(self):
        """Return a number of nodes."""
        return self._nodes.shape[0]

    def number_of_edges(self):
        """Return a number of nodes."""
        return self._edges.shape[0]

    def _write_node(self, node_id, node_type, context, attrs):
        if node_type is not None:
            attrs["@type"] = set(node_type)
        self._nodes = self._nodes.append(
            {"@id": node_id, **attrs}, ignore_index=True)

    def _write_edge(self, source_id, target_id, edge_type, attrs):
        attrs["@type"] = {edge_type}
        self._edges = self._edges.append({
            "@source_id": source_id,
            "@target_id": target_id,
            **attrs
        }, ignore_index=True)

    def _aggregate_nodes(self):
        res = self._nodes.groupby("@id").aggregate(
            _aggregate_values)
        self._nodes = res

    def _aggregate_edges(self):
        self._edges = self._edges.groupby(["@source_id", "@target_id"]).aggregate(
            _aggregate_values)

    def get_node_typing(self, as_dict=False):
        if "@type" in self._nodes.columns:
            types = self._nodes["@type"]
            if as_dict is True:
                types = types.to_dict()
            return types
        else:
            raise ValueError("Graph nodes are not typed")

    def get_edge_typing(self):
        if "@type" in self._edges.columns:
            return self._edges["@type"]
        else:
            raise ValueError("Graph edges are not typed")

    @staticmethod
    def aggregate_properties(frame, func, into="aggregation_result"):
        if "@type" in frame.columns:
            df = frame.drop("@type", axis=1)
            frame = pd.DataFrame(
                {
                    into: df.aggregate(func, axis=1),
                    "@type": frame["@type"]
                },
                index=frame.index)
        else:
            frame = pd.DataFrame(
                {
                    into: frame.aggregate(func, axis=1)
                },
                index=frame.index)
        return frame

    @staticmethod
    def _export_csv(frame, path):
        frame.to_csv(path)

    @staticmethod
    def _load_csv(path, index_col):
        df = pd.read_csv(path, index_col=index_col)
        if "@type" in df.columns:
            df["@type"] = df["@type"].apply(str_to_set)
        return df

    def to_triples(self, predicate_prop="@type", include_type=True,
                   include_literals=True):
        triple_sets = []

        # create triples from edges
        triple_sets.append(
            self._edges.reset_index()[
                ["@source_id", predicate_prop, "@target_id"]].to_numpy())

        # create triples from literals
        if include_literals:
            for prop in self.node_properties(include_type=include_type):
                df = pd.DataFrame(
                    self._nodes[self._nodes[prop].notna()][prop]).reset_index()
                df["predicate"] = prop
                triple_sets.append(df[["@id", "predicate", prop]].to_numpy())

        return np.concatenate(triple_sets)

    def filter_nodes(self, nodes):
        return self._nodes[self._nodes.index.isin(nodes)]

    def filter_edges(self, edges):
        return self._edges[self._edges.index.isin(edges)]

    def subgraph(self, nodes=None, edges=None, remove_isolated_nodes=False):
        if nodes is not None:
            # construct the node-induced subgraph
            if edges is None:
                edges = self.edges()
            edges = set(
                (s, t)
                for (s, t) in edges
                if s in nodes and t in nodes
            )

        elif edges is not None:
            # construct the edge-induced subgraph
            nodes = set([n for e in edges for n in e])
        else:
            raise ValueError(
                "Either node or edge set should be specified to "
                "construct a subgraph")

        subgraph = PandasPGFrame()
        subgraph._nodes = self.filter_nodes(nodes)
        subgraph._edges = self.filter_edges(edges)
        subgraph._node_prop_types = {**self._node_prop_types}
        subgraph._edge_prop_types = {**self._edge_prop_types}
        if remove_isolated_nodes is True:
            subgraph.remove_isolated_nodes()
        return subgraph

    def copy(self):
        """Create a copy of the pgframe."""
        nodes_copy = self._nodes.copy()
        edges_copy = self._edges.copy()

        node_prop_types = self._node_prop_types.copy()
        edge_prop_types = self._edge_prop_types.copy()
        return PandasPGFrame.from_frames(
            nodes_copy, edges_copy,
            node_prop_types, edge_prop_types)

    @classmethod
    def from_frames(cls, nodes, edges,
                    node_prop_types=None, edge_prop_types=None):
        graph = cls()
        graph._nodes = nodes.copy()
        graph._edges = edges.copy()

        if node_prop_types is None:
            node_prop_types = {}

        if edge_prop_types is None:
            edge_prop_types = {}

        graph._node_prop_types = node_prop_types
        graph._edge_prop_types = edge_prop_types

        return graph

    def _edge_sources(self):
        return self._edges.index.get_level_values(0).unique()

    def _edge_targets(self):
        return self._edges.index.get_level_values(1).unique()

    def isolated_nodes(self):
        nodes = self.nodes()
        sources = self._edge_sources()
        targets = self._edge_targets()
        isolates = []
        for n in nodes:
            if n not in sources and n not in targets:
                isolates.append(n)
        return isolates

    def remove_isolated_nodes(self):
        isolates = self.isolated_nodes()
        # Remove nodes
        self._nodes = self._nodes.loc[~self._nodes.index.isin(isolates)]

    def to_json(self):
        nodes_json = self._nodes.reset_index().to_dict(
            orient="records")
        edges_json = self._edges.reset_index().to_dict(
            orient="records")
        return {
            "nodes": nodes_json,
            "edges": edges_json,
            "node_property_types": self._node_prop_types,
            "edge_property_types": self._edge_prop_types
        }

    @classmethod
    def from_json(cls, json_data):
        frame = cls()
        frame._nodes = pd.DataFrame(json_data["nodes"]).set_index("@id")
        if len(json_data["edges"]) > 0:
            frame._edges = pd.DataFrame(json_data["edges"]).set_index(
                ["@source_id", "@target_id"])
        else:
            frame._edges = pd.DataFrame(columns=[
                "@source_id", "@target_id"])
        frame._node_prop_types = json_data["node_property_types"]
        frame._edge_prop_types = json_data["edge_property_types"]
        return frame

    def export_json(self, path):
        data = self.to_json()
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
            return PandasPGFrame.from_json(data)


class SparkPGFrame(PGFrame):
    """Class for storing typed PGs as a collection of Spark DataFrames."""

    def __init__(self):
        """Initalize a SparkPGFrame."""
        pass


class GraphProcessor(ABC):
    """Abstract class for a graph processor.

    The provided interface allows to convert PGFrames
    into backend-specific graph objects and vice versa. It also allows
    to access nodes/edges and properties of backend-specific graph objects
    through its interface.
    """

    def __init__(self, pgframe=None, directed=True):
        if pgframe is not None:
            self.graph = self._generate_graph(
                pgframe, directed=directed)
        else:
            self.graph = None
        self.directed = directed

    @staticmethod
    @abstractmethod
    def _generate_graph(pgframe, directed=True):
        """Generate a graph object from the pgframe (backend specific)."""
        pass

    @abstractmethod
    def _generate_pgframe(self, node_prop_types=None, edge_prop_types=None,
                          node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        pass

    @staticmethod
    @abstractmethod
    def _is_directed(graph):
        pass

    @abstractmethod
    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        pass

    @abstractmethod
    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
        pass

    @abstractmethod
    def nodes(self, properties=False):
        pass

    @abstractmethod
    def get_node(self, node):
        pass

    @abstractmethod
    def remove_node(self, node):
        pass

    @abstractmethod
    def rename_nodes(self, node_mapping):
        pass

    @abstractmethod
    def set_node_properties(self, node, properties):
        pass

    @abstractmethod
    def edges(self, properties=False):
        pass

    @abstractmethod
    def get_edge(self, edge):
        pass

    @abstractmethod
    def remove_edge(self, source, target):
        pass

    @abstractmethod
    def add_edge(self, source, target, properties=None):
        pass

    @abstractmethod
    def set_edge_properties(self, source, target, properties):
        pass

    @abstractmethod
    def subgraph(self, nodes_to_include=None, edges_to_include=None,
                 nodes_to_exclude=None, edges_to_exclude=None):
        pass

    @abstractmethod
    def neighbors(self, node_id):
        """Get neighors of the node."""
        pass

    @abstractmethod
    def _get_adjacency_matrix(self, nodes, weight=None):
        pass

    @abstractmethod
    def _get_node_property_values(self, prop, nodes):
        pass

    def _dispatch_processing_result(self, new_property, metric_name,
                                    write=False,
                                    write_property=None):
        if write:
            if write_property is None:
                raise GraphProcessor.ProcessingException(
                    "{} processing has the write option set to True, "
                    "the write property name must be specified".format(
                        metric_name.capitalize()))
            self._write_node_property(new_property, write_property)
        else:
            return self._yeild_node_property(new_property)

    @classmethod
    def from_graph_object(cls, graph_object):
        """Initialize directly from the input graph object."""
        processor = cls()
        processor.graph = graph_object
        processor.directed = cls._is_directed(graph_object)
        return processor

    def get_pgframe(self, node_prop_types=None, edge_prop_types=None,
                    node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        return self._generate_pgframe(
            node_prop_types=node_prop_types, edge_prop_types=edge_prop_types,
            node_filter=node_filter, edge_filter=edge_filter)

    class ProcessorException(BlueGraphException):
        pass
