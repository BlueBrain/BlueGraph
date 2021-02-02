"""Collection of data structures for representing property graphs as data frames."""
from abc import ABC, abstractmethod

import os
import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from bluegraph.core.utils import (_aggregate_values,
                                  element_has_type,
                                  str_to_set)
from bluegraph.exceptions import PGFrameException, BlueGraphException


class PGFrame(ABC):
    """Class for storing typed property graphs as a collection of frames."""

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
    def add_node_properties(self, prop_column):
        pass

    @abstractmethod
    def add_edge_properties(self, prop_column, prop_type=None):
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

    # -------------------- Concrete methods --------------------

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
                f"Cannot cast the values of the node property '{prop}' to '{prop_type}'"
            )

    def _set_edge_prop_type(self, prop, prop_type):
        if self._valid_edge_prop_type(prop, prop_type):
            self._edge_prop_types[prop] = prop_type
        else:
            raise ValueError(
                f"Cannot cast the values of the edge property '{prop}' to '{prop_type}'"
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

    def to_csv(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self._export_csv(self._nodes, os.path.join(path, "nodes.csv"))
        self._export_csv(self._edges, os.path.join(path, "edges.csv"))

    @classmethod
    def from_csv(cls, path, node_property_types=None, edge_property_types=None):
        graph = cls()
        if os.path.exists(path):
            graph._nodes = graph._load_csv(
                os.path.join(path, "nodes.csv"),
                index_col="@id")
            graph._edges = graph._load_csv(
                os.path.join(path, "edges.csv"),
                index_col=["@source_id", "@target_id"])
            # Set default node and edge types
            graph._set_default_prop_types()
        if node_property_types:
            graph._node_prop_types.update(node_property_types)
        if edge_property_types:
            graph._edge_prop_types.update(edge_property_types)
        return graph

    @classmethod
    @abstractmethod
    def from_frames(self, nodes, edges):
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
            raise PGFrameException(
                f"Invalid property data type '{prop_type}', "
                "allowed types 'text', 'numeric', 'category'")
        self._set_node_prop_type(prop_name, prop_type)

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
            raise PGFrameException(
                f"Invalid property data type '{prop_type}', "
                "allowed types 'text', 'numeric', 'category'")
        self._set_edge_prop_type(prop_name, prop_type)

    @staticmethod
    def _is_numeric_column(frame, prop):
        return is_numeric_dtype(frame[prop])

    @staticmethod
    def _is_string_column(frame, prop):
        return is_string_dtype(frame[prop])

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

    def get_node_property_values(self, prop, typed_by=None):
        df = self._nodes
        if typed_by is not None:
            if "@type" not in self._nodes:
                return []
            df = self._nodes[
                self._nodes["@type"].apply(
                    lambda x: element_has_type(x, typed_by))]
        return df[prop]

    def get_edge_property_values(self, prop, typed_by=None):
        df = self._edges
        if typed_by is not None:
            if "@type" not in self._edges:
                return []
            df = self._edges[
                self._edges["@type"].apply(
                    lambda x: element_has_type(x, typed_by))]
        return df[prop]

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
            frame = pd.DataFrame({
                    into: df.aggregate(func, axis=1),
                    "@type": frame["@type"]
                },
                index=frame.index)
        else:
            frame = pd.DataFrame({
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

    def to_triples(self, predicate_prop="@type", include_type=True, include_literals=True):
        triple_sets = []

        # create triples from edges
        triple_sets.append(
            self._edges.reset_index()[["@source_id", predicate_prop, "@target_id"]].to_numpy())

        # create triples from literals
        if include_literals:
            for prop in self.node_properties(include_type=include_type):
                df = pd.DataFrame(self._nodes[self._nodes[prop].notna()][prop]).reset_index()
                df["predicate"] = prop
                triple_sets.append(df[["@id", "predicate", prop]].to_numpy())

            for prop in self.edge_properties():
                df = pd.DataFrame(self._nodes[self._nodes[prop].notna()][prop]).reset_index()
                df["predicate"] = prop
                triple_sets.append(df[["@id", "predicate", prop]].to_numpy())

        return np.concatenate(triple_sets)

    def filter_nodes(self, nodes):
        return self._nodes[self._nodes.index.isin(nodes)]

    def filter_edges(self, edges):
        return self._edges[self._edges.index.isin(edges)]

    def subgraph(self, nodes=None, edges=None):
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
        subgraph._edges = self.ilter_edges(edges)
        return subgraph

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


class SparkPGFrame(PGFrame):
    """Class for storing typed PGs as a collection of Spark DataFrames."""

    def __init__(self):
        """Initalize a SparkPGFrame."""
        pass


class GraphProcessor(ABC):

    def __init__(self, pgframe):
        self.graph = self._generate_graph(pgframe)

    @staticmethod
    @abstractmethod
    def _generate_graph(pgframe):
        pass

    @abstractmethod
    def _generate_pgframe(self, node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        pass

    @classmethod
    def from_graph_object(cls, graph_object):
        processor = cls()
        processor.graph = graph_object
        return processor

    def get_pgframe(self, node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        return self._generate_pgframe(
            node_filter=node_filter, edge_filter=edge_filter)

    class ProcessorException(BlueGraphException):
        pass
