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
import math
import pandas as pd
import graph_tool as gt
from graph_tool import GraphView
from graph_tool.spectral import adjacency
from graph_tool.util import find_vertex

from bluegraph.core.io import PandasPGFrame, GraphProcessor


NUMERIC_TYPES = [
    "int16_t", "int32_t", "int64_t", "double", "long double"
]


def _get_vertex_obj(graph, node_id):
    for v in graph.iter_vertices(vprops=[graph.vp["@id"]]):
        if v[1] == node_id:
            return v[0]


def _get_edge_obj(graph, source_id, target_id):
    source = _get_vertex_obj(graph, source_id)
    target = _get_vertex_obj(graph, target_id)
    e = graph.edge(source, target)
    if e is None and not graph.is_directed():
        e = graph.edge(graph.vertex(target), graph.vertex(source))
    return e


def _get_node_id(graph, vertex_obj):
    return graph.vp["@id"][vertex_obj]


def _infer_type(value):
    prop_type = "string"
    if isinstance(value, float):
        prop_type = "double"
    elif isinstance(value, int):
        prop_type = "int32_t"
    elif isinstance(value, bool):
        prop_type = "bool"
    return prop_type


def pgframe_to_graph_tool(pgframe, directed=True):
    """Create a graph-tool graph from the PGFrame."""
    graph = gt.Graph(directed=directed)

    # Add nodes
    graph.add_vertex(pgframe.number_of_nodes())

    # Encode original node id's as a new property `@label`
    prop = graph.new_vertex_property(
        "string", vals=pgframe._nodes.index)
    graph.vertex_properties["@id"] = prop

    # Add node properties
    for c in pgframe._nodes.columns:
        prop_type = "string"
        if c in pgframe._node_prop_types and\
           pgframe._node_prop_types[c] == "numeric":
            prop_type = "double"

        prop = graph.new_vertex_property(
            prop_type, vals=pgframe._nodes[c])
        graph.vertex_properties[c] = prop

    # Add edges and edge properties
    new_props = []
    for c in pgframe._edges.columns:
        prop_type = "string"
        if c in pgframe._edge_prop_types and\
           pgframe._edge_prop_types[c] == "numeric":
            prop_type = "double"
        props = graph.new_edge_property(prop_type)
        graph.edge_properties[c] = props
        new_props.append(props)

    gt_edges = pgframe._edges.reset_index()
    gt_edges["@source_id"] = gt_edges["@source_id"].apply(
        lambda x: pgframe._nodes.index.get_loc(x))
    gt_edges["@target_id"] = gt_edges["@target_id"].apply(
        lambda x: pgframe._nodes.index.get_loc(x))
    graph.add_edge_list(gt_edges.values.tolist(), eprops=new_props)
    return graph


def graph_tool_to_pgframe(graph, node_prop_types=None, edge_prop_types=None):
    """Create a PGFrame from the graph-tool object."""
    pgframe = PandasPGFrame(nodes=graph.vp["@id"])
    for k, v in graph.vp.items():
        if k != "@id":
            prop = pd.DataFrame(list(v), index=graph.vp["@id"], columns=[k])
            if k == "@type":
                pgframe.assign_node_types(prop)
            else:
                prop_type = v.value_type()
                if node_prop_types is not None:
                    result_type = node_prop_types[k]
                else:
                    result_type = "category"
                    if prop_type in NUMERIC_TYPES:
                        result_type = "numeric"
                pgframe.add_node_properties(prop)
                pgframe._set_node_prop_type(k, result_type)

    edges = [
        (graph.vp["@id"][e.source()], graph.vp["@id"][e.target()])
        for e in graph.edges()]
    pgframe.add_edges(edges)

    for k, v in graph.ep.items():
        prop = pd.DataFrame(
            [
                [edges[i][0], edges[i][1], el]
                for i, el in enumerate(list(v))
            ],
            columns=["@source_id", "@target_id", k]
        )
        prop.set_index(["@source_id", "@target_id"])
        if k == "@type":
            pgframe.assign_edge_types(prop)
        else:
            prop_type = v.value_type()
            if edge_prop_types is not None:
                result_type = edge_prop_types[k]
            else:
                result_type = "category"
                if prop_type in NUMERIC_TYPES:
                    result_type = "numeric"
            pgframe.add_edge_properties(prop)
            pgframe._set_edge_prop_type(k, result_type)

    return pgframe


class GTGraphProcessor(GraphProcessor):
    """graph-tool graph processor.

    The provided interface allows to convert graph-tool objects
    into PGFrames and vice versa.
    """

    @staticmethod
    def _generate_graph(pgframe, directed=True):
        return pgframe_to_graph_tool(pgframe, directed=directed)

    def _generate_pgframe(self, node_prop_types=None, edge_prop_types=None,
                          node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        return graph_tool_to_pgframe(
            self.graph, node_prop_types=node_prop_types,
            edge_prop_types=edge_prop_types)

    @staticmethod
    def _is_directed(graph):
        return graph.is_directed()

    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        if isinstance(new_property, gt.VertexPropertyMap):
            new_property = new_property.a
        return dict(
            zip(list(self.graph.vertex_properties["@id"]), new_property))

    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
        self.graph.vertex_properties[property_name] = new_property

    def _get_adjacency_matrix(self, nodes, weight=None):
        return adjacency(self.graph, weight=weight)

    def _get_node_property_values(self, prop, nodes):
        return self.graph.vp[prop]

    def nodes(self, properties=False):
        if not properties:
            return list(self.graph.vp["@id"])
        else:
            props = list(self.graph.vp.keys())
            result = []
            for v in self.graph.iter_vertices(vprops=[
                    self.graph.vp[p] for p in props]):
                raw_props = dict(zip(props, v[1:]))
                node_id = raw_props["@id"]
                del raw_props["@id"]
                result.append((node_id, raw_props))
            return result

    def get_node(self, node):
        v = _get_vertex_obj(self.graph, node)
        props = {}
        for k in self.graph.vertex_properties.keys():
            if k not in ["@id", "@type"]:
                value = self.graph.vertex_properties[k][v]
                if isinstance(value, float):
                    if not math.isnan(value):
                        props[k] = value
                else:
                    props[k] = value
        return props

    def remove_node(self, node):
        v = _get_vertex_obj(self.graph, node)
        self.graph.remove_vertex(v)

    def rename_nodes(self, node_mapping):
        for old_node, new_node in node_mapping.items():
            v = _get_vertex_obj(self.graph, old_node)
            self.graph.vertex_properties["@id"][v] = new_node

    def set_node_properties(self, node, properties):
        vertex = _get_vertex_obj(self.graph, node)
        for k, v in properties.items():
            if k not in self.graph.vertex_properties.keys():
                vp = self.graph.new_vertex_property(_infer_type(v))
                self.graph.vertex_properties[k] = vp
            if v is not None:
                self.graph.vertex_properties[k][vertex] = v
            else:
                self.graph.vertex_properties[k][vertex] = 'nan'
        for k in self.graph.vertex_properties.keys():
            if k not in properties and k != "@id":
                self.graph.vertex_properties[k][vertex] = 'nan'

    def edges(self, properties=False):
        if not properties:
            return [
                (self.graph.vp["@id"][s], self.graph.vp["@id"][t])
                for s, t in self.graph.iter_edges()
            ]
        else:
            props = list(self.graph.ep.keys())
            result = []
            for edge in self.graph.iter_edges(eprops=[
                    self.graph.ep[p] for p in props]):
                s = self.graph.vp["@id"][edge[0]]
                t = self.graph.vp["@id"][edge[1]]
                raw_props = dict(zip(props, edge[2:]))
                result.append((s, t, raw_props))
            return result

    def get_edge(self, source, target):
        e = _get_edge_obj(self.graph, source, target)
        if e:
            props = {}
            for k in self.graph.edge_properties.keys():
                if k not in ["@type"]:
                    value = self.graph.edge_properties[k][e]
                    if isinstance(value, float):
                        if not math.isnan(value):
                            props[k] = value
                    else:
                        props[k] = value
            return props

    def remove_edge(self, source, target):
        e = _get_edge_obj(self.graph, source, target)
        self.graph.remove_edge(e)

    def add_edge(self, source, target, properties=None):
        s = _get_vertex_obj(self.graph, source)
        t = _get_vertex_obj(self.graph, target)
        e = self.graph.add_edge(s, t)
        if properties is not None:
            for k, v in properties.items():
                if k not in self.graph.edge_properties.keys():
                    ep = self.graph.new_edge_property(_infer_type(v))
                    self.graph.edge_properties[k] = ep
                self.graph.edge_properties[k][e] = v

    def set_edge_properties(self, source, target, properties):
        edge = _get_edge_obj(self.graph, source, target)
        for k, v in properties.items():
            if k not in self.graph.edge_properties.keys():
                vp = self.graph.new_edge_property(_infer_type(v))
                self.graph.edge_properties[k] = vp
            if v is not None:
                self.graph.edge_properties[k][edge] = v
            else:
                self.graph.edge_properties[k][edge] = math.nan
        for k in self.graph.edge_properties.keys():
            if k not in properties:
                self.graph.edge_properties[k][edge] = math.nan

    def neighbors(self, node_id):
        """Get neighors of the node."""
        node_id = _get_vertex_obj(self.graph, node_id)
        neighors = self.graph.vertex(node_id).out_neighbors()
        return [
            _get_node_id(self.graph, n) for n in neighors
        ]

    def subgraph(self, nodes_to_include=None, edges_to_include=None,
                 nodes_to_exclude=None, edges_to_exclude=None):
        """Produce a graph induced by the input nodes."""
        if nodes_to_include is None:
            node_filter_prop = self.graph.new_vertex_property(
                "bool", val=True)
            if nodes_to_exclude is not None:
                indices_to_exclude = [
                    _get_vertex_obj(self.graph, n)
                    for n in nodes_to_exclude
                ]
                for i in indices_to_exclude:
                    node_filter_prop[i] = False
        else:
            node_filter_prop = self.graph.new_vertex_property(
                "bool", val=False)
            indices_to_include = [
                _get_vertex_obj(self.graph, n)
                for n in nodes_to_include
            ]
            for i in indices_to_include:
                node_filter_prop[i] = True

        if edges_to_include is None:
            edge_filter_prop = self.graph.new_edge_property(
                "bool", val=True)
            if edges_to_exclude is not None:
                sources = [
                    _get_vertex_obj(self.graph, s)
                    for s, _ in edges_to_exclude
                ]
                targets = [
                    _get_vertex_obj(self.graph, t)
                    for _, t in edges_to_exclude
                ]
                for s, t in zip(sources, targets):
                    edge_filter_prop[self.graph.edge(s, t)] = False
        else:
            edge_filter_prop = self.graph.new_edge_property(
                "bool", val=False)
            sources = [
                _get_vertex_obj(self.graph, s)
                for s, _ in edges_to_include
            ]
            targets = [
                _get_vertex_obj(self.graph, t)
                for _, t in edges_to_include
            ]
            for s, t in zip(sources, targets):
                edge_filter_prop[self.graph.edge(s, t)] = True

        return GraphView(
            self.graph,
            vfilt=node_filter_prop,
            efilt=edge_filter_prop)

