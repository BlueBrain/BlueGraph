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
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix

import pandas as pd

from bluegraph.core.io import GraphProcessor, PandasPGFrame


def pgframe_to_networkx(pgframe, directed=True):
    """Create a NetworkX graph from the PGFrame."""
    if directed:
        create_using = nx.DiGraph
    else:
        create_using = nx.Graph
    edge_attr = True if len(pgframe.edge_properties()) > 0 else None
    graph = nx.from_pandas_edgelist(
        pgframe._edges.reset_index(),
        source="@source_id", target="@target_id",
        edge_attr=edge_attr, create_using=create_using)
    nx.set_node_attributes(graph, pgframe._nodes.to_dict("index"))
    return graph


def networkx_to_pgframe(nx_object, node_prop_types=None, edge_prop_types=None):
    """Create a PGFrame from the networkx object."""
    pgframe = PandasPGFrame(nodes=nx_object.nodes())

    aggregated_props = {}
    for n, d in nx_object.nodes.items():
        for k in d:
            if k not in aggregated_props:
                aggregated_props[k] = {}
            aggregated_props[k][n] = d[k]

    for k in aggregated_props:
        pgframe.add_node_properties(
            pd.DataFrame(
                aggregated_props[k].items(), columns=["@id", k]))

    edges = nx.to_pandas_edgelist(nx_object).rename(
        columns={
            "source": "@source_id", "target": "@target_id"
        }).set_index(["@source_id", "@target_id"])
    pgframe._edges = edges
    if node_prop_types:
        pgframe._node_prop_types = node_prop_types.copy()
    if edge_prop_types:
        pgframe._edge_prop_types = edge_prop_types.copy()
    return pgframe


class NXGraphProcessor(GraphProcessor):
    """NetworkX graph processor.

    The provided interface allows to convert NetworkX objects
    into PGFrames and vice versa.
    """

    @staticmethod
    def _generate_graph(pgframe, directed=True):
        return pgframe_to_networkx(pgframe, directed=directed)

    def _generate_pgframe(self, node_prop_types=None, edge_prop_types=None,
                          node_filter=None, edge_filter=None,):
        """Get a new pgframe object from the wrapped graph object."""
        return networkx_to_pgframe(
            self.graph, node_prop_types=node_prop_types,
            edge_prop_types=edge_prop_types)

    @staticmethod
    def _is_directed(graph):
        return graph.is_directed()

    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        return new_property

    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
        nx.set_node_attributes(
            self.graph, new_property, property_name)

    def _get_adjacency_matrix(self, nodes, weight=None):
        return adjacency_matrix(self.graph, nodelist=nodes, weight=weight)

    def _get_node_property_values(self, prop, nodes):
        attrs = nx.get_node_attributes(self.graph, prop)
        props = [attrs[n] for n in nodes]
        return props

    def nodes(self, properties=False):
        return list(self.graph.nodes(data=properties))

    def get_node(self, node):
        return self.graph.nodes[node]

    def remove_node(self, node):
        self.graph.remove_node(node)

    def rename_nodes(self, node_mapping):
        nx.relabel_nodes(self.graph, node_mapping, copy=False)

    def set_node_properties(self, node, properties):
        self.graph.nodes[node].clear()
        self.graph.nodes[node].update(properties)

    def edges(self, properties=False):
        return list(self.graph.edges(data=properties))

    def get_edge(self, source, target):
        return self.graph.edges[source, target]

    def remove_edge(self, source, target):
        self.graph.remove_edge(source, target)

    def add_edge(self, source, target, properties=None):
        if properties is None:
            properties = {}
        self.graph.add_edge(source, target, **properties)

    def set_edge_properties(self, source, target, properties):
        self.graph.edges[source, target].clear()
        self.graph.edges[source, target].update(properties)

    def neighbors(self, node_id):
        """Get neighors of the node."""
        return list(self.graph.neighbors(node_id))

    def subgraph(self, nodes_to_include=None, edges_to_include=None,
                 nodes_to_exclude=None, edges_to_exclude=None):
        """Produce a graph induced by the input nodes."""
        if nodes_to_include is not None:
            subgraph = self.graph.subgraph(nodes_to_include)
        else:
            if nodes_to_exclude is not None:
                subgraph = self.graph.subgraph([
                    n for n in self.graph.nodes()
                    if n not in nodes_to_exclude
                ])
            else:
                subgraph = self.graph

        if edges_to_include is not None:
            subgraph = subgraph.edge_subgraph(
                edges_to_include)
        else:
            if edges_to_exclude is not None:
                subgraph = subgraph.edge_subgraph(
                    [
                        e for e in subgraph.edges()
                        if e not in edges_to_exclude
                    ]
                )

        return subgraph
