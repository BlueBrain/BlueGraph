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
from bluegraph.core.analyse.paths import PathFinder, graph_elements_from_paths

from ..io import NXGraphProcessor

import networkx as nx


def handle_exclude_nx_edge(method):
    """Method decorator that removes and restores the direct s/t edge."""
    def wrapper(graph, source, target, **kwargs):
        exclude_edge = False
        if "exclude_edge" in kwargs:
            exclude_edge = kwargs["exclude_edge"]

        subgraph = graph
        if exclude_edge:
            if nx.is_directed(graph):
                if (source, target) in graph.edges():
                    subgraph = subgraph.edge_subgraph(
                        [e for e in subgraph.edges() if e != (source, target)]
                    )
            else:
                if (source, target) in graph.edges() or\
                   (target, source) in graph.edges():
                    subgraph = subgraph.edge_subgraph([
                        e for e in subgraph.edges()
                        if e != (source, target) and e != (target, source)
                    ])

        result = method(subgraph, source, target, **kwargs)
        return result

    return wrapper


class NXPathFinder(NXGraphProcessor, PathFinder):
    """NetworkX-based shortest paths finder."""

    def get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        return self.graph.edges[source, target][distance]

    def get_subgraph_from_paths(self, paths):
        """Get a subgraph given the input paths."""
        nodes, edges = graph_elements_from_paths(paths)
        subgraph = self.graph.subgraph(nodes).edge_subgraph(edges)
        return subgraph

    @staticmethod
    @handle_exclude_nx_edge
    def _compute_shortest_path(graph, s, t, distance=None, exclude_edge=False):
        path = tuple(nx.shortest_path(graph, s, t, weight=distance))
        return path

    @staticmethod
    @handle_exclude_nx_edge
    def _compute_all_shortest_paths(graph, s, t, exclude_edge=False):
        try:
            all_paths = [tuple(p) for p in nx.all_shortest_paths(
                graph, s, t)]
        except nx.exception.NetworkXNoPath:
            raise PathFinder.NoPathException(
                "Path from '{}' to '{}' does not exist".format(s, t))
        return all_paths

    @staticmethod
    @handle_exclude_nx_edge
    def _compute_yen_shortest_paths(graph, source, target, n=None,
                                    distance=None, exclude_edge=False):
        if n is None:
            raise PathFinder.PathSearchException(
                "Number of paths must be specified when calling"
                "`NXPathFinder.compute_yen_shortest_paths`")
        generator = nx.shortest_simple_paths(
            graph, source, target, weight=distance)
        i = 0
        paths = []
        for path in generator:
            paths.append(tuple(path))
            i += 1
            if i == n:
                break
        return paths

    def minimum_spanning_tree(self, distance, write=False, write_property=None):
        """Compute the minimum spanning tree.

        Parameters
        ----------
        distance : str
            Distance to minimize when computing the minimum spanning tree (MST)
        write : bool, optional
            Flag indicating whether the MST should be returned as a new graph
            object or saved within a Boolean edge property being True whenever
            a given edge belongs to the MST.
        write_property : str, optional
            Edge property name for marking edges beloning to the MST.

        Returns
        -------
        tree : nx.Graph
            The minimum spanning tree graph object
        """
        tree = nx.minimum_spanning_tree(self.graph, weight=distance)
        if write:
            if write_property is None:
                raise PathFinder.PathSearchException(
                    "The minimum spanning tree finder has the write option set "
                    "to True, the write property name must be specified")

            mst_property = {}
            for e in self.graph.edges():
                mst_property[e] = e in tree.edges()
            nx.set_edge_attributes(self.graph, mst_property, write_property)
        else:
            return tree
