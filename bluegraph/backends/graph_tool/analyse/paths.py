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

from graph_tool import GraphView
from graph_tool.topology import shortest_path, min_spanning_tree
from graph_tool.topology import all_shortest_paths as gt_all_shortest_paths

from ..io import GTGraphProcessor, _get_vertex_obj, _get_node_id, _get_edge_obj


def handle_exclude_gt_edge(method):
    """Method decorator that removes and restores the direct s/t edge."""
    def wrapper(graph, source, target, **kwargs):
        exclude_edge = False
        if "exclude_edge" in kwargs:
            exclude_edge = kwargs["exclude_edge"]

        direct_edge = None

        direct_edge = _get_edge_obj(graph, source, target)
        edge_filter = graph.new_edge_property("bool", val=True)

        if direct_edge and exclude_edge is True:
            edge_filter[direct_edge] = False
            graph.set_edge_filter(edge_filter)

        result = method(graph, source, target, **kwargs)

        if direct_edge:
            graph.clear_filters()
        return result

    return wrapper


class GTPathFinder(GTGraphProcessor, PathFinder):
    """graph-tool-based shortest paths finder."""

    def get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        edge = _get_edge_obj(self.graph, source, target)
        return self.graph.ep[distance][edge]

    def get_subgraph_from_paths(self, paths):
        """Get subgraph induced by a path."""
        nodes, edges = graph_elements_from_paths(paths)
        return self.subgraph(
            nodes_to_include=nodes, edges_to_include=edges)

    @staticmethod
    @handle_exclude_gt_edge
    def _compute_shortest_path(graph, source, target, distance=None,
                               exclude_edge=False):
        """Compute the single shortest path from the source to the target.

        Parameters
        ----------
        source : str
            Source node ID
        target : str
            Target node ID
        exclude_edge : bool, optional
            Flag indicating whether the direct edge from the source to
            the target should be excluded from the result (if exists).
        """
        source_vertex = graph.vertex(_get_vertex_obj(graph, source))
        target_vertex = graph.vertex(_get_vertex_obj(graph, target))

        path, _ = shortest_path(
            graph,
            source_vertex, target_vertex,
            weights=graph.edge_properties[distance] if distance else None)

        return tuple([
            graph.vp["@id"][el] for el in path
        ])

    @staticmethod
    @handle_exclude_gt_edge
    def _compute_all_shortest_paths(graph, source, target, exclude_edge=False):
        """Compute all shortest paths from the source to the target.

        This function computes all the shortest (unweighted) paths
        from the source to the target.

        Parameters
        ----------
        source : str
            Source node ID
        target : str
            Target node ID
        exclude_edge : bool, optional
            Flag indicating whether the direct edge from the source to
            the target should be excluded from the result (if exists).
        """
        source_vertex = _get_vertex_obj(graph, source)
        target_vertex = _get_vertex_obj(graph, target)

        paths = gt_all_shortest_paths(graph, source_vertex, target_vertex)

        return [
            tuple([
                graph.vp["@id"][el]
                for el in path
            ]) for path in paths
        ]

    @staticmethod
    def _compute_yen_shortest_paths(graph, source, target, n,
                                    distance, exclude_edge=False):
        """Compute n shortest paths using the Yen's algo."""
        raise PathFinder.NotImplementedError(
            "Yen's algorithm for finding n shortest paths "
            "is currently not implemented for graph-tool backend")

    def n_shortest_paths(self, source, target, n, distance=None,
                         strategy="naive", exclude_edge=False):
        """Compute n shortest paths from the source to the target.

        Two search strategies are available: 'naive' and 'yen'.
        The naive strategy first finds the set of all shortest paths from the
        source to the target node, it then ranks them by the cumulative distance
        score and returns n best paths. The second strategy uses Yen's
        algorithm [1] for finding n shortest paths. The first naive strategy
        performs better for highly dense graphs (where every node is connected to
        almost every other node). Note that if there are less than n unweighted
        shortest paths in the graph, the naive strategy may return less than n
        paths.


        1. Yen, Jin Y. "Finding the k shortest loopless paths in a network".
        Management Science 17.11 (1971): 712-716.

        Parameters
        ----------
        source : str
            Source node ID
        target : str
            Target node ID
        n : int
            Number of top paths to include in the result
        distance : str, optional
            The name of the attribute to use as the edge distance
        path_condition : func, optional
            Edge filtering function returning Boolean flag
        strategy : str, optional
            Path finding strategy: `naive` or `yen`. By default, `naive`.
        exclude_edge : bool, optional
            Flag indicating whether the direct edge from the source to
            the target should be excluded from the result (if exists).
        Returns
        -------
        paths : list
            List containing top n best paths according to the distance score
        """
        if strategy == "yen":
            raise PathFinder.NotImplementedError(
                "Yen's algorithm for finding n shortest paths "
                "is currently not implemented")
        else:
            return super().n_shortest_paths(
                source, target, n, distance=distance,
                strategy="naive", exclude_edge=exclude_edge)

    def minimum_spanning_tree(self, distance, write=False,
                              write_property=None):
        """Compute the minimum spanning tree.

        Parameters
        ----------
        distance : str
            Distance to minimize when computing the minimum spanning tree
            (MST)
        write : bool, optional
            Flag indicating whether the MST should be returned as
            a new graph
            object or saved within a Boolean edge property being True whenever
            a given edge belongs to the MST.
        write_property : str, optional
            Edge property name for marking edges beloning to the MST.

        Returns
        -------
        tree : nx.Graph
            The minimum spanning tree graph object
        """
        mst_property = min_spanning_tree(
            self.graph, weights=self.graph.ep[distance])
        if write:
            if write_property is None:
                raise PathFinder.PathSearchException(
                    "The minimum spanning tree finder has the write option set "
                    "to True, the write property name must be specified")

            self.graph.ep[write_property] = mst_property
        else:
            tree = GraphView(
                self.graph,
                efilt=mst_property)
            return tree
