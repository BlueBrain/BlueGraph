from bluegraph.core.analyse.paths import PathFinder
from bluegraph.core.utils import top_n

from bluegraph.exceptions import PathSearchException

from ..io import pgframe_to_networkx

import networkx as nx


def handle_exclude_nx_edge(method):
    """Method decorator that removes and restores the direct s/t edge."""
    def wrapper(graph, source, target, **kwargs):
        exclude_edge = False
        if "exclude_edge" in kwargs:
            exclude_edge = kwargs["exclude_edge"]

        backup_edge = None

        if exclude_edge and (source, target) in graph.edges():
            backup_edge = {
                **graph.edges[source, target]
            }
            graph.remove_edge(source, target)

        result = method(graph, source, target, **kwargs)

        if backup_edge is not None:
            graph.add_edge(source, target, **backup_edge)
        return result

    return wrapper


class NXPathFinder(PathFinder):
    """NetworkX-based shortest paths finder."""

    @staticmethod
    def _generate_graph(pgframe):
        """Generate the appropiate graph representation from a PGFrame."""
        return pgframe_to_networkx(pgframe)

    @staticmethod
    def _get_distance(graph, source, target, distance):
        """Get distance value between source and target."""
        return graph.edges[source, target][distance]

    @staticmethod
    def _get_neighbors(graph, node_id):
        """Get neighors of the node."""
        return list(graph.neighbors(node_id))

    def _get_subgraph(self, nodes_to_include):
        """Produce a graph induced by the input nodes."""
        return self.graph.subgraph(nodes_to_include)

    @staticmethod
    @handle_exclude_nx_edge
    def _compute_shortest_path(graph, s, t, distance=None, exclude_edge=False):
        path = tuple(nx.shortest_path(graph, s, t, weight=distance))
        return path

    @staticmethod
    @handle_exclude_nx_edge
    def _compute_all_shortest_paths(graph, s, t, exclude_edge=False):
        all_paths = [tuple(p) for p in nx.all_shortest_paths(
            graph, s, t)]
        return all_paths

    @staticmethod
    @handle_exclude_nx_edge
    def _compute_yen_shortest_paths(graph, source, target, n=None,
                                    distance=None, exclude_edge=False):
        if n is None:
            raise PathSearchException(
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
                raise PathSearchException(
                    "The minimum spanning tree finder has the write option set "
                    "to True, the write property name must be specified")

            mst_property = {}
            for e in self.graph.edges():
                mst_property[e] = e in tree.edges()
            nx.set_edge_attributes(self.graph, mst_property, write_property)
        else:
            return tree
