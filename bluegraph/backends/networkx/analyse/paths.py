from bluegraph.core.analyse.paths import PathFinder
from bluegraph.core.utils import top_n

from bluegraph.exceptions import PathSearchException

from ..io import pgframe_to_networkx

import networkx as nx


def handle_exclude_nx_edge(method):
    """Method decorator that removes and restores the direct s/t edge."""
    def wrapper(finder, source, target, **kwargs):
        exclude_edge = False
        if "exclude_edge" in kwargs:
            exclude_edge = kwargs["exclude_edge"]

        backup_edge = None

        if exclude_edge and (source, target) in finder.graph.edges():
            backup_edge = {
                **finder.graph.edges[source, target]
            }
            finder.graph.remove_edge(source, target)

        result = method(finder, source, target, **kwargs)

        if backup_edge is not None:
            finder.graph.add_edge(source, target, **backup_edge)
        return result

    return wrapper


class NXPathFinder(PathFinder):
    """NetworkX-based shortest paths finder."""

    @staticmethod
    def _generate_graph(pgframe):
        """Generate the appropiate graph representation from a PGFrame."""
        return pgframe_to_networkx(pgframe)

    def _get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        return self.graph.edges[source, target][distance]

    def _get_neighbors(self, node_id):
        """Get neighors of the node."""
        return list(self.graph.neighbors(node_id))

    @handle_exclude_nx_edge
    def _compute_shortest_path(self, s, t, distance=None, exclude_edge=False):
        path = tuple(nx.shortest_path(self.graph, s, t, weight=distance))
        return path

    @handle_exclude_nx_edge
    def _compute_all_shortest_paths(self, s, t, exclude_edge=False):
        all_paths = [tuple(p) for p in nx.all_shortest_paths(
            self.graph, s, t)]
        return all_paths

    @handle_exclude_nx_edge
    def _compute_yen_shortest_paths(self, source, target, n=None,
                                    distance=None, exclude_edge=False):
        if n is None:
            raise PathSearchException(
                "Number of paths must be specified when calling"
                "`NXPathFinder.compute_yen_shortest_paths`")
        generator = nx.shortest_simple_paths(
            self.graph, source, target, weight=distance)
        i = 0
        paths = []
        for path in generator:
            paths.append(tuple(path))
            i += 1
            if i == n:
                break
        return paths

    def minimum_spanning_tree(self, weight):
        """Compute the minimum spanning tree."""
        pass
        return nx.minimum_spanning_tree(self.graph, weight=weight)
