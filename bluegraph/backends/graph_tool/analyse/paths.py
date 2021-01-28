from bluegraph.core.analyse.paths import PathFinder

from graph_tool.topology import shortest_path
from graph_tool.topology import all_shortest_paths as gt_all_shortest_paths
from graph_tool.util import find_vertex

from ..io import pgframe_to_graph_tool


def handle_exclude_gt_edge(method):
    """Method decorator that removes and restores the direct s/t edge."""
    def wrapper(finder, source, target, **kwargs):
        exclude_edge = False
        if "exclude_edge" in kwargs:
            exclude_edge = kwargs["exclude_edge"]

        source_vertex = find_vertex(
            finder.graph, finder.graph.vp["@id"], source)[0]
        target_vertex = find_vertex(
            finder.graph, finder.graph.vp["@id"], target)[0]

        direct_edge = finder.graph.edge(source_vertex, target_vertex)
        edge_filter = finder.graph.new_edge_property("bool", val=True)

        if direct_edge and exclude_edge is True:
            edge_filter[direct_edge] = False
            finder.graph.set_edge_filter(edge_filter)

        result = method(finder, source, target, **kwargs)

        finder.graph.clear_filters()
        return result

    return wrapper


class GTPathFinder(PathFinder):
    """graph-tool-based shortest paths finder."""

    def _get_vertex_obj(self, node_id):
        v = find_vertex(
            self.graph, self.graph.vp["@id"], node_id)
        if len(v) == 1:
            return v[0]

    def _get_node_id(self, vertex_obj):
        return self.graph.vp["@id"][vertex_obj]

    @staticmethod
    def _generate_graph(pgframe):
        """Generate the appropiate graph representation from a PGFrame."""
        return pgframe_to_graph_tool(pgframe)

    def _get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        source = self._get_vertex_obj(source)
        target = self._get_vertex_obj(target)

        edge = self.graph.edge(source, target)
        return self.graph.ep[distance][edge]

    def _get_neighbors(self, node_id):
        """Get neighors of the node."""
        node_obj = self._get_vertex_obj(node_id)
        neighors = node_obj.out_neighbors()
        return [
            self._get_node_id(n) for n in neighors
        ]

    @handle_exclude_gt_edge
    def _compute_shortest_path(self, source, target, distance=None,
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
        source_vertex = self._get_vertex_obj(source)
        target_vertex = self._get_vertex_obj(target)

        path, _ = shortest_path(
            self.graph,
            source_vertex, target_vertex,
            weights=self.graph.edge_properties[distance] if distance else None)

        return tuple([
            self.graph.vp["@id"][el] for el in path
        ])

    @handle_exclude_gt_edge
    def _compute_all_shortest_paths(self, source, target, exclude_edge=False):
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
        source_vertex = self._get_vertex_obj(source)
        target_vertex = self._get_vertex_obj(target)

        paths = gt_all_shortest_paths(self.graph, source_vertex, target_vertex)

        return [
            tuple([
                self.graph.vp["@id"][el]
                for el in path
            ]) for path in paths
        ]

    def _compute_yen_shortest_paths(self, source, target, n,
                                    distance, exclude_edge=False):
        """Compute n shortest paths using the Yen's algo."""
        raise NotImplementedError(
            "Yen's algorithm for finding n shortest paths "
            "is currently not implemented")

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
            raise NotImplementedError(
                "Yen's algorithm for finding n shortest paths "
                "is currently not implemented")
        else:
            return super().n_shortest_paths(
                source, target, n, distance=distance,
                strategy="naive", exclude_edge=exclude_edge)
