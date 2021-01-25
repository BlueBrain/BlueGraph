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
    @handle_exclude_nx_edge
    def compute_shortest_path(graph, s, t, distance=None, exclude_edge=False):
        path = tuple(nx.shortest_path(
                graph, s, t, weight=distance))
        return path

    @staticmethod
    @handle_exclude_nx_edge
    def compute_all_shortest_paths(graph, s, t, exclude_edge=False):
        all_paths = [tuple(p) for p in nx.all_shortest_paths(
            graph, s, t)]
        return all_paths

    @staticmethod
    def get_cumulative_distances(graph, paths, distance):
        """Get cumulative distance scores for provided paths."""
        def _sumup_distances(path):
            result = 0
            for i in range(1, len(path)):
                source = path[i - 1]
                target = path[i]
                if distance is not None:
                    result += graph.edges[
                        source, target][distance]
                else:
                    result += 1
            return result

        path_ranking = {
            p: _sumup_distances(p) for p in paths
        }

        return path_ranking

    @staticmethod
    @handle_exclude_nx_edge
    def compute_yen_shortest_paths(graph, source, target, n=None,
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

    @staticmethod
    def compute_n_shortest_paths(graph, source, target, n, distance=None,
                                 strategy="naive", exclude_edge=False):
        if n == 1:
            return [
                tuple(
                    NXPathFinder.compute_shortest_path(
                        graph, source, target, distance))
            ]

        if strategy == "naive":
            all_paths = NXPathFinder.compute_all_shortest_paths(
                graph, source, target, exclude_edge=True)
            path_ranking = NXPathFinder.get_cumulative_distances(
                graph, all_paths, distance)

            if not exclude_edge:
                path_ranking[(source, target)] = (
                    graph.edges[source, target][distance]
                    if distance else 1
                )
            paths = top_n(path_ranking, n, smallest=True)
        elif strategy == "yen":
            paths = NXPathFinder.compute_yen_shortest_paths(
                graph, source, target, n=n,
                distance=distance, exclude_edge=exclude_edge)
        else:
            PathSearchException(
                f"Unknown path search strategy '{strategy}'")
        return paths

    def shortest_path(self, source, target, distance=None, exclude_edge=False):
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
        return self.compute_shortest_path(
            self.graph, source, target, distance=distance,
            exclude_edge=exclude_edge)

    def all_shortest_paths(self, source, target, exclude_edge=False):
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
        return self.compute_all_shortest_paths(
            self.graph, source, target, exclude_edge=exclude_edge)

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
        return self.compute_n_shortest_paths(
            self.graph, source, target, n, distance=distance,
            strategy=strategy, exclude_edge=exclude_edge)

    def shortest_tripath(self, source, intermediary, target,
                         distance=None, exclude_edge=False, overlap=True):
        """Compute the shortest tri-path from the source to the target.

        The shortest tripath is given by two paths: the shortest path from
        the source to the intermediary and from the intermediary to the target.
        These sets can be overlapping or not. If the sets
        are non-overlapping, all the nodes encountered on the paths from A to B
        are excluded from the search of paths from B to C.

        Parameters
        ----------
        source : str
            Source node ID
        intermediary : str
            Intermediate node ID
        target : str
            Target node ID
        distance : str, optional
            The name of the attribute to use as the edge distance
        exclude_edge : bool, optional
            Flag indicating whether the direct edge from the source to
            the target should be excluded from the result (if exists).
        overlap : bool, optional.
            Flag indicating whether the two paths are allowed to
            intersect (to pass through the same nodes). By default True.

        Returns
        -------
        a_b_path : tuple
            The shortest path from A to B
        b_c_path : tuple
            The shortest path from B to C
        """
        raise PathSearchException(
            "Tripath search is currently not implemented")
        # a_b_path = self.shortest_path(
        #     source, intermediary, distance=distance, exclude_edge=exclude_edge)

        # if overlap is False:
        #     search_nodes = [
        #         n for n in self.graph.nodes()
        #         if n not in list(a_b_path)[1:-1]
        #     ]
        #     graph = self.graph.subgraph(search_nodes)
        # else:
        #     graph = self.graph

        # b_c_path = self.compute_shortest_path(
        #     graph, intermediary, target,
        #     distance=distance,
        #     exclude_edge=exclude_edge)

        # return a_b_path, b_c_path

    def n_shortest_tripaths(self, source, intermediary, target,
                            n, distance=None, strategy="naive",
                            exclude_edge=False, overlap=True):
        """Compute n shortest tri-paths from the source to the target.

        Tripaths cosist of two path sets, from the source (A) to the
        intermediary (B) and from the intermediary to the target (C).
        These sets can be overlapping or not. If the sets
        are non-overlapping, all the nodes encountered on the paths from A to B
        are excluded from the search of paths from B to C.

        Parameters
        ----------
        source : str
            Source node ID
        intermediary : str
            Intermediate node ID
        target : str
            Target node ID
        n : int
            Number of best paths to search for
        distance : str, optional
            The name of the attribute to use as the edge distance
        exclude_edge : bool, optional
            Flag indicating whether the direct edge from the source to
            the target should be excluded from the result (if exists).
        overlap : bool, optional.
            Flag indicating whether the two paths are allowed to
            intersect (to pass through the same nodes). By default True.

        Returns
        -------
        a_b_paths : list
            List containing the shortest paths from A to B
        b_c_paths : list
            List containing the shortest paths from B to C
        """
        raise PathSearchException(
            "Tripath search is currently not implemented")
        # a_b_paths = self.n_shortest_paths(
        #     source, intermediary, n, distance=distance,
        #     strategy=strategy, exclude_edge=exclude_edge)

        # if overlap is False:
        #     visited_nodes = set()
        #     for p in a_b_paths:
        #         visited_nodes.update(list(p)[-1:1])
        #     search_nodes = [
        #         n for n in self.graph.nodes()
        #         if n not in visited_nodes
        #     ]
        #     graph = self.graph.subgraph(search_nodes)
        # else:
        #     graph = self.graph

        # b_c_paths = self.compute_n_shortest_paths(
        #     graph, intermediary, target, n,
        #     distance=distance, strategy=strategy,
        #     exclude_edge=exclude_edge)

        # return a_b_paths, b_c_paths
