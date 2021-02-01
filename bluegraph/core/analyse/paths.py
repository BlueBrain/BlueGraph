from abc import (ABC, abstractmethod)

from bluegraph.core.utils import top_n

from bluegraph.exceptions import BlueGraphException


class PathFinder(ABC):
    """Abstract class for a path finder."""

    @staticmethod
    @abstractmethod
    def _get_edges(graph, properties=False):
        """Get edges of the underlying graph."""
        pass

    @staticmethod
    @abstractmethod
    def _generate_graph(pgframe):
        """Generate the appropiate graph representation from a PGFrame."""
        pass

    @staticmethod
    @abstractmethod
    def _get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        pass

    @staticmethod
    @abstractmethod
    def _get_neighbors(self, node_id):
        """Get neighors of the node."""
        pass

    @abstractmethod
    def _get_subgraph(self, node_filter, edge_filter=None):
        """Get a node/edge induced subgraph."""
        pass

    @staticmethod
    @abstractmethod
    def _compute_shortest_path(graph, source, target, distance=None,
                               exclude_edge=False):
        """Backend-dependent method for computing the shortest path."""
        pass

    @staticmethod
    @abstractmethod
    def _compute_all_shortest_paths(graph, source, target, exclude_edge=False):
        """Backend-dependent method for computing all the shortest paths."""

    @staticmethod
    @abstractmethod
    def _compute_yen_shortest_paths(graph, target, n,
                                    distance, exclude_edge=False):
        """Compute n shortest paths using the Yen's algo."""
        pass

    @abstractmethod
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
        tree : graph object
            The minimum spanning tree graph object (backend-dependent)
        """
        pass

    def get_subgraph(self, nodes_to_include=None,
                     node_filter=None,
                     edges_to_include=None,
                     edge_filter=None):
        """Produce a graph induced by the input nodes."""
        def include_node(x):
            included = True
            if nodes_to_include is not None:
                included = x in nodes_to_include
            unfiltered = True
            if node_filter is not None:
                unfiltered = node_filter(x)
            return included and unfiltered

        def include_edge(x):
            included = True
            if edges_to_include is not None:
                included = x in edges_to_include
            unfiltered = True
            if edge_filter is not None:
                unfiltered = edge_filter(x)
            return included and unfiltered

        subgraph = self._get_subgraph(include_node, include_edge)
        return subgraph

    def top_neighbors(self, node, n, weight, smallest=False):
        """Get top n neighbours of the specified node by weight."""
        neigbours = {}
        for neighbor in self._get_neighbors(self.graph, node):
            neigbours[neighbor] = self._get_distance(
                self.graph, node, neighbor, weight)
        return {
            el: neigbours[el]
            for el in top_n(neigbours, n, smallest=smallest)
        }

    def _get_cumulative_distances(self, paths, distance):
        """Get cumulative distance scores for provided paths."""
        def _sumup_distances(path):
            result = 0
            for i in range(1, len(path)):
                source = path[i - 1]
                target = path[i]
                if distance is not None:
                    result += self._get_distance(
                        self.graph, source, target, distance)
                else:
                    result += 1
            return result

        path_ranking = {
            p: _sumup_distances(p) for p in paths
        }

        return path_ranking

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
        return self._compute_shortest_path(
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
        return self._compute_all_shortest_paths(
            self.graph, source, target, exclude_edge=exclude_edge)

    def _compute_n_shortest_paths(self, graph, source, target, n,
                                  distance=None, strategy="naive",
                                  exclude_edge=False):
        if n == 1:
            return [
                self._compute_shortest_path(
                    graph, source, target, distance,
                    exclude_edge=exclude_edge)]

        if strategy == "naive":
            all_paths = self._compute_all_shortest_paths(
                graph, source, target, exclude_edge=True)

            path_ranking = self._get_cumulative_distances(
                all_paths, distance)

            if not exclude_edge:
                s_t_distance = self._get_cumulative_distances(
                        [(source, target)], distance)
                path_ranking.update(s_t_distance)

            paths = top_n(path_ranking, n, smallest=True)
        elif strategy == "yen":
            paths = self._compute_yen_shortest_paths(
                graph, source, target, n=n,
                distance=distance, exclude_edge=exclude_edge)
        else:
            PathFinder.PathSearchException(
                f"Unknown path search strategy '{strategy}'")
        return paths

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
        return self._compute_n_shortest_paths(
            self.graph, source, target, n, distance=distance,
            strategy=strategy, exclude_edge=exclude_edge)

    def nested_shortest_path(self, source, target, depth=1, distance=None,
                             exclude_edge=True):
        """Find the shortest nested path."""
        current_paths = [[source, target]]

        all_paths = set()
        visited = set()
        for level in range(depth):
            new_paths = []
            for current_path in current_paths:
                for i in range(1, len(current_path)):
                    s = current_path[i - 1]
                    t = current_path[i]
                    if s != t and (s, t) not in visited and (t, s) not in visited:
                        visited.add((s, t))
                        path = self.shortest_path(
                            s, t, distance=distance,
                            exclude_edge=exclude_edge)
                        all_paths.add(path)
                        new_paths += [list(path)]

            current_paths = new_paths
        return all_paths

    def n_nested_shortest_paths(self, source, target,  top_level_n,
                                nested_n=None, depth=1, distance=None,
                                strategy="naive", exclude_edge=False):
        """Find top n nested paths.
        Nested paths are found iteratively for each level of depth. For example,
        if `e1 <-> e2 <-> ... <-> eN` is a path on the current level of depth,
        then the function searches for paths between each consecutive pair of
        nodes (e1 and e2, e2 and e3, etc.).
        Parameters
        ----------
        graph : nx.Graph
            Input graph object
        source : str
            Source node ID
        target : str
            Target node ID
        top_level_n : int
            Number of top paths to include in the result
        nested_n : int
            Number of top paths to include in the result for the depth > 1
        depth : int, optional
            Number of interactions of the path search
        distance : str, optional
            The name of the attribute to use as the edge distance
        strategy : str, optional
            Path finding strategy: `naive` or `yen`. By default, `naive`.

        Returns
        -------
        current_paths : list
            List containing best nested paths according to the distance score
        """
        if nested_n is None:
            nested_n = top_level_n

        current_paths = [[source, target]]
        all_paths = set()
        visited = set()

        for level in range(depth):
            new_paths = []
            for path in current_paths:
                for i in range(1, len(path)):
                    s = path[i - 1]
                    t = path[i]
                    if (s, t) not in visited and (t, s) not in visited:
                        visited.add((s, t))
                        paths = self.n_shortest_paths(
                            s, t,
                            top_level_n if level == 0 else nested_n,
                            strategy=strategy, distance=distance,
                            exclude_edge=exclude_edge)
                        all_paths.update(paths)
                        new_paths += paths
            current_paths = new_paths
        return all_paths

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

        a_b_path = self.shortest_path(
            source, intermediary,
            distance=distance, exclude_edge=exclude_edge)

        def encountered_node(x):
            if overlap is False:
                return (
                    x not in list(a_b_path)[1:-1] or
                    x == intermediary or
                    x == target
                )
            else:
                return True

        subgraph = self.get_subgraph(
            node_filter=encountered_node)

        b_c_path = self._compute_shortest_path(
            subgraph,
            intermediary, target,
            distance=distance,
            exclude_edge=exclude_edge)

        return a_b_path, b_c_path

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
        a_b_paths = self.n_shortest_paths(
            source, intermediary, n, distance=distance,
            strategy=strategy, exclude_edge=exclude_edge)

        visited_nodes = set()
        for p in a_b_paths:
            visited_nodes.update(
                [
                    el for el in list(p)[1:-1]
                    if el != intermediary and el != target
                ])

        def encountered_node(x):
            if overlap is False:
                return x not in visited_nodes
            else:
                return True

        subgraph = self.get_subgraph(
            node_filter=encountered_node)
        try:
            b_c_paths = self._compute_n_shortest_paths(
                subgraph, intermediary, target, n,
                distance=distance, strategy=strategy,
                exclude_edge=exclude_edge)
        except PathFinder.NoPathException:
            raise PathFinder.NoPathException(
                "No paths satisfying the contraints from the "
                f"intermediary '{intermediary}' to the "
                f"target '{target}' exists")

        return a_b_paths, b_c_paths

    class PathSearchException(BlueGraphException):
        """Exception class for generic path search error."""
        pass

    class NoPathException(BlueGraphException):
        """Exception class for 'path does not exist."""
        pass

    class NotImplementedError(BlueGraphException):
        """Exception class for not implemented bits."""
        pass
