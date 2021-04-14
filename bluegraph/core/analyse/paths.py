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

from abc import (ABC, abstractmethod)

from bluegraph.core.utils import top_n

from bluegraph.exceptions import BlueGraphException


def pretty_print_paths(paths, as_repr=False):
    """Pretty print a set of same source/target paths."""
    a = paths[0][0]
    b = paths[0][-1]
    a_repr = "{} <-> ".format(a)
    b_repr = " <-> {}".format(b)
    path_repr = [
        " <-> ".join(p[1:-1])
        for p in paths
    ]
    lines = ["{}{}{}".format(
        a_repr, " " * max(len(p) for p in path_repr), b_repr)]
    lines += ["{}{}".format(" " * len(a_repr), p) for p in path_repr]
    if as_repr:
        return "\n".join(lines)
    for line in lines:
        print(line)


def pretty_print_tripaths(a, b, c, n, a_b_paths, b_c_paths, as_repr=False):
    a_b_paths_repr = [
        " -> ".join(p[1:-1]) for p in a_b_paths
    ]

    b_c_paths_repr = [
        " -> ".join(p[1:-1]) for p in b_c_paths
    ]

    max_left = max([len(el) for el in a_b_paths_repr])
    max_right = max([len(el) for el in b_c_paths_repr])

    a_repr = "{} ->".format(a)
    b_repr = "-> {} ->".format(b)
    c_repr = "-> {}".format(c)
    lines = ["{}{}{}{}{}".format(
        a_repr,
        " " * max_left,
        b_repr,
        " " * max_right,
        c_repr)]
    for i in range(n):
        if i >= len(a_b_paths) and i >= len(b_c_paths):
            break
        left = a_b_paths_repr[i] if i < len(a_b_paths) else (" " * max_left)
        right = b_c_paths_repr[i] if i < len(b_c_paths) else (" " * max_right)
        lines += (
            " " * len(a_repr), left,
            " " * (max_left - len(left) + len(b_repr)),
            right)
    if as_repr:
        return "\n".join(lines)
    for line in lines:
        print(line)


def graph_elements_from_paths(paths):
    """Create a graph from a set of paths.

    Resulting graph contains nodes and edges from the input
    set  of paths. If the source graph is provided, the attributes
    of the selected nodes and edges are copied from the source graph.
    """
    nodes = set()
    edges = set()
    for p in paths:
        nodes.add(p[0])
        for i in range(1, len(p)):
            nodes.add(p[i])
            edges.add((p[i - 1], p[i]))
    return nodes, edges


class PathFinder(ABC):
    """Abstract class for a path finder."""

    @abstractmethod
    def get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        pass

    @abstractmethod
    def get_subgraph_from_paths(self, paths):
        """Get a subgraph given the input paths."""
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
            The minimum spanning tree graph object
        """
        pass

    def top_neighbors(self, node, n, weight, smallest=False):
        """Get top n neighbours of the specified node by weight."""
        neighbors = {}
        for neighbor in self.neighbors(node):
            neighbors[neighbor] = self.get_distance(
                node, neighbor, weight)
        return {
            el: neighbors[el]
            for el in top_n(neighbors, n, smallest=smallest)
        }

    def _get_cumulative_distances(self, paths, distance):
        """Get cumulative distance scores for provided paths."""
        def _sumup_distances(path):
            result = 0
            for i in range(1, len(path)):
                source = path[i - 1]
                target = path[i]
                if distance is not None:
                    result += self.get_distance(
                        source, target, distance)
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
                if (source, target) in self.edges():
                    s_t_distance = self._get_cumulative_distances(
                        [(source, target)], distance)
                    path_ranking.update(s_t_distance)
                else:
                    if not self.directed and (target, source) in self.edges():
                        t_s_distance = self._get_cumulative_distances(
                            [(target, source)], distance)
                        path_ranking.update({
                            (source, target): t_s_distance[(target, source)]
                        })

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
        return list(all_paths)

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

        subgraph = self.subgraph(
            nodes_to_exclude=[
                x
                for x in list(a_b_path)[1:-1]
                if x != intermediary and x != target
            ])

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

        subgraph = self.subgraph(
            nodes_to_exclude=visited_nodes if overlap is False else None)
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
