from abc import (ABC, abstractmethod)


class PathFinder(ABC):
    """Abstract class for a path finder."""
    def __init__(self, pgframe):
        self.graph = self._generate_graph(pgframe)

    @staticmethod
    @abstractmethod
    def _generate_graph(pgframe):
        """Generate the appropiate graph representation from a PGFrame."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def all_shortest_paths(self, source, target):
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
        pass

    @abstractmethod
    def n_shortest_paths(self, source, target, n, distance=None,
                         strategy="naive"):
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
        pass

    @abstractmethod
    def shortest_tripath(self, source, intermediary, target, distance=None):
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
        pass

    @abstractmethod
    def n_shortest_tripaths(self, source, intermediary, target,
                            n, distance=None, strategy="naive", overlap=True):
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
        pass
