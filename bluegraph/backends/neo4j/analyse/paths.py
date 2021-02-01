from bluegraph.core.analyse.paths import PathFinder

from ..io import Neo4jGraphProcessor


class Neo4jPathFinder(Neo4jGraphProcessor, PathFinder):
    """Neo4j-based shortest paths finder."""

    @staticmethod
    def _get_edges(graph, properties=False):
        """Get edges of the underlying graph."""
        pass

    @staticmethod
    def _get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        pass

    @staticmethod
    def _get_neighbors(graph, node_id):
        """Get neighors of the node."""
        pass

    def _get_subgraph(self, node_filter, edge_filter=None):
        """Get a node/edge induced subgraph."""
        pass

    @staticmethod
    def _compute_shortest_path(graph, source, target, distance=None,
                               exclude_edge=False):
        """Backend-dependent method for computing the shortest path."""
        pass

    @staticmethod
    def _compute_all_shortest_paths(graph, source, target, exclude_edge=False):
        """Backend-dependent method for computing all the shortest paths."""

    @staticmethod
    def _compute_yen_shortest_paths(graph, target, n,
                                    distance, exclude_edge=False):
        """Compute n shortest paths using the Yen's algo."""
        pass

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

    def top_neighbors(self, node, n, weight, smallest=False):
        """Get top n neighbours of the specified node by weight."""
        order = "ASC" if smallest else "DESC"
        query = (
            f"MATCH (n:{self.node_label} {{id: '{node}'}})-"
            f"[r:{self.edge_label}]-(m:{self.node_label})\n"
            f"RETURN m.id as neighor_id, r.{weight} as distance\n"
            f"ORDER by distance {order} LIMIT {n}"
        )
        result = self.execute(query)
        return {
            record["neighor_id"]: record["distance"]
            for record in result
        }

    def _generate_match_query(self, source, target):
        return (
            f"MATCH (start:{self.node_label} {{id: '{source}'}}), "
            f"(end:{self.node_label} {{id: '{target}'}})\n"
        )

    def _generate_path_search_call(self, source, target, procedure,
                                   distance=None, exclude_edge=False,
                                   extra_params=None):
        if extra_params is None:
            extra_params = {}

        if exclude_edge is False:
            distance_selector = (
                f"       properties: '{distance}',\n"
                if distance else ""
            )
            node_edge_selector = (
                f"  nodeProjection: '{self.node_label}',\n"
                f"  relationshipProjection: {{\n"
                f"    Edge: {{\n"
                f"      type: '{self.edge_label}',\n{distance_selector}"
                f"      orientation: 'UNDIRECTED'\n"
                f"    }}\n"
                "  }\n"
            )
        else:
            distance_selector = f", r.{distance} as distance'\n"
            node_edge_selector = (
                f"  nodeQuery: 'MATCH (n:{self.node_label}) RETURN id(n) as id',\n"
                f"  relationshipQuery: 'MATCH (n)-[r:{self.edge_label}]-(m) WHERE "
                f"NOT (n.id=\"{source}\" AND m.id=\"{target}\") "
                f"RETURN id(n) AS source, id(m) AS target {distance_selector}"
            )

        distance_setter = (
            f"  relationshipWeightProperty: '{distance}',\n"
            if distance else ""
        )

        extra_params = ("," if len(extra_params) > 0 else "") + "\n".join(
            f"{k}: {v}" for k, v in extra_params.items())

        query = (
            f"CALL {procedure}({{\n"
            f"{node_edge_selector},\n{distance_setter}"
            f"    startNode: start,\n"
            f"    endNode: end{extra_params}\n"
            "})\n"
        )
        return query

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
        query = (
            self._generate_match_query(source, target) +
            self._generate_path_search_call(
                source, target,
                "gds.alpha.shortestPath.stream", distance, exclude_edge) +
            f"YIELD nodeId\n"
            f"RETURN gds.util.asNode(nodeId).id AS node_id\n"
        )
        result = self.execute(query)
        return tuple(record["node_id"] for record in result)

    def all_shortest_paths(self, source, target, exclude_edge=False,
                           max_length=4):
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
        max_length : int, optional
            Maximum allowed path length (the larger the value, the slower
            is the performance)
        """
        exclude_statement = "WHERE length(path) > 1\n" if exclude_edge else ""
        query = (
            self._generate_match_query(source, target) +
            "WITH start, end\n"
            "MATCH path = allShortestPaths((start)-"
            f"[:{self.edge_label}*..{max_length}]-(end))\n{exclude_statement}"
            "RETURN [n IN nodes(path) | n.id] as path"
        )
        result = self.execute(query)
        return [
            tuple(record["path"])
            for record in result
        ]

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
        if n == 1:
            return [
                self.shortest_path(
                    source, target, distance, exclude_edge)]

        if strategy == "naive":
            raise PathFinder.NotImplementedError(
                "Naive algorithm for finding n shortest paths "
                "is currently not implemented for Neo4j backend")
        elif strategy == "yen":
            query = (
                self._generate_match_query(source, target) +
                self._generate_path_search_call(
                    source, target,
                    "gds.alpha.kShortestPaths.stream",
                    distance, exclude_edge,
                    extra_params={"k": n}) +
                "YIELD nodeIds\n"
                "RETURN [node IN gds.util.asNodes(nodeIds) | node.id] AS nodes"
            )
            result = self.execute(query)
            paths = [
                tuple(record["nodes"])
                for record in result
            ]
        else:
            PathFinder.PathSearchException(
                f"Unknown path search strategy '{strategy}'")
        return paths