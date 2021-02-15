from bluegraph.core.analyse.paths import PathFinder

from ..io import Neo4jGraphProcessor, Neo4jGraphView


class Neo4jPathFinder(Neo4jGraphProcessor, PathFinder):
    """Neo4j-based shortest paths finder."""

    def _get_identity_view(self):
        return Neo4jGraphView(
            self.driver, self.node_label, self.edge_label,
            directed=self.directed)

    @staticmethod
    def _get_nodes(graph, properties=False):
        """Get nodes of the input graph."""
        query = graph._get_nodes_query()
        result = graph.execute(query)
        nodes = []
        for record in result:
            n = record["node_id"]
            if properties:
                props = record["node"]
                nodes.append((n, props))
            else:
                nodes.append(n)
        return nodes

    def get_nodes(self, properties=False):
        """Get nodes of the underlying graph."""
        graph = self._get_identity_view()
        return self._get_nodes(
            graph, properties=properties)

    @staticmethod
    def _get_edges(graph, properties=False):
        """Get edges of the input graph."""
        query = graph._get_edges_query(single_direction=True)
        result = graph.execute(query)
        edges = []
        for record in result:
            s = record["source_id"]
            t = record["target_id"]
            if properties:
                props = record["edge"]
                edges.append((s, t, props))
            else:
                edges.append((s, t))
        return edges

    def get_edges(self, properties=False):
        """Get edges of the underlying graph."""
        graph = self._get_identity_view()
        return self._get_edges(
            graph, properties=properties)

    def get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        graph = self._get_identity_view()
        query = graph._get_edge_query(source, target)
        result = self.execute(query)
        return [record["edge"][distance] for record in result][0]

    def get_neighbors(self, node_id):
        """Get neighors of the node."""
        query = (
            f"MATCH (n:{self.node_label} {{id: '{node_id}'}})-"
            f"[r:{self.edge_label}]-(m:{self.node_label})\n"
            "RETURN m.id as neighor"
        )
        result = self.execute(query)
        return [record["neighor"] for record in result]

    def get_subgraph(self, nodes_to_exclude, edges_to_exclude=None):
        """Get a node/edge induced subgraph."""
        return Neo4jGraphView(
            self.driver, self.node_label, self.edge_label,
            nodes_to_exclude=nodes_to_exclude,
            edges_to_exclude=edges_to_exclude,
            directed=self.directed)

    @staticmethod
    def _generate_path_search_call(graph, source, target, procedure,
                                   distance=None, exclude_edge=False,
                                   extra_params=None):
        if extra_params is None:
            extra_params = {}

        node_edge_selector = graph.get_projection_query(
            distance)

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

    @staticmethod
    def _compute_shortest_path(graph, source, target, distance=None,
                               exclude_edge=False):
        """Backend-dependent method for computing the shortest path."""
        if exclude_edge is True:
            graph.edges_to_exclude.append((source, target))

        query = (
            graph._generate_st_match_query(source, target) +
            Neo4jPathFinder._generate_path_search_call(
                graph, source, target,
                "gds.alpha.shortestPath.stream",
                distance, exclude_edge) +
            "YIELD nodeId\n"
            f"RETURN gds.util.asNode(nodeId).id AS node_id\n"
        )
        result = graph.execute(query)
        return tuple(record["node_id"] for record in result)

    @staticmethod
    def _compute_all_shortest_paths(graph, source, target, exclude_edge=False,
                                    max_length=4):
        """Backend-dependent method for computing all the shortest paths."""
        exclude_statement = "WHERE length(path) > 1\n" if exclude_edge else ""
        query = (
            graph._generate_st_match_query(source, target) +
            "WITH start, end\n"
            "MATCH path = allShortestPaths((start)-"
            f"[:{graph.edge_label}*..{max_length}]-(end))\n{exclude_statement}"
            "RETURN [n IN nodes(path) | n.id] as path"
        )
        result = graph.execute(query)
        return [
            tuple(record["path"])
            for record in result
        ]

    @staticmethod
    def _compute_yen_shortest_paths(graph, source, target, n,
                                    distance, exclude_edge=False):
        """Compute n shortest paths using the Yen's algo."""
        query = (
            graph._generate_st_match_query(source, target) +
            graph._generate_path_search_call(
                source, target,
                "gds.alpha.kShortestPaths.stream",
                distance, exclude_edge,
                extra_params={"k": n}) +
            "YIELD nodeIds\n"
            "RETURN [node IN gds.util.asNodes(nodeIds) | node.id] AS nodes"
        )
        result = graph.execute(query)
        return [
            tuple(record["nodes"])
            for record in result
        ]

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
        graph_view = Neo4jGraphView(
                self.driver, self.node_label,
                self.edge_label,
                edges_to_exclude=[(source, target)] if exclude_edge else None,
                directed=self.directed)
        return self._compute_shortest_path(
            graph_view, source, target,
            distance=distance, exclude_edge=exclude_edge)

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
        graph_view = Neo4jGraphView(
                self.driver, self.node_label,
                self.edge_label,
                edges_to_exclude=[(source, target)] if exclude_edge else None,
                directed=self.directed)
        return self._compute_all_shortest_paths(
            graph_view, source, target, exclude_edge=exclude_edge,
            max_length=max_length)

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
            graph_view = Neo4jGraphView(
                self.driver, self.node_label,
                self.edge_label,
                edges_to_exclude=[(source, target)] if exclude_edge else None,
                directed=self.directed)
            paths = self._compute_yen_shortest_paths(
                graph_view, source, target, n=n,
                distance=distance, exclude_edge=exclude_edge)
        else:
            PathFinder.PathSearchException(
                f"Unknown path search strategy '{strategy}'")
        return paths

    def minimum_spanning_tree(self, distance,
                              write=False, write_edge_label=None,
                              start_node=None):
        """Compute the minimum spanning tree.

        Parameters
        ----------
        distance : str
            Distance to minimize when computing the minimum spanning tree (MST)
        write : bool, optional
            Flag indicating whether the MST should be returned as a new graph
            object or saved within a Boolean edge property being True whenever
            a given edge belongs to the MST.
        write_edge_label : str, optional
            Edge label for creating edges beloning to the MST.

        Returns
        -------
        tree : graph object
            The minimum spanning tree graph object (backend-dependent)
        """
        if write is False:
            raise PathFinder.PathSearchException(
                "Minimum spanning tree computation with "
                "the parameter `write=False` is currently is not "
                "supported for Neo4j graphs")
        else:
            if write_edge_label is None:
                raise PathFinder.PathSearchException(
                    "The minimum spanning tree computation "
                    "has the `write` option set to `True`, "
                    "the write property name must be specified")

        if start_node is not None:
            head = f"MATCH (n:{self.node_label} {{id: '{start_node}'}})\n"
        else:
            head = (
                f"MATCH (n:{self.node_label})\n"
                "WITH n, rand() as r ORDER BY r LIMIT 1\n"
            )

        graph = self._get_identity_view()

        query = (
            head +
            "CALL gds.alpha.spanningTree.minimum.write({\n" +
            graph.get_projection_query(distance) + ","
            "   startNodeId: id(n),\n"
            f"   relationshipWeightProperty: '{distance}',\n"
            f"   writeProperty: '{write_edge_label}',\n"
            "   weightWriteProperty: 'writeCost'\n"
            "})\n"
            "YIELD createMillis, computeMillis, writeMillis\n"
            "RETURN createMillis, computeMillis, writeMillis"

        )
        self.execute(query)
