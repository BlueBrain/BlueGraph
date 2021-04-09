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

from ..io import Neo4jGraphProcessor, Neo4jGraphView


class Neo4jPathFinder(Neo4jGraphProcessor, PathFinder):
    """Neo4j-based shortest paths finder."""

    def get_distance(self, source, target, distance):
        """Get distance value between source and target."""
        graph = self._get_identity_view()
        query = graph._get_edge_query(source, target)
        result = self.execute(query)
        return [record["edge"][distance] for record in result][0]

    @staticmethod
    def _generate_path_search_call(graph, source, target, procedure,
                                   distance=None, exclude_edge=False,
                                   extra_params=None):
        if extra_params is None:
            extra_params = {}

        node_edge_selector = graph.get_projection_query(
            distance)

        distance_setter = (
            "  relationshipWeightProperty: '{}',\n".format(distance)
            if distance else ""
        )

        extra_params = ("," if len(extra_params) > 0 else "") + "\n".join(
            "{}: {}".format(k, v) for k, v in extra_params.items())

        query = (
            "CALL {}({{\n".format(procedure) +
            "{},\n{}".format(
                node_edge_selector, distance_setter) +
            "    sourceNode: id(start),\n" +
            "    targetNode: id(end){}\n".format(extra_params) +
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
                "gds.beta.shortestPath.dijkstra.stream",
                distance, exclude_edge) +
            "YIELD nodeIds\n"
            "UNWIND [n IN nodeIds | gds.util.asNode(n).id] AS node_id \n"
            "RETURN node_id"
        )
        result = graph.execute(query)
        return tuple(record["node_id"] for record in result)

    @staticmethod
    def _compute_all_shortest_paths(graph, source, target, exclude_edge=False,
                                    max_length=4):
        """Backend-dependent method for computing all the shortest paths."""
        exclude_statement = "WHERE length(path) > 1\n" if exclude_edge else ""
        arrow = ">" if graph.directed else ""
        query = (
            graph._generate_st_match_query(source, target) +
            "WITH start, end\n"
            "MATCH path = allShortestPaths((start)-"
            "[:{}*..{}]-{}(end))\n{}".format(
                graph.edge_label, max_length, arrow, exclude_statement) +
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
            Neo4jPathFinder._generate_path_search_call(
                graph, source, target,
                "gds.beta.shortestPath.yens.stream",
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

    def get_subgraph_from_paths(self, paths):
        """Get a subgraph given the input paths."""
        nodes, edges = graph_elements_from_paths(paths)
        subgraph = Neo4jGraphView(
            driver=self.driver, node_label=self.node_label,
            edge_label=self.edge_label, nodes_to_exclude=[
                n for n in self.nodes() if n not in nodes
            ], edges_to_exclude=[
                e for e in self.edges() if e not in edges
            ])
        return subgraph

    def top_neighbors(self, node, n, weight, smallest=False):
        """Get top n neighbours of the specified node by weight."""
        order = "ASC" if smallest else "DESC"
        query = (
            "MATCH (n:{} {{id: '{}'}})-[r:{}]-(m:{})\n".format(
                self.node_label, node,
                self.edge_label, self.node_label) +
            "RETURN m.id as neighor_id, r.{} as distance\n".format(weight) +
            "ORDER by distance {} LIMIT {}".format(order, n)
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
                "Unknown path search strategy '{}'".format(strategy))
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
            head = "MATCH (n:{} {{id: '{}'}})\n".format(
                self.node_label, start_node)
        else:
            head = (
                "MATCH (n:{})\n".format(self.node_label) +
                "WITH n, rand() as r ORDER BY r LIMIT 1\n"
            )

        graph = self._get_identity_view()

        query = (
            head +
            "CALL gds.alpha.spanningTree.minimum.write({\n" +
            graph.get_projection_query(distance) + ","
            "   startNodeId: id(n),\n"
            "   relationshipWeightProperty: '{}',\n".format(distance) +
            "   writeProperty: '{}',\n".format(write_edge_label) +
            "   weightWriteProperty: 'writeCost'\n"
            "})\n"
            "YIELD createMillis, computeMillis, writeMillis\n"
            "RETURN createMillis, computeMillis, writeMillis"

        )
        self.execute(query)
