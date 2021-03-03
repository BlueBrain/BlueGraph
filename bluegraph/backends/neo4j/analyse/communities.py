from bluegraph.core.analyse.communities import CommunityDetector
from ..io import Neo4jGraphProcessor, Neo4jGraphView


class Neo4jCommunityDetector(Neo4jGraphProcessor, CommunityDetector):
    """Neo4j-based community detection interface.

    https://neo4j.com/docs/graph-data-science/current/algorithms/community/
    - Louvain
    - Label Propagation
    """
    def _run_community_gdc_query(self, function, weight=None,
                                 write=False, write_property=None):
        """Run a query for computing various communities."""
        graph_view = Neo4jGraphView(
            self.driver, self.node_label,
            self.edge_label, directed=self.directed)

        node_edge_selector = graph_view.get_projection_query(weight)
        if write:
            if write_property is None:
                raise CommunityDetector.EvaluationError(
                    "Community processing has the write "
                    "option set to True, "
                    "the write property name must be specified")
            query = (
                f"""CALL gds.{function}.write({{
                  {node_edge_selector},\n
                  writeProperty: '{write_property}'
                }})
                YIELD communityCount
                """
            )
            result = self.execute(query)
        else:
            query = (
                f"""
                CALL gds.{function}.stream({{
                  {node_edge_selector}
                }})
                YIELD nodeId, communityId
                """
            )
            result = self.execute(query)
            return {record["nodeId"]: record["communityId"] for record in result}

    def _run_louvain(self, weight=None, write=False, write_property=None,
                     **kwargs):
        result = self._run_community_gdc_query(
            "louvain", weight=weight, write=write,
            write_property=write_property)
        return result

    def _run_girvan_newman(self, weight=None, n_communitites=2,
                           intermediate=False, write=False, write_property=None,
                           **kwargs):
        result = self._run_community_gdc_query(
            "beta.modularityOptimization", weight=weight,
            write=write, write_property=write_property)
        return result

    def _run_stochastic_block_model(self, write=False, write_property=None,
                                    **kwargs):
        raise CommunityDetector.PartitionError(
            "Stochastic block model is not implemented "
            "for Neo4j-based graphs")

    def _run_label_propagation(self, weight=None, write=False,
                               write_property=None, **kwargs):
        result = self._run_community_gdc_query(
            "labelPropagation", weight=weight, write=write,
            write_property=write_property)
        return result

    def _compute_modularity(self, partition, weight=None):
        partition_repr = "{{ {} }}".format(
            ", ".join([f"node{k}: {v}" for k, v in partition.items()]))
        s_degree_clause = (
            "count(DISTINCT r1)"
            if weight is None else f"sum(DISTINCT r1.{weight})"
        )
        t_degree_clause = (
            "count(DISTINCT r2)"
            if weight is None else f"sum(DISTINCT r2.{weight})"
        )
        s_t_edge_clause = "1" if weight is None else f"r.{weight}"

        query = (
            f"""
            WITH {partition_repr} AS partition
            MATCH (n1:{self.node_label})-[r1:{self.edge_label}]-(:{self.node_label}), 
                (n2:{self.node_label})-[r2:{self.edge_label}]-(:{self.node_label})
            WHERE id(n1) < id(n2)
            OPTIONAL MATCH (n1)-[r:{self.edge_label}]-(n2)
            WITH n1.id AS s, n2.id AS t, {s_degree_clause} AS s_degree,
               {t_degree_clause} AS t_degree,
               CASE WHEN r IS NULL THEN 0 ELSE {s_t_edge_clause} END AS s_t_edge,
               CASE WHEN partition["node" + n1.id] = partition["node" + n2.id] THEN 1 ELSE 0 END AS s_t_community
            WITH collect([s, t, s_degree, t_degree, s_t_edge, s_t_community]) AS data, sum(s_t_edge) AS edges
            WITH [el IN data WHERE el[5] = 1 | el] AS filtered_data, edges
            WITH edges, [el IN filtered_data | el[4] - toFloat(el[2] * el[3]) / (2 * edges)] AS terms
            UNWIND terms AS term
            RETURN (1 / (2 * toFloat(edges))) * sum(term) as modularity
            """
        )
        result = self.execute(query)
        modularity = None
        for record in result:
            modularity = record["modularity"]
            break
        return modularity

    def _compute_performance(self, partition, weight=None):
        query = (
            f"""MATCH (n1:{self.node_label}), (n2:{self.node_label})
            WHERE id(n1) <= id(n2)
            WITH n1, n2
            OPTIONAL MATCH (n1)-[r:{self.edge_label}]-(n2)
            WITH collect([n1, n2, r]) as data, collect(DISTINCT n1) as nodes
            WITH [el IN data WHERE el[0].louvain_community <> el[1].louvain_community AND el[2] IS NULL] as non_interedges,
                 [el in data WHERE el[0].louvain_community = el[1].louvain_community AND NOT el[2] IS NULL| el] as intraedges,
                 nodes
            RETURN toFloat(size(intraedges) + size(non_interedges)) / (toFloat(size(nodes) * (size(nodes) - 1)) / 2) as performance
            """
        )
        result = self.execute(query)
        performance = None
        for record in result:
            performance = record["performance"]
            break
        return performance

    def _compute_coverage(self, partition, weight=None):
        if weight is None:
            formula = (
                "toFloat(size([el in edges WHERE el[1].louvain_community = el[2].louvain_community | el])) / size(edges)"
            )
        else:
            formula = (
                "toFloat(reduce(sum = 0, el IN edges "
                "WHERE el[1].louvain_community = el[2].louvain_community | " +
                f"sum + el[0].{weight}) / reduce(sum = 0, el IN edges | sum + el[0].{weight})"
            )
        query = (
            f"""
            MATCH (n1:{self.node_label})-[r:{self.edge_label}]->(n2:{self.node_label})
            WITH collect([r, n1, n2]) as edges
            RETURN {formula} as coverage
            """)
        result = self.execute(query)
        coverage = None
        for record in result:
            coverage = record["coverage"]
            break
        return coverage

    def detect_communities(self, strategy="louvain", weight=None,
                           n_communities=2, intermediate=False,
                           write=False, write_property=None, **kwargs):
        """Detect community partition using the input strategy."""
        if strategy not in CommunityDetector._strategies.keys():
            raise CommunityDetector.PartitionError(
                f"Unknown community detection strategy '{strategy}'")
        partition = getattr(self, CommunityDetector._strategies[strategy])(
            weight=weight, n_communities=n_communities,
            intermediate=intermediate, write=write,
            write_property=write_property,
            **kwargs)
        return partition
