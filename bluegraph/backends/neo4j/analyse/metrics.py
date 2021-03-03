import warnings

from bluegraph.core.analyse.metrics import MetricProcessor

from ..io import Neo4jGraphProcessor, Neo4jGraphView


class Neo4jMetricProcessor(Neo4jGraphProcessor, MetricProcessor):
    """Class for metric processing based on Neso4j graphs."""

    def _run_gdc_query(self, function, metric_name, weight=None,
                       write=False, write_property=None,
                       score_name="score"):
        """Run a query for computing various centrality measures."""
        graph_view = Neo4jGraphView(
            self.driver, self.node_label,
            self.edge_label, directed=self.directed)

        node_edge_selector = graph_view.get_projection_query(
            weight)

        if write:
            if write_property is None:
                raise MetricProcessor.MetricProcessingException(
                    f"{metric_name.capitalize()} processing has the write "
                    "option set to True, "
                    "the write property name must be specified")

            query = (
                f"""
                CALL {function}.write({{
                   {node_edge_selector},\n
                   writeProperty: '{write_property}'
                }})
                YIELD createMillis
                """
            )
            self.execute(query)
        else:
            query = (
                f"""CALL {function}.stream({{
                    {node_edge_selector}
                }})
                YIELD nodeId, {score_name}
                RETURN gds.util.asNode(nodeId).id AS node_id, {score_name} AS {
                    metric_name}
                """
            )
            result = self.execute(query)
            return {
                record["node_id"]: record[metric_name]
                for record in result
            }

    def degree_centrality(self, weight=None, write=False,
                          write_property=None):
        """Compute (weighted) degree centrality."""
        result = self._run_gdc_query(
            "gds.alpha.degree", "degree", weight=weight,
            write=write, write_property=write_property)
        return result

    def pagerank_centrality(self, weight=None, write=False,
                            write_property=None):
        """Compute (weighted) PageRank centrality."""
        result = self._run_gdc_query(
            "gds.pageRank", "degree", weight=weight,
            write=write, write_property=write_property)
        return result

    def betweenness_centrality(self, distance=None, write=False,
                               write_property=None):
        """Compute (weighted) betweenness centrality."""
        if distance is not None:
            warnings.warn(
                "Weighted betweenness centrality for Neo4j graphs "
                "is not implemented: computing the unweighted version",
                MetricProcessor.MetricProcessingWarning)
        result = self._run_gdc_query(
            "gds.betweenness", "betweenness", weight=None,
            write=write, write_property=write_property)
        return result

    def closeness_centrality(self, distance=None, write=False,
                             write_property=None):
        """Compute (weighted) closeness centrality."""
        if distance is not None:
            warnings.warn(
                "Weighted closeness centrality for Neo4j graphs "
                "is not implemented: computing the unweighted version",
                MetricProcessor.MetricProcessingWarning)
        result = self._run_gdc_query(
            "gds.alpha.closeness", "closeness", weight=None,
            write=write, write_property=write_property,
            score_name="centrality")
        return result
