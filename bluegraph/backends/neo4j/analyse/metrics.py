import warnings

from bluegraph.core.analyse.metrics import MetricProcessor

from ..io import Neo4jGraphProcessor


class Neo4jMetricProcessor(Neo4jGraphProcessor, MetricProcessor):
    """Class for metric processing based on Neso4j graphs."""

    def _run_gdc_query(self, function, metric_name, weight=None,
                       write=False, write_property=None,
                       score_name="score"):
        """Compute (weighted) degree centrality."""
        property_projection = (
            f",\nproperties: '{weight}'"
            if weight else ""
        )
        property_name = (
            f",\nrelationshipWeightProperty: '{weight}'"
            if weight else ""
        )
        if write:
            if write_property is None:
                raise MetricProcessor.MetricProcessingException(
                    f"{metric_name.capitalize()} processing has the write "
                    "option set to True, "
                    "the write property name must be specified")

            orientation = 'NATURAL' if self.directed else 'UNDIRECTED'
            query = (
                f"""
                CALL {function}.write({{
                    nodeProjection: '{self.node_label}',
                    relationshipProjection: {{
                       Edge: {{
                           type: '{self.edge_label}',
                           orientation: '{orientation}'{property_projection}
                       }}
                   }}{property_name},
                   writeProperty: '{write_property}'
                }})
                YIELD createMillis
                """
            )
            self.execute(query)
        else:
            orientation = 'NATURAL' if self.directed else 'UNDIRECTED'
            query = (
                f"""CALL {function}.stream({{
                    nodeProjection: '{self.node_label}',
                    relationshipProjection: {{
                        Edge: {{
                            type: '{self.edge_label}',
                            orientation: '{orientation}'{property_projection}
                        }}
                    }}{property_name}
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
