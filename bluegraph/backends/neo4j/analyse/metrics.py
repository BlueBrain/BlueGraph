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
import warnings

from bluegraph.core.analyse.metrics import MetricProcessor

from ..io import Neo4jGraphProcessor, Neo4jGraphView


class Neo4jMetricProcessor(Neo4jGraphProcessor, MetricProcessor):
    """Class for metric processing based on Neso4j graphs."""

    def density(self):
        graph_view = self._get_identity_view()

        node_match = graph_view._get_nodes_query(no_return=True)
        edge_match = graph_view._get_edges_query(single_direction=True, no_return=True)
        query = (
            f"""{node_match}
            WITH count(n) as n_nodes
            {edge_match}
            RETURN toFloat(count(r)) / (n_nodes * (n_nodes - 1)) as density
            """
        )
        result = self.execute(query)
        for record in result:
            density = record["density"]
            break
        if not self.directed:
            density = density * 2
        return density

    def _run_gdc_query(self, function, metric_name, weight=None,
                       write=False, write_property=None,
                       score_name="score"):
        """Run a query for computing various centrality measures."""
        graph_view = self._get_identity_view()

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
