import warnings

from bluegraph.core.analyse.metrics import MetricProcessor
from bluegraph.exceptions import (MetricProcessingException,
                                  MetricProcessingWarning)

from neo4j import GraphDatabase
from ..io import (pgframe_to_neo4j, neo4j_to_pgframe)


class Neo4jMetricProcessor(MetricProcessor):
    """Class for metric processing based on Neso4j graphs."""

    def __init__(self, pgframe=None, uri=None, username=None, password=None,
                 driver=None, node_label=None, edge_label=None):
        if node_label is None:
            raise MetricProcessingException(
                "Cannot initialize a Neo4jMetricProcessor: "
                "node label must be specified")
        if edge_label is None:
            raise MetricProcessingException(
                "Cannot initialize a Neo4jMetricProcessor: "
                "edge label must be specified")
        if driver is None:
            self.driver = GraphDatabase.driver(
                uri, auth=(username, password))
        else:
            self.driver = driver
        self.node_label = node_label
        self.edge_label = edge_label
        if pgframe is not None:
            self._generate_graph(
                pgframe, driver=driver,
                node_label=node_label, edge_label=edge_label)

    @classmethod
    def from_graph_object(cls, graph_object):
        """Instantiate a MetricProcessor directly from a Graph object."""
        raise MetricProcessingException(
            "Neo4jMetricProcessor cannot be initialized from a graph object")

    @staticmethod
    def _generate_graph(pgframe, driver=None,
                        node_label=None, edge_label=None, directed=True):
        return pgframe_to_neo4j(
            pgframe=pgframe,
            driver=driver, node_label=node_label,
            edge_label=edge_label)

    def execute(self, query):
        session = self.driver.session()
        response = session.run(query)
        result = response.data()
        session.close()
        return result

    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        pass

    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
        pass

    def _dispatch_processing_result(self, new_property, metric_name,
                                    write=False,
                                    write_property=None):
        pass

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
                raise MetricProcessingException(
                    f"{metric_name.capitalize()} processing has the write "
                    "option set to True, "
                    "the write property name must be specified")
            query = (
                f"""
                CALL {function}.write({{
                    nodeProjection: '{self.node_label}',
                    relationshipProjection: {{
                       Edge: {{
                           type: '{self.edge_label}',
                           orientation: 'UNDIRECTED'{property_projection}
                       }}
                   }}{property_name},
                   writeProperty: '{write_property}'
                }})
                YIELD createMillis
                """
            )
            self.execute(query)
        else:
            query = (
                f"""CALL {function}.stream({{
                    nodeProjection: '{self.node_label}',
                    relationshipProjection: {{
                        Edge: {{
                            type: '{self.edge_label}',
                            orientation: 'UNDIRECTED'{property_projection}
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
                MetricProcessingWarning)
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
                MetricProcessingWarning)
        result = self._run_gdc_query(
            "gds.alpha.closeness", "closeness", weight=None,
            write=write, write_property=write_property,
            score_name="centrality")
        return result

    def get_pgframe(self):
        """Get a new pgframe object from the wrapped graph object."""
        return neo4j_to_pgframe(self.driver, self.node_label, self.edge_label)
