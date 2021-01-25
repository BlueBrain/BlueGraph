from bluegraph.core.analyse.metrics import MetricProcessor

import networkx as nx

from ..io import pgframe_to_networkx


class NXMetricProcessor(MetricProcessor):
    """Class for metric processing based on NetworkX graphs."""

    @staticmethod
    def _generate_graph(pgframe, directed=True):
        return pgframe_to_networkx(pgframe)

    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        return new_property

    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
        nx.set_node_attributes(
            self.graph, new_property, property_name)

    def degree_centrality(self, weight=None, write=False,
                          write_property=None):
        """Compute (weighted) degree centrality."""
        degree_centrality = dict(self.graph.degree(weight=weight))
        return self._dispatch_processing_result(
            degree_centrality, "degree", write, write_property)

    def pagerank_centrality(self, weight=None, write=False,
                            write_property=None):
        """Compute (weighted) PageRank centrality."""
        pagerank_centrality = nx.pagerank(self.graph, weight=weight)
        return self._dispatch_processing_result(
            pagerank_centrality, "pageRank", write, write_property)

    def betweenness_centrality(self, distance=None, write=False,
                               write_property=None):
        """Compute (weighted) betweenness centrality."""
        betweenness_centrality = nx.betweenness_centrality(
            self.graph, weight=distance)
        return self._dispatch_processing_result(
            betweenness_centrality, "betweenness", write, write_property)

    def closeness_centrality(self, distance=None, write=False,
                             write_property=None):
        """Compute (weighted) closeness centrality."""
        closeness_centrality = nx.closeness_centrality(
            self.graph, distance=distance)
        return self._dispatch_processing_result(
            closeness_centrality, "closeness", write, write_property)
