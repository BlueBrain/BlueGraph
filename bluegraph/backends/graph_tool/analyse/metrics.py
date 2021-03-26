from bluegraph.core.analyse.metrics import MetricProcessor

from graph_tool.centrality import pagerank as gt_pagerank
from graph_tool.centrality import betweenness as gt_betweenness
from graph_tool.centrality import closeness as gt_closeness

from ..io import GTGraphProcessor


class GTMetricProcessor(GTGraphProcessor, MetricProcessor):

    def density(self):
        factor = 2 if self.undirected else 1
        return (
            self.graph.num_edges() /
            ((self.graph.num_vertices() * (self.graph.num_vertices() - 1)) / factor)
        )

    def degree_centrality(self, weight=None, write=False,
                          write_property=None):
        """Compute (weighted) degree centrality."""
        weight = (
            self.graph.edge_properties[weight]
            if weight is not None
            else None
        )
        degree = self.graph.degree_property_map("out", weight=weight)
        return self._dispatch_processing_result(
            degree, "degree", write, write_property)

    def pagerank_centrality(self, weight=None, write=False,
                            write_property=None):
        """Compute (weighted) PageRank centrality."""
        weight = (
            self.graph.edge_properties[weight]
            if weight is not None
            else None
        )
        pagerank = gt_pagerank(self.graph, weight=weight)
        return self._dispatch_processing_result(
            pagerank, "pageRank", write, write_property)

    def betweenness_centrality(self, distance=None, write=False,
                               write_property=None):
        """Compute (weighted) betweenness centrality."""
        distance = (
            self.graph.edge_properties[distance]
            if distance is not None
            else None
        )
        betweenness, _ = gt_betweenness(
            self.graph, weight=distance)
        return self._dispatch_processing_result(
            betweenness, "betweenness", write, write_property)

    def closeness_centrality(self, distance=None, write=False,
                             write_property=None):
        """Compute (weighted) closeness centrality."""
        distance = (
            self.graph.edge_properties[distance]
            if distance is not None
            else None
        )
        closeness = gt_closeness(
            self.graph, weight=distance)
        return self._dispatch_processing_result(
            closeness, "closeness", write, write_property)
