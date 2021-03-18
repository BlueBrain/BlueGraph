from abc import (ABC, abstractmethod)


from bluegraph.exceptions import BlueGraphException, BlueGraphWarning


class MetricProcessor(ABC):
    """Abstract class for processing various graph metrics."""

    @classmethod
    def from_graph_object(cls, graph_object):
        """Instantiate a MetricProcessor directly from a Graph object."""
        processor = cls()
        processor.graph = graph_object
        return processor

    @staticmethod
    @abstractmethod
    def _generate_graph(pgframe):
        """Generate the appropiate graph representation from a PGFrame."""
        pass

    @abstractmethod
    def degree_centrality(self, weight=None, write=False):
        """Compute (weighted) degree centrality."""
        pass

    @abstractmethod
    def pagerank_centrality(self, weight=None, write=False):
        """Compute (weighted) PageRank centrality."""
        pass

    @abstractmethod
    def betweenness_centrality(self, distance=None, write=False):
        """Compute (weighted) betweenness centrality."""
        pass

    @abstractmethod
    def closeness_centrality(self, distance, write=False):
        """Compute (weighted) closeness centrality."""
        pass

    @abstractmethod
    def get_pgframe(self):
        """Get a new pgframe object from the wrapped graph object."""
        pass

    def compute_all_node_metrics(self,
                                 degree_weights=None,
                                 pagerank_weights=None,
                                 betweenness_weights=None,
                                 closeness_weights=None):
        if degree_weights is None:
            degree_weights = []
        if pagerank_weights is None:
            pagerank_weights = []
        if betweenness_weights is None:
            betweenness_weights = []
        if closeness_weights is None:
            closeness_weights = []

        results = {
            "degree": {},
            "pagerank": {},
            "betweenness": {},
            "closeness": {}
        }
        for weight in degree_weights:
            results["degree"][weight] = self.degree_centrality(weight)
        for weight in pagerank_weights:
            results["pagerank"][weight] = self.pagerank_centrality(weight)
        for weight in betweenness_weights:
            results["betweenness"][weight] = self.betweenness_centrality(
                weight)
        for weight in closeness_weights:
            results["closeness"][weight] = self.closeness_centrality(
                weight)
        return results

    class MetricProcessingException(BlueGraphException):
        pass

    class MetricProcessingWarning(BlueGraphWarning):
        pass
