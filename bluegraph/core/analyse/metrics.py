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
    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        pass

    @abstractmethod
    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
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

    def _dispatch_processing_result(self, new_property, metric_name,
                                    write=False,
                                    write_property=None):
        if write:
            if write_property is None:
                raise MetricProcessor.MetricProcessingException(
                    "{} processing has the write option set to True, "
                    "the write property name must be specified".format(
                        metric_name.capitalize()))
            self._write_node_property(new_property, write_property)
        else:
            return self._yeild_node_property(new_property)

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
            closeness_weights

        results = {
            "degree": {},
            "pagerank": {},
            "betweenness": {},
            "closeness": {}
        }
        for weight in degree_weights:
            results["degree"][weight] = self.degree_centrality(weight)
        for weight in degree_weights:
            results["pagerank"][weight] = self.pagerank_centrality(weight)
        for weight in degree_weights:
            results["betweenness"][weight] = self.betweenness_centrality(
                weight)
        for weight in degree_weights:
            results["closeness"][weight] = self.closeness_centrality(
                weight)
        return results

    class MetricProcessingException(BlueGraphException):
        pass

    class MetricProcessingWarning(BlueGraphWarning):
        pass
