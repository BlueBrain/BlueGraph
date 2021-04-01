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

    @abstractmethod
    def density(self):
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
