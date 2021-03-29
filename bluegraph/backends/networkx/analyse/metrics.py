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
from bluegraph.core.analyse.metrics import MetricProcessor

import networkx as nx

from ..io import NXGraphProcessor


class NXMetricProcessor(NXGraphProcessor, MetricProcessor):
    """Class for metric processing based on NetworkX graphs."""

    def density(self):
        return nx.density(self.graph)

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