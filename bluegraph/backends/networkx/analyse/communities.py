import itertools
from operator import itemgetter
from functools import partial

import community as community_louvain
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.label_propagation import asyn_lpa_communities
from networkx.algorithms.community.quality import (performance,
                                                   coverage)

from sklearn.cluster import AgglomerativeClustering

from bluegraph.core.analyse.communities import CommunityDetector
from ..io import NXGraphProcessor


def _get_community_sets(partition):
    communities = set(partition.values())
    community_sets = []
    for c in communities:
        nodes = [
            k for k, v in partition.items() if v == c
        ]
        community_sets.append(nodes)
    return community_sets


def heaviest(weight, G):
    u, v, w = max(G.edges(data="weight"), key=itemgetter(2))
    return (u, v)


def community_sets_to_dict(communities, nodes):
    partition = {}
    for i, community in enumerate(communities):
        members = itemgetter(*community)(nodes)
        try:
            len(members)
        except TypeError:
            members = [members]
        partition.update({n: i for n in members})
    return partition


class NXCommunityDetector(NXGraphProcessor, CommunityDetector):
    """NetworkX-based community detection interface."""

    def _run_louvain(self, weight=None):
        """Detect node communities using Louvain algo."""
        weight = "weight" if weight is None else weight
        partition = community_louvain.best_partition(
            self.graph, weight=weight)
        return partition

    def _run_girvan_newman(self, weight=None, n_communities=2,
                           intermediate=False):
        most_valuable_edge = (
            partial(heaviest, weight) if weight is not None
            else None
        )
        communities = girvan_newman(
            self.graph, most_valuable_edge=most_valuable_edge)

        layered_communities = list(itertools.takewhile(
            lambda c: len(c) <= n_communities, communities))

        nodes = list(self.graph.nodes())
        if intermediate is False:
            # take the last iteration of the algo
            partition = community_sets_to_dict(
                layered_communities[-1], nodes)
        else:
            # take all the interation of the algo
            partition = {n: [] for n in nodes}
            for layer in layered_communities:
                for i, community in enumerate(layer):
                    for el in community:
                        partition[el].append(i)
        return partition

    def _run_hierarchical_clustering(self, weight=None, n_communities=2,
                                     feature_vectors=None,
                                     feature_vector_prop=None,
                                     linkage="ward",
                                     connectivity=True):
        nodes = list(self.graph.nodes())
        if feature_vectors is None:
            if feature_vector_prop is None:
                raise ValueError()
            feature_vectors = self._get_node_property_values(
                feature_vector_prop, nodes)

        if connectivity is True:
            connectivity_matrix = self._get_adjacency_matrix(
                nodes, weight=weight)
        model = AgglomerativeClustering(linkage=linkage,
                                        connectivity=connectivity_matrix,
                                        n_clusters=n_communities)
        model.fit(feature_vectors)
        clusters = model.labels_
        return {n: clusters[i] for i, n in enumerate(nodes)}

    def _run_stochastic_block_model(self):
        raise CommunityDetector.PartitionError(
            "Stochastic block model is not implemented "
            "for NetworkX-based graphs")

    def _run_label_propagation(self, weight=None):
        communities = asyn_lpa_communities(self.graph, weight=weight)
        return community_sets_to_dict(
            communities, list(self.graph.nodes()))

    def _compute_modularity(self, partition, weight=None):
        return community_louvain.modularity(
            partition, self.graph, weight=weight)

    def _compute_performance(self, partition, weight=None):
        return performance(self.graph, _get_community_sets(partition))

    def _compute_coverage(self, partition, weight=None):
        return coverage(self.graph, _get_community_sets(partition))
