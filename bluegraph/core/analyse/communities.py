from abc import ABC, abstractmethod

from sklearn.cluster import AgglomerativeClustering

from bluegraph.exceptions import BlueGraphException, BlueGraphWarning


# Algos to support
# - Hierarchical clustering (backend-independent)
# - Girvan–Newman algorithm
# - Modulatity maximization (Louvain / RenEEL)
# - Statistical inference (stochastic block model)
# - Label propagation


class CommunityDetector(ABC):
    """Abstract class for a community detector."""

    _strategies = {
        "louvain": "_run_louvain",
        "girvan-newman": "_run_girvan_newman",
        "sbm": "_run_stochastic_block_model",
        "lpa": "_run_label_propagation",
        "hierarchical": "_run_hierarchical_clustering"
    }

    @abstractmethod
    def _run_louvain(self, weight=None, **kwargs):
        pass

    @abstractmethod
    def _run_girvan_newman(self, weight=None, n_communitites=2,
                           intermediate=False):
        pass

    @abstractmethod
    def _run_stochastic_block_model(self, **kwargs):
        pass

    @abstractmethod
    def _run_label_propagation(self, **kwargs):
        pass

    @abstractmethod
    def _compute_modularity(self, partition, weight=None):
        pass

    @abstractmethod
    def _compute_performance(self, partition, weight=None):
        pass

    @abstractmethod
    def _compute_coverage(self, partition, weight=None):
        pass

    def _run_hierarchical_clustering(self, weight=None, n_communities=2,
                                     feature_vectors=None,
                                     feature_vector_prop=None,
                                     linkage="ward",
                                     connectivity=True, **kwargs):
        nodes = self.nodes()
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

    def detect_communities(self, strategy="louvain", weight=None,
                           n_communities=2, intermediate=False,
                           write=False, write_property=None, **kwargs):
        """Detect community partition using the input strategy."""
        if strategy not in CommunityDetector._strategies.keys():
            raise CommunityDetector.PartitionError(
                f"Unknown community detection strategy '{strategy}'")
        partition = getattr(self, CommunityDetector._strategies[strategy])(
            weight=weight, n_communities=n_communities,
            intermediate=intermediate, **kwargs)
        return self._dispatch_processing_result(
            partition, "Community", write, write_property)

    def evaluate_parition(self, partition, metric="modularity", weight=None):
        if metric == "modularity":
            return self._compute_modularity(partition, weight=weight)
        elif metric == "performance":
            return self._compute_performance(partition, weight=weight)
        elif metric == "coverage":
            return self._compute_coverage(partition, weight=weight)
        else:
            raise CommunityDetector.EvaluationError(
                f"Unknown evaluation metric '{metric}'")

    class EvaluationError(BlueGraphException):
        pass

    class PartitionError(BlueGraphException):
        pass

    class EvaluationWarning(BlueGraphWarning):
        pass