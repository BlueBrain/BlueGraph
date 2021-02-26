from abc import ABC, abstractmethod

from bluegraph.exceptions import BlueGraphException, BlueGraphWarning


# Algos to support
# - Hierarchical clustering
# - Girvan–Newman algorithm
# - Modulatity maximization (Louvain / RenEEL)
# - Statistical inference (stochastic block model)
# - Label propagation


class CommunityDetector(ABC):
    """Abstract class for a community detector."""

    @abstractmethod
    def _run_hierarchical_clustering(self):
        pass

    @abstractmethod
    def _run_louvain(self, weight=None):
        pass

    @abstractmethod
    def _run_girvan_newman(self, weight=None, n_communitites=2,
                           intermediate=False):
        pass

    @abstractmethod
    def _run_stochastic_block_model(self):
        pass

    @abstractmethod
    def _run_label_propagation(self):
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

    def detect_communities(self, strategy="louvain", weight=None,
                           n_communities=2, intermediate=False,
                           write=False, write_property=None, **kwargs):
        """Detect community partition using the input strategy."""
        if strategy == "louvain":
            partition = self._run_louvain(weight=weight)
        elif strategy == "girvan-newman":
            partition = self._run_girvan_newman(
                weight=weight, n_communities=n_communities,
                intermediate=intermediate)
        elif strategy == "lpa":
            partition = self._run_label_propagation(
                weight=weight)
        elif strategy == "hierarchical":
            partition = self._run_hierarchical_clustering(
                weight=weight, n_communities=n_communities, **kwargs)
        else:
            raise CommunityDetector.PartitionError(
                f"Unknown community detection strategy '{strategy}'")
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
