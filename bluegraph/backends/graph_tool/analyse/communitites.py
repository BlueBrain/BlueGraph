from bluegraph.core.analyse.communities import CommunityDetector
from ..io import GTGraphProcessor


class GTCommunityDetector(GTGraphProcessor, CommunityDetector):
    """Graph-tool-based community detection interface.

    https://graph-tool.skewed.de/static/doc/demos/inference/inference.html
    """
    def _run_louvain(self, weight=None):
        raise CommunityDetector.PartitionError(
            "Louvain algorithm is not implemented "
            "for Neo4j-based graphs")

    def _run_girvan_newman(self, weight=None, n_communitites=2,
                           intermediate=False):
        raise CommunityDetector.PartitionError(
            "Girvan-Newman algorithm is not implemented "
            "for Neo4j-based graphs")

    def _run_stochastic_block_model(self):
        pass

    def _run_label_propagation(self):
        raise CommunityDetector.PartitionError(
            "Label propagation algorithm is not implemented "
            "for Neo4j-based graphs")

    def _compute_modularity(self, partition, weight=None):
        pass

    def _compute_performance(self, partition, weight=None):
        pass

    def _compute_coverage(self, partition, weight=None):
        pass
