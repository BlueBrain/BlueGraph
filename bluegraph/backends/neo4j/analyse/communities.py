from bluegraph.core.analyse.communities import CommunityDetector
from ..io import Neo4jGraphProcessor


class Neo4jCommunityDetector(Neo4jGraphProcessor, CommunityDetector):
    """Neo4j-based community detection interface.

    https://neo4j.com/docs/graph-data-science/current/algorithms/community/
    - Louvain
    - Label Propagation
    """
    def _run_louvain(self, weight=None):
        pass

    def _run_girvan_newman(self, weight=None, n_communitites=2,
                           intermediate=False):
        raise CommunityDetector.PartitionError(
            "Girvan-Newman algorithm is not implemented "
            "for Neo4j-based graphs")

    def _run_stochastic_block_model(self):
        raise CommunityDetector.PartitionError(
            "Stochastic block model is not implemented "
            "for Neo4j-based graphs")

    def _run_label_propagation(self):
        pass

    def _compute_modularity(self, partition, weight=None):
        pass

    def _compute_performance(self, partition, weight=None):
        pass

    def _compute_coverage(self, partition, weight=None):
        pass
