from abc import ABC


class CommunityDetector(ABC):
    """Abstract class for a community detector."""

    def detect_communities(self, strategy="louvain",
                           write=False, write_property=None):
        """Detect community partition using the input strategy."""
        pass

    def compute_modularity(self, partition):
        pass
