from abc import ABC, abstractmethod


DEFAULT_PARAMS = {
    "batch_size": 50,
    "embedding_dimension": 10,
    "negative_samples": 10,
    "epochs": 12,
    "length": 5,
    "number_of_walks": 4,
    "num_samples": [10, 5]
}


class NodeEmbedder(ABC):
    """Abstract class for a graph embedder."""

    @staticmethod
    @abstractmethod
    def _generate_graph(pgframe):
        """Generate the appropiate graph representation from a PGFrame."""
        pass

    @abstractmethod
    def fit_model(self, **kwargs):
        """Train specified model on the provided graph."""
        pass

    @abstractmethod
    def predict_embeddings(self, graph, batch_size=None, num_samples=None):
        """Predict embeddings of out-sample elements."""
        pass

    # @staticmethod
    # @abstractmethod
    # def evalute_model(test_set, model, **kwargs):
    #     """Train specified model on the provided training set."""
    #     pass

    @abstractmethod
    def save(self, path, compress=True):
        """Save the embedder."""
        pass

    @staticmethod
    @abstractmethod
    def load(path):
        """Load a dumped embedder."""
        pass

    def set_graph(self, pgframe, directed=True, include_type=True,
                  feature_prop=None):
        """Set a graph for the embedding model training."""
        self._graph = self._generate_graph(
            pgframe, directed, include_type, feature_prop)
