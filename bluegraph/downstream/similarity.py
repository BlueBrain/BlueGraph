import numpy as np

from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityProcessor(object):

    def _compute_similarity_matrix(self, x, y=None):
        if self.similarity == "cosine":
            return cosine_similarity(x, y)
        else:
            raise ValueError(
                f"Unknown similarity type '{self.simlarity}'")

    def __init__(self, embeddings, similarity="cosine", **kwargs):
        self.embeddings = embeddings

        if isinstance(embeddings["embedding"].iloc[0], list):
            self.dimension = len(embeddings["embedding"].iloc[0])
        elif isinstance(embeddings["embedding"].iloc[0], np.ndarray):
            self.dimension = embeddings["embedding"].iloc[0].shape[0]

        points = embeddings["embedding"].tolist()
        self.similarity = similarity
        self.similarity_params = kwargs
        self.similarity_matrix = self._compute_similarity_matrix(points)

    def get_similar_points(self, vector=None, existing_index=None, k=10,
                           add_to_index=False, new_point_index=None):
        if existing_index is not None:
            existing_integer_index = self.embeddings.index.get_loc(
                existing_index)
        else:
            if vector.shape[0] != self.dimension:
                raise ValueError(
                    f"Provided vector does not have a right dimension ({self.dimension})")
            new_similarities = self._compute_similarity_matrix(
                self.embeddings["embedding"].tolist(), [vector])
            existing_integer_index = 0

            if add_to_index is True:
                if new_point_index is None:
                    raise ValueError(
                        "Parameter 'add_to_index' is set to True, "
                        "'new_point_index' must be specified")
                self.embeddings.loc[new_point_index, "embedding"] = vector
                # Add a new vector along axis 1
                new_similarity_matrix = np.concatenate(
                    [self.similarity_matrix, new_similarities], axis=1)

                # Add a new vector along axis 0
                self.similarity_matrix = np.concatenate([
                    new_similarity_matrix,
                    np.transpose(
                        np.concatenate(
                            [new_similarities, np.ones((1, 1))], axis=0))
                ])

        similar_indices = self.similarity_matrix[
            existing_integer_index].argsort()[-k:]  # [::-1]
        return dict(zip(
            self.embeddings.iloc[similar_indices].index,
            self.similarity_matrix[existing_integer_index][similar_indices]
        ))


class ClosenessProcessor(object):

    def __init__(self, embeddings, metric="minkowski", **kwargs):
        self.embeddings = embeddings
        points = embeddings["embedding"].tolist()
        self.metric = metric
        self.distance_params = kwargs
        self.tree = KDTree(points, metric=metric, **kwargs)

    def get_closest_points(self, vector=None, existing_index=None, k=10):
        if vector is None and existing_index is not None:
            vector = self.embeddings.loc[existing_index]["embedding"]
        dist, ind = self.tree.query([vector], k=10)
        return dict(zip(list(self.embeddings.index[ind[0]]), dist[0].tolist()))
