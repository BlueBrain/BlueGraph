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
import numpy as np
import pandas as pd
import pickle

import faiss
import joblib
import os
import warnings

from abc import ABC, abstractmethod

from sklearn.neighbors import BallTree, KDTree, DistanceMetric, VALID_METRICS


from bluegraph.exceptions import BlueGraphException, BlueGraphWarning


# This is to avoid a wierd Faiss segmentation fault (TODO: investigate)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

VALID_METRICS["ball_tree"].append("poincare")


def poincare_distance(v1, v2):
    """Compute Poincare distance between two vectors."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    euclidean_distance = np.linalg.norm(v1 - v2)
    value = 1 + 2 * (
        (euclidean_distance ** 2) / ((1 - v1_norm ** 2) * (1 - v2_norm ** 2))
    )
    return np.arccosh(value)


class SimilarityIndex(ABC):
    """An interface for similarity indices.

    This class specifies an interface for vector similarity indices
    that can be plugged into BlueGraph's SimilarityProcessor
    """
    @abstractmethod
    def add(self, vectors):
        """Add new vectors to the index."""
        pass

    @abstractmethod
    def search(self, vectors, k):
        """Search for k nearest neighbors to the provided vectors."""
        pass

    @abstractmethod
    def reconstruct(self, index):
        """Get a vector by its integer index."""
        pass

    @staticmethod
    @abstractmethod
    def load_index(self, path):
        """Load backend-specific similarity index object."""
        pass

    @staticmethod
    @abstractmethod
    def export_index(self, index, path):
        """Dump backend-specific similarity index object."""
        pass

    def export(self, path, index_path):
        """Dump index object."""
        index = self.index
        self.index = None
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
            self.export_index(index, index_path)
        except Exception as e:
            self.index = index
            raise e
        else:
            self.index = index

    @staticmethod
    def load(path, index_path):
        """Load index object."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj.index = obj.load_index(index_path)
        return obj

    class SimilarityWarning(BlueGraphWarning):
        pass

    class TrainException(BlueGraphException):
        pass

    class SimilarityException(BlueGraphException):
        pass

    class IndexException(BlueGraphException):
        pass

    class QueryException(BlueGraphException):
        pass


class FaissSimilarityIndex(SimilarityIndex):
    """Similarity index based of faiss indices.

    This class allows to build vector similarity indices using
    Faiss. It allows to segment the search space into Voronoi
    cells (see example:
    https://github.com/facebookresearch/faiss/wiki/Faster-search)
    allowing to speed up the search.
    """

    def __init__(self, dimension=None, similarity="euclidean",
                 initial_vectors=None, n_segments=1):

        if similarity not in ["euclidean", "dot", "cosine"]:
            raise SimilarityIndex.SimilarityException(
                f"Unknown similarity measure '{similarity}'")

        self.dimension = dimension
        self.similarity = similarity

        if similarity == "euclidean":
            index = faiss.IndexFlatL2(dimension)
            metric = faiss.METRIC_L2
        elif similarity in ["dot", "cosine"]:
            index = faiss.IndexFlatIP(dimension)
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise SimilarityIndex.SimilarityException(
                f"Similarity metric '{similarity}' is not "
                "implemented for Faiss index")

        if n_segments > 1:
            self.index = faiss.IndexIVFFlat(
                index, dimension, n_segments, metric)

            if initial_vectors is not None:
                self._train(initial_vectors)
        else:
            self.index = index

        if initial_vectors is not None:
            self.add(initial_vectors)

    def _preprocess_vectors(self, vectors):
        if isinstance(vectors, pd.Series):
            vectors = np.array(vectors.to_list())
        else:
            vectors = np.array(vectors)
        vectors = vectors.astype(np.float32)
        if self.similarity == "cosine":
            faiss.normalize_L2(vectors)
        return vectors

    def add(self, vectors):
        vectors = self._preprocess_vectors(vectors)
        if not self.index.is_trained:
            warnings.warn(
                "Faiss segmented index is not trained, training on "
                "the provided vectors",
                SimilarityIndex.SimilarityWarning)
            self._train(vectors)
        self.index.add(vectors)

    def search(self, vectors, k):
        vectors = self._preprocess_vectors(vectors)
        return self.index.search(vectors, int(k))

    def reconstruct(self, index):
        if index >= self.index.ntotal:
            raise SimilarityIndex.SimilarityException(
                f"Point index '{index}' is out of range")
        return self.index.reconstruct(int(index))

    def _train(self, vectors):
        vectors = self._preprocess_vectors(vectors)
        print(self.index)
        self.index.train(vectors)
        self.index.make_direct_map()

    @staticmethod
    def load_index(path):
        return faiss.read_index(path)

    @staticmethod
    def export_index(index, path):
        faiss.write_index(index, path)


class ScikitLearnSimilarityIndex(SimilarityIndex):
    """Similarity index based of scikit-learn indices.

    This class allows to build vector similarity indices using
    scikit-learn. It allows to use various distance metrics
    with KDTrees and BallTrees.
    """
    def __init__(self, dimension, similarity="euclidean",
                 initial_vectors=None, leaf_size=40, index_type="balltree"):

        if initial_vectors is None:
            raise SimilarityIndex.SimilarityException(
                "Initial vectors must be provied (scikit learn "
                "indices are not updatable) ")

        self.dimension = dimension
        self.similarity = similarity

        if index_type == "kdtree":
            self.index = KDTree(
                initial_vectors, leaf_size=leaf_size, metric=similarity)
        elif index_type == "balltree":
            if similarity == "poincare":
                similarity = DistanceMetric.get_metric(
                    'pyfunc', func=poincare_distance)
            self.index = BallTree(
                initial_vectors, leaf_size=leaf_size, metric=similarity)

    def add(self, vectors):
        raise NotImplementedError(
            "Scikit learn indices are not updatable")

    def search(self, vectors, k):
        return self.index.query(vectors, k)

    def reconstruct(self, index):
        vectors, _, _, _ = self.index.get_arrays()
        if len(vectors) <= index:
            raise SimilarityIndex.SimilarityException(
                f"Point index '{index}' is out of range")
        return vectors[index]

    @staticmethod
    def load_index(path):
        return joblib.load(path)

    @staticmethod
    def export_index(index, path):
        joblib.dump(index, path)


class SimilarityProcessor(object):
    """Vector similarity processor.

    This class wraps the indices (names or IDs) of the points,
    vector space and similarity measure configs.
    """

    def __init__(self, similarity_index, point_ids=None):
        self.index = similarity_index
        self.dimension = similarity_index.dimension
        self.similarity = similarity_index.similarity

        if point_ids is None:
            point_ids = pd.Index([])

        self.point_ids = pd.Index(point_ids)

    def info(self):
        info = {
            "similarity": self.similarity,
            "dimension": self.dimension,
            "index_type": type(self.index)
        }
        return info

    def export(self, object_path, index_path):
        index_obj = self.index.index
        self.index.index = None
        try:
            with open(object_path, "wb") as f:
                pickle.dump(self, f)
            self.index.export_index(index_obj, index_path)
        except Exception as e:
            self.index.index = index_obj
            raise e
        else:
            self.index.index = index_obj

    @staticmethod
    def load(object_path, index_path):
        with open(object_path, "rb") as f:
            processor = pickle.load(f)
            processor.index.index = processor.index.load_index(
                index_path)
            return processor

    def get_vectors(self, existing_points):
        """Get vectors for passed point indices."""
        if self.point_ids is not None:
            int_point_indices = self.point_ids.get_indexer(
                existing_points)
        x = []
        for i in int_point_indices:
            if i == -1:
                x.append(None)
            else:
                try:
                    x.append(self.index.reconstruct(i))
                except RuntimeError:
                    x.append(None)
        return x

    def query_existing(self, existing_points, k=10):
        """Query existing points."""
        return self.query_new(self.get_vectors(existing_points), k)

    def query_new(self, vectors, k=10):
        """Query input vectors."""
        # Filter None vectors (for points that were not found
        # in the index)
        non_empty_flag = [
            True if el is not None else False
            for i, el in enumerate(vectors)
        ]
        non_empty_vectors = [
            v for i, v in enumerate(vectors)
            if non_empty_flag[i] is True
        ]
        if len(non_empty_vectors) > 0:
            distance, int_index = self.index.search(non_empty_vectors, k)
        else:
            distance = []
            int_index = []
        # Bring back None vectors
        all_distances = []
        all_indices = []
        non_empty_index = 0
        for flag in non_empty_flag:
            if flag is True:
                all_distances.append(distance[non_empty_index])
                all_indices.append(int_index[non_empty_index])
                non_empty_index += 1
            else:
                all_distances.append(None)
                all_indices.append(None)
        return all_distances, all_indices

    def add(self, vectors, point_ids=None):
        """Add new points to the index."""
        if point_ids is not None:
            # Normalize to pandas index
            point_ids = pd.Index(point_ids)
            new_flag = [
                i not in self.point_ids
                for i in point_ids
            ]
            existing_points = point_ids[[
                not f for f in new_flag]]
            if len(existing_points) > 0:
                warnings.warn(
                    "Points {} already exist in the index, ".format(
                        existing_points) +
                    "ignoring...",
                    SimilarityProcessor.SimilarityProcessorWarning)

            # Add non-existing vectors
            vectors = np.array(vectors)[new_flag]
            new_points = point_ids[new_flag]
            self.point_ids = self.point_ids.append(pd.Index(new_points))
        self.index.add(vectors)

    def get_neighbors(self, vectors=None, point_ids=None,
                      existing_points=None, k=10,
                      add_to_index=False):
        """Get top N similar points."""
        vectors = np.array(vectors)
        if existing_points is not None:
            distance, int_index = self.query_existing(existing_points, k)
        elif vectors is not None:
            if vectors.shape[1] != self.dimension:
                raise SimilarityProcessor.QueryException(
                    "Provided vector does not have a "
                    f"right dimension ({self.dimension})")
            if add_to_index is True:
                if point_ids is None:
                    raise SimilarityProcessor.SimilarityException(
                        "Parameter 'add_to_index' is set to True, "
                        "'vector_indices' must be specified")
                self.add(vectors, point_ids)
            distance, int_index = self.query_new(vectors, k)
        else:
            raise ValueError("here")

        # Get indices
        if self.point_ids is not None:
            indices = [
                (self.point_ids[[x for x in el if x != -1]]
                 if el is not None
                 else None)
                for el in int_index
            ]
        else:
            indices = int_index
        return distance, indices

    class SimilarityProcessorWarning(BlueGraphWarning):
        pass

    class QueryException(BlueGraphException):
        pass

    class SimilarityException(BlueGraphException):
        pass


class NodeSimilarityProcessor(object):
    """Node similarity processor.

    This class allows to build and query node similarity indices using
    Faiss. In wraps the underlying graph object and the vector
    similarity processor and provides.
    """
    def __init__(self, pgframe, vector_property,
                 similarity="euclidean", index_configs=None):
        """Initialize node similarity processor."""
        if index_configs is None:
            index_configs = {
                "backend": "faiss",
                "n_segments": 1
            }

        self.graph = pgframe
        self.vector_property = vector_property
        self.similarity = similarity
        initial_vectors = self.graph.get_node_property_values(
            vector_property).tolist()
        self.dimension = len(initial_vectors[0])

        if index_configs["backend"] == "faiss":
            index = FaissSimilarityIndex(
                self.dimension,
                similarity=self.similarity,
                n_segments=index_configs["n_segments"],
                initial_vectors=initial_vectors)
        elif index_configs["backend"] == "sklearn":
            index = ScikitLearnSimilarityIndex(
                self.dimension,
                similarity=self.similarity,
                leaf_size=(
                    index_configs["leaf_size"]
                    if "leaf_size" in index_configs else None
                ),
                index_type=(
                    index_configs["index_type"]
                    if "index_type" in index_configs else None
                ),
                initial_vectors=initial_vectors)
        else:
            raise BlueGraphException(
                f"Unknown similarity backend `{index_configs['backend']}`, "
                "available backends are `faiss` and `sklearn`")

        self.processor = SimilarityProcessor(
            similarity_index=index, point_ids=self.graph.nodes())

    def get_neighbors(self, input_nodes, k=10):
        neighbors, dist = self.processor.get_neighbors(
            existing_points=input_nodes, k=k)
        result = {}
        for i, ns in enumerate(neighbors):
            result[input_nodes[i]] = dict(zip(ns, dist[i]))
        return result

    def update_similarity_model(self):
        new_nodes = [
            node
            for node in self.graph.nodes()
            if node not in self.processor.index
        ]
        if len(new_nodes) > 0:
            self.processor.add(
                vectors=self.graph.get_node_property_values(
                    self.vector_property).loc[new_nodes].tolist(),
                vector_indices=new_nodes)
