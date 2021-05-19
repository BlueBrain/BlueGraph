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
import os
import warnings

from bluegraph.exceptions import BlueGraphException, BlueGraphWarning


from bluegraph.exceptions import BlueGraphException


# This is to avoid a wierd Faiss segmentation fault (TODO: investigate)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SimilarityProcessor(object):
    """Vector similarity processor.

    This class allows to build vector similarity indices using
    Faiss. In wraps the indices (names or IDs) of the points,
    vector space and similarity measure configs. It also allows
    to segment the search space into Voronoi cells (see example:
    https://github.com/facebookresearch/faiss/wiki/Faster-search)
    allowing to speed up the search.
    """

    def __init__(self, dimension, similarity="euclidean",
                 initial_vectors=None, initial_index=None,
                 n_segments=1):
        if similarity not in ["euclidean", "dot", "cosine"]:
            raise SimilarityProcessor.SimilarityException(
                f"Unknown similarity measure '{similarity}'")

        self.dimension = dimension
        self.similarity = similarity
        self.n_segments = n_segments

        self.index = pd.Index([])

        self._initialize_model(initial_vectors)

        if initial_vectors is not None:
            self.add(initial_vectors, initial_index)

    def info(self):
        info = {
            "similarity": self.similarity,
            "dimension": self.dimension,
            "segmented": True if self.n_segments and self.n_segments > 1 else False,
        }
        return info

    def export(self, object_path, index_path):
        model = self._model
        self._model = None
        with open(object_path, "wb") as f:
            pickle.dump(self, f)
        faiss.write_index(model, index_path)
        self._model = model

    @staticmethod
    def load(object_path, index_path):
        with open(object_path, "rb") as f:
            processor = pickle.load(f)
            processor._model = faiss.read_index(index_path)
            return processor

    def _preprocess_vectors(self, vectors):
        not_empty_flag = [
            True if el is not None else False
            for i, el in enumerate(vectors)
        ]
        if isinstance(vectors, pd.Series):
            vectors = np.array(vectors.to_list())
        else:
            vectors = np.array(vectors)
        vectors = vectors.astype(np.float32)
        if self.similarity == "cosine":
            faiss.normalize_L2(vectors)
        return vectors

    def _initialize_model(self, initial_vectors=None):
        if self.similarity == "euclidean":
            index = faiss.IndexFlatL2(self.dimension)
            metric = faiss.METRIC_L2
        elif self.similarity in ["dot", "cosine"]:
            index = faiss.IndexFlatIP(self.dimension)
            metric = faiss.METRIC_INNER_PRODUCT

        if self.n_segments > 1:
            model = faiss.IndexIVFFlat(
                index, self.dimension, self.n_segments, metric)

            if initial_vectors is not None:
                initial_vectors = self._preprocess_vectors(initial_vectors)
                model.train(initial_vectors)
                model.make_direct_map()
        else:
            model = index
        self._model = model

    def get_vectors(self, existing_indices):
        """Get vectors for passed point indices."""
        if self.index is not None:
            int_indices = self.index.get_indexer(existing_indices)
        x = []
        for i in int_indices:
            if i == -1:
                x.append(None)
            else:
                try:
                    x.append(self._model.reconstruct(int(i)))
                except RuntimeError:
                    x.append(None)
        return x

    def query_existing(self, existing_indices, k=10):
        """Query existing points."""
        return self.query_new(self.get_vectors(existing_indices), k)

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
            vectors = self._preprocess_vectors(non_empty_vectors)
            distance, int_index = self._model.search(vectors, k)
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

    def add(self, vectors, vector_indices=None):
        """Add new points to the index."""
        vectors = self._preprocess_vectors(vectors)
        if not self._model.is_trained:
            warnings.warn(
                "Similarity index is not trained, training on "
                "the provided vectors",
                SimilarityProcessor.SimilarityWarning)

            self._model.train(vectors)
            self._model.make_direct_map()
            if vector_indices is not None:
                self.index = self.index.append(pd.Index(vector_indices))
        else:
            if vector_indices is not None:
                # Normalize to pandas index
                vector_indices = pd.Index(vector_indices)
                new_flag = [
                    i not in self.index
                    for i in vector_indices
                ]
                existing_indices = vector_indices[[
                    not f for f in new_flag]]
                if len(existing_indices) > 0:
                    warnings.warn(
                        "Points {} already exist in the index, ".format(
                            existing_indices) +
                        "ignoring...",
                        SimilarityProcessor.SimilarityWarning)

                # Add non-existing vectors
                vectors = vectors[new_flag]
                new_indices = vector_indices[new_flag]
                self.index = self.index.append(pd.Index(new_indices))
        self._model.add(vectors)

    def get_similar_points(self, vectors=None, vector_indices=None,
                           existing_indices=None, k=10,
                           add_to_index=False):
        """Get top N similar points."""
        if existing_indices is not None:
            distance, int_index = self.query_existing(existing_indices, k)
        elif vectors is not None:
            vectors = self._preprocess_vectors(vectors)
            if vectors.shape[1] != self.dimension:
                raise SimilarityProcessor.QueryException(
                    "Provided vector does not have a "
                    f"right dimension ({self.dimension})")
            if add_to_index is True:
                if vector_indices is None:
                    raise SimilarityProcessor.SimilarityException(
                        "Parameter 'add_to_index' is set to True, "
                        "'vector_indices' must be specified")
                self.add(vectors, vector_indices)
            distance, int_index = self.query_new(vectors, k)

        # Get indices
        if self.index is not None:
            indices = [
                (self.index[[x for x in el if x != -1]]
                 if el is not None
                 else None)
                for el in int_index
            ]
        else:
            indices = int_index
        return indices, distance

    class TrainException(BlueGraphException):
        pass

    class SimilarityException(BlueGraphException):
        pass

    class SimilarityWarning(BlueGraphWarning):
        pass

    class IndexException(BlueGraphException):
        pass

    class QueryException(BlueGraphException):
        pass


class NodeSimilarityProcessor(object):
    """Node similarity processor.

    This class allows to build and query node similarity indices using
    Faiss. In wraps the underlying graph object and the vector
    similarity processor and provides.
    """

    def __init__(self, pgframe, vector_property, similarity="euclidean"):
        self.graph = pgframe
        self.vector_property = vector_property
        self.similarity = similarity
        initial_vectors = self.graph.get_node_property_values(
            vector_property).tolist()
        self.dimension = len(initial_vectors[0])

        self.processor = SimilarityProcessor(
            dimension=self.dimension,
            similarity=self.similarity,
            initial_vectors=initial_vectors,
            initial_index=self.graph.nodes())

    def get_similar_nodes(self, input_nodes, k=10):
        neighbors, dist = self.processor.get_similar_points(
            existing_indices=input_nodes, k=k)
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
            self.processor._add(
                vectors=self.graph.get_node_property_values(
                    self.vector_property).loc[new_nodes].tolist(),
                vector_indices=new_nodes)
