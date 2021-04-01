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

        self._model = self._initialize_model(initial_vectors)

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
        if isinstance(vectors, pd.Series):
            vectors = np.array(vectors.to_list())
        elif not isinstance(vectors, np.ndarray):
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

            if initial_vectors is None:
                SimilarityProcessor.TrainException(
                    "Initial vectors should be specified for "
                    "Faiss segmented indexer "
                )

            initial_vectors = self._preprocess_vectors(initial_vectors)
            model.train(initial_vectors)
            model.make_direct_map()
        else:
            model = index
        return model

    def get_vectors(self, existing_indices):
        if self.index is not None:
            existing_indices = self.index.get_indexer(existing_indices)
        x = [self._model.reconstruct(int(i)) for i in existing_indices]
        return x

    def query_existing(self, existing_indices, k=10):
        return self.query_new(self.get_vectors(existing_indices), k)

    def query_new(self, vectors, k=10):
        vectors = self._preprocess_vectors(vectors)
        distance, int_index = self._model.search(vectors, k)
        return distance, int_index

    def add(self, vectors, vector_indices=None):
        vectors = self._preprocess_vectors(vectors)
        if vector_indices is not None:
            for i in vector_indices:
                if i in self.index:
                    raise SimilarityProcessor.IndexException(
                        "Index '{}' already exists".format(i))
            self.index = self.index.append(pd.Index(vector_indices))

        self._model.add(vectors)

    def get_similar_points(self, vectors=None, vector_indices=None,
                           existing_indices=None, k=10,
                           add_to_index=False, new_point_index=None):
        if existing_indices is not None:
            distance, int_index = self.query_existing(existing_indices, k)
        else:
            if vectors.shape[1] != self.dimension:
                raise SimilarityProcessor.QueryException(
                    "Provided vector does not have a "
                    f"right dimension ({self.dimension})")
            if add_to_index is True:
                if new_point_index is None:
                    raise ValueError(
                        "Parameter 'add_to_index' is set to True, "
                        "'new_point_index' must be specified")
                self.add(vectors, vector_indices)
            distance, int_index = self.query_new(vectors, k)

        # Get indices
        if self.index is not None:
            indices = list(map(lambda x: self.index[x], int_index))
        else:
            indices = int_index
        return indices, distance

    class TrainException(Exception):
        pass

    class SimilarityException(Exception):
        pass

    class IndexException(Exception):
        pass

    class QueryException(Exception):
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
