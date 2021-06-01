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
import pytest
import numpy as np

from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.downstream.similarity import (SimilarityProcessor,
                                             NodeSimilarityProcessor)



def test_faiss_similarity_values():
    a = [5, 0]
    b = [0, 5]
    c = [3, 2]
    d = [5, 0.5]

    initial_vectors = np.array([a, b, c, d])

    initial_index = ["a", "b", "c", "d"]

    sim = SimilarityProcessor(
        dimension=2,
        similarity="euclidean",
        initial_vectors=initial_vectors,
        initial_index=initial_index)
    ind, dist = sim.get_similar_points(
        existing_indices=["a", "b"], k=4)

    assert(list(ind[0]) == ["a", "d", "c", "b"])
    assert(dist[0][0] == 0)
    assert(list(ind[1]) == ["b", "c", "d", "a"])
    assert(dist[1][0] == 0)

    sim = SimilarityProcessor(
        dimension=2,
        similarity="cosine",
        initial_vectors=initial_vectors,
        initial_index=initial_index)
    ind, dist = sim.get_similar_points(
        existing_indices=["a", "b"], k=4)

    assert(list(ind[0]) == ["a", "d", "c", "b"])
    assert(dist[0][0] == 1.0)
    assert(list(ind[1]) == ["b", "c", "d", "a"])
    assert(dist[1][0] == 1.0)

    sim = SimilarityProcessor(
        dimension=2,
        similarity="dot",
        initial_vectors=initial_vectors,
        initial_index=initial_index)
    ind, dist = sim.get_similar_points(
        existing_indices=["a", "b"], k=4)

    assert(list(ind[0]) == ["d", "a", "c", "b"])
    assert(dist[0][0] == 25.0)
    assert(list(ind[1]) == ["b", "c", "d", "a"])
    assert(dist[1][0] == 25.0)


def test_plain_faiss_similarity():
    a = [5, 0]
    b = [0, 5]
    c = [3, 2]
    d = [5, 0.5]

    initial_vectors = np.array([a, b, c, d])

    initial_index = ["a", "b", "c", "d"]

    sim = SimilarityProcessor(
        dimension=2,
        similarity="euclidean",
        initial_vectors=initial_vectors,
        initial_index=initial_index)

    # Test the interface
    sim.info()
    vectors = sim.get_vectors(["a", "b"])
    assert(len(vectors) == 2)
    sim.query_existing(existing_indices=["a", "b"])
    sim.query_new(vectors=[[0, 0], [1, 1]])
    sim.add(vectors=[[0, 0], [1, 1]], vector_indices=["e", "f"])
    assert("e" in sim.index and "f" in sim.index)

    with pytest.raises(SimilarityProcessor.SimilarityException):
        sim.get_similar_points(
            vectors=[[0.1, 1], [0.7, 0.25]],
            # vector_indices=None,
            k=10, add_to_index=True)

    sim.get_similar_points(
        vectors=[[0.1, 1], [0.7, 0.25]],
        vector_indices=["g", "h"],
        k=10, add_to_index=True)
    assert("g" in sim.index and "h" in sim.index)

    assert("not found" not in sim.index)
    ind, dist = sim.get_similar_points(
        existing_indices=["g", "not found", "h", "not found"])
    assert(ind[1] is None)
    assert(ind[3] is None)


def test_segmented_faiss_similarity():
    N = 10000
    n_segments = 10
    vectors = np.random.rand(N, 2)
    vectors = np.array(vectors).astype(np.float32)

    # Initial vectors specified
    sim = SimilarityProcessor(
        dimension=2,
        similarity="euclidean",
        initial_vectors=vectors,
        initial_index=[
            f"element{ n + 1 }" for n in range(N)],
        n_segments=n_segments)
    ind, dist = sim.get_similar_points(
        existing_indices=["element4", "element5"], k=4)

    # Initial vectors not specified
    sim = SimilarityProcessor(
        dimension=2,
        similarity="euclidean",
        # initial_vectors=vectors,
        # initial_index=[
        #     f"element{ n + 1 }" for n in range(N)],
        n_segments=n_segments)
    sim.add(
        vectors=vectors,
        vector_indices=[f"element{ n + 1 }" for n in range(N)])

    # Test the interface
    sim.info()
    vectors = sim.get_vectors(["element1", "element2"])
    assert(len(vectors) == 2)
    sim.query_existing(existing_indices=["element1", "element2"])
    sim.query_new(vectors=[[0, 0], [1, 1]])

    sim.add(vectors=[[0, 0], [1, 1]], vector_indices=["new1", "new2"])
    assert("new1" in sim.index and "new2" in sim.index)

    with pytest.raises(SimilarityProcessor.SimilarityException):
        sim.get_similar_points(
            vectors=[[0.1, 1], [0.7, 0.25]],
            # vector_indices=None,
            k=10, add_to_index=True)

    sim.get_similar_points(
        vectors=[[0.1, 1], [0.7, 0.25]],
        vector_indices=["g", "h"],
        k=10, add_to_index=True)
    assert("g" in sim.index and "h" in sim.index)
    sim.export("test_sim_proc.pkl", "test_sim_proc.faiss")
    el = SimilarityProcessor.load(
        "test_sim_proc.pkl", "test_sim_proc.faiss")


def test_node_similarity(random_pgframe):
    random_pgframe.rename_nodes({
        n: str(n)
        for n in random_pgframe.nodes()
    })

    node2vec_embedder = StellarGraphNodeEmbedder(
        "node2vec", edge_weight="mi",
        embedding_dimension=10, length=5, number_of_walks=10)
    node2vec_embedding = node2vec_embedder.fit_model(random_pgframe)
    random_pgframe.add_node_properties(
        node2vec_embedding.rename(columns={"embedding": "node2vec"}))
    node2vec_l2 = NodeSimilarityProcessor(
        random_pgframe, "node2vec", similarity="euclidean")
    node2vec_cosine = NodeSimilarityProcessor(
        random_pgframe, "node2vec", similarity="cosine")
    similar = node2vec_l2.get_similar_nodes(["0", "1"], k=10)
    assert("0" in similar)
    assert("1" in similar)
    assert(len(similar["0"]) == 10 and len(similar["1"]) == 10)
    similar = node2vec_cosine.get_similar_nodes(["0", "1"], k=10)
    assert("0" in similar)
    assert("1" in similar)
    assert(len(similar["0"]) == 10 and len(similar["1"]) == 10)
