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
from bluegraph.downstream.similarity import (ScikitLearnSimilarityIndex,
                                             FaissSimilarityIndex,
                                             SimilarityProcessor,
                                             NodeSimilarityProcessor)


FAISS_INDEX_TEST_DATA = [
    ("euclidean", 1, True),
    ("euclidean", 1, False),
    ("euclidean", 10, True),
    ("euclidean", 10, False),
    ("cosine", 1, True),
    ("cosine", 10, False)
]

SKLEARN_INDEX_TEST_DATA = [
    ("euclidean"),
    ("poincare")
]


def test_faiss_similarity_values():
    a = [5, 0]
    b = [0, 5]
    c = [3, 2]
    d = [5, 0.5]

    initial_vectors = np.array([a, b, c, d])

    initial_index = ["a", "b", "c", "d"]

    index = FaissSimilarityIndex(
        dimension=2,
        similarity="euclidean",
        initial_vectors=initial_vectors)

    sim = SimilarityProcessor(index, point_ids=initial_index)
    dist, ind = sim.get_neighbors(
        existing_points=["a", "b"], k=4)

    assert(list(ind[0]) == ["a", "d", "c", "b"])
    assert(dist[0][0] == 0)
    assert(list(ind[1]) == ["b", "c", "d", "a"])
    assert(dist[1][0] == 0)

    index = FaissSimilarityIndex(
        dimension=2,
        similarity="cosine",
        initial_vectors=initial_vectors)
    sim = SimilarityProcessor(
        index, point_ids=initial_index)
    dist, ind = sim.get_neighbors(
        existing_points=["a", "b"], k=4)

    assert(list(ind[0]) == ["a", "d", "c", "b"])
    assert(dist[0][0] == 1.0)
    assert(list(ind[1]) == ["b", "c", "d", "a"])
    assert(dist[1][0] == 1.0)

    index = FaissSimilarityIndex(
        dimension=2,
        similarity="dot",
        initial_vectors=initial_vectors)
    sim = SimilarityProcessor(
        index, point_ids=initial_index)
    dist, ind = sim.get_neighbors(
        existing_points=["a", "b"], k=4)

    assert(list(ind[0]) == ["d", "a", "c", "b"])
    assert(dist[0][0] == 25.0)
    assert(list(ind[1]) == ["b", "c", "d", "a"])
    assert(dist[1][0] == 25.0)


@pytest.mark.parametrize("similarity,n_segments,init", FAISS_INDEX_TEST_DATA)
def test_faiss_similarity_interface(similarity, n_segments, init):
    d = 64
    initial_vectors = np.random.rand(100, 64)
    new_vectors = np.random.rand(20, 64)
    index = FaissSimilarityIndex(
        d, similarity=similarity, n_segments=n_segments,
        initial_vectors=initial_vectors if init is True else None)
    index.add(new_vectors)
    vector = index.reconstruct(0)
    scores, points = index.search([vector], 10)
    index.export("faiss.pkl", "faiss.faiss")
    new_index = FaissSimilarityIndex.load("faiss.pkl", "faiss.faiss")
    scores, points = new_index.search([vector], 10)


@pytest.mark.parametrize("similarity", SKLEARN_INDEX_TEST_DATA)
def test_sklearn_similarity_interface(similarity):
    d = 64
    initial_vectors = np.random.rand(100, 64)
    new_vectors = np.random.rand(20, 64)

    index = ScikitLearnSimilarityIndex(
        d, similarity=similarity,
        initial_vectors=initial_vectors)

    vector = index.reconstruct(0)
    scores, points = index.search([vector], 10)
    vector = new_vectors[0]
    index.search([vector], 10)
    index.export("sklearn.pkl", "sklearn.joblib")
    new_index = ScikitLearnSimilarityIndex.load("sklearn.pkl", "sklearn.joblib")
    scores, points = new_index.search([vector], 10)


def test_processors():
    d = 64
    initial_vectors = np.random.rand(100, 64)
    point_names = [f"point{el}" for el in range(100)]

    indices = [
        FaissSimilarityIndex(
            64, similarity="euclidean", n_segments=10,
            initial_vectors=initial_vectors),
        ScikitLearnSimilarityIndex(
            64, similarity="poincare", leaf_size=20,
            initial_vectors=initial_vectors)
    ]
    for index in indices:
        processor = SimilarityProcessor(index, point_ids=point_names)
        processor.info()
        processor.export("index.pkl", "index")
        processor = SimilarityProcessor.load("index.pkl", "index")
        vectors = processor.get_vectors(point_names[:10])
        assert(len(vectors) == 10)
        res = processor.query_existing(point_names[:10], 20)
        res = processor.query_new(vectors, 20)
        new_vectors = np.random.rand(20, processor.dimension)
        try:
            processor.add(new_vectors,
                          point_ids=[f"random_point{i + 1}" for i in range(20)])
            assert(
                "random_point1" in processor.point_ids and
                "random_point2" in processor.point_ids)
        except NotImplementedError:
            pass
        res = processor.get_neighbors(
            vectors=new_vectors, k=10)
        res = processor.get_neighbors(
            existing_points=point_names[:10], k=10)

        ind, dist = processor.get_neighbors(
            existing_points=["point0", "a", "b"], k=10)
        assert(ind[0] is not None)
        assert(ind[1] is None)
        assert(ind[2] is None)

        try:
            new_vectors = np.random.rand(20, processor.dimension)
            res = processor.get_neighbors(
                vectors=new_vectors, k=10,
                point_ids=[f"new_random_point{i + 1}" for i in range(20)],
                add_to_index=True)
        except NotImplementedError as e:
            print(e)


def test_node_similarity(random_pgframe):
    random_pgframe.rename_nodes({
        n: str(n)
        for n in random_pgframe.nodes()
    })

    # create embedding vectors
    node2vec_embedder = StellarGraphNodeEmbedder(
        "node2vec", edge_weight="mi",
        embedding_dimension=10, length=5, number_of_walks=10)
    node2vec_embedding = node2vec_embedder.fit_model(random_pgframe)
    random_pgframe.add_node_properties(
        node2vec_embedding.rename(columns={"embedding": "node2vec"}))

    # test node similarity
    index_configs = {
        "backend": "faiss",
        "n_segments": 2
    }
    node2vec_l2 = NodeSimilarityProcessor(
        random_pgframe, vector_property="node2vec", similarity="euclidean",
        index_configs=index_configs)
    node2vec_cosine = NodeSimilarityProcessor(
        random_pgframe, vector_property="node2vec", similarity="cosine",
        index_configs=index_configs)
    node2vec_poincare = NodeSimilarityProcessor(
        random_pgframe, vector_property="node2vec", similarity="poincare",
        index_configs={
            "backend": "sklearn", "index_type": "balltree", "leaf_size": 10
        })

    similar = node2vec_l2.get_neighbors(["0", "1"], k=10)
    assert("0" in similar)
    assert("1" in similar)
    assert(len(similar["0"]) == 10 and len(similar["1"]) == 10)
    similar = node2vec_cosine.get_neighbors(["0", "1"], k=10)
    assert("0" in similar)
    assert("1" in similar)
    assert(len(similar["0"]) == 10 and len(similar["1"]) == 10)
    similar = node2vec_poincare.get_neighbors(["0", "1"], k=10)
    assert("0" in similar)
    assert("1" in similar)
    assert(len(similar["0"]) == 10 and len(similar["1"]) == 10)
