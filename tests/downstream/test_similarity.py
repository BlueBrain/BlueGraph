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
from bluegraph.downstream.similarity import SimilarityProcessor


def test_faiss_similarity():
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

    # N = 10000
    # n_segments = 100
    # vectors = np.random.rand(N, 2)
    # vectors = np.array(vectors).astype(np.float32)

    # sim = SimilarityProcessor(
    #     dimension=2,
    #     similarity="euclidean",
    #     initial_vectors=vectors,
    #     initial_index=[
    #         f"element{ n + 1 }" for n in range(N)],
    #     n_segments=n_segments)
    # ind, dist = sim.get_similar_points(
    #     existing_indices=["element4", "element5"], k=4)

