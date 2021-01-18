#
# Blue Brain Graph is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blue Brain Graph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Blue Brain Graph. If not, see <https://choosealicense.com/licenses/lgpl-3.0/>.

import networkx as nx
import pytest
from kganalytics.paths import top_n_paths, get_cumulative_distance, single_shortest_path, top_n_tripaths
import pandas as pd
from pytest_bdd import when

"""

def test_get_cumulative_weight(paper_comention_network_100_most_frequent):




    shortest_paths = list(
        nx.all_shortest_paths(paper_comention_network_100_most_frequent, "virus",
                              "transcription factors"))
    path_ranking = {}
    for p in shortest_paths:
        if path_condition is None or path_condition(p):
            path_ranking[tuple(p[1:-1])] = get_cumulative_distance(
                graph, p, distance=distance)

    get_cumulative_distance(paper_comention_network_100_most_frequent, p, distance=distance)

    weights = ["frequency"]
    get_cumulative_weight(paper_comention_network_100_most_frequent, path, weights[0])




    tree = get_cumulative_distance(paper_comention_network, weight="distance_npmi")
    weights = ["frequency"]
    degree_centrality = compute_degree_centrality(paper_comention_network_100_most_frequent, weights, 10)

    assert degree_centrality is not None
    assert type(degree_centrality) == dict
    assert len(degree_centrality.keys()) == 1
    assert len(degree_centrality[weights[0]].keys()) == 100
"""
@when("I look for paths from one node that cannot be reached from another one. The path search should fail.")
def test_top_n_paths_no_undirect_paths(paper_comention_network_100_most_frequent):
    with pytest.raises(ValueError):
        top_n_paths(paper_comention_network_100_most_frequent, "virus", "transcription factors", n=10, distance="distance_npmi",
            strategy="naive")

@when("I look for paths using a non supported strategy. The path search should fail.")
def test_top_n_paths_not_supported_strategy_error(paper_comention_network_100_most_frequent):
    with pytest.raises(ValueError):
        top_n_paths(paper_comention_network_100_most_frequent, "a34r", "hsv-1", n=10, distance="distance_npmi",
            strategy="non_supported")

def test_top_n_paths(paper_comention_network_100_most_frequent):

    paths = top_n_paths(paper_comention_network_100_most_frequent, "a34r", "hsv-1", n=10,
                distance="distance_npmi", strategy="naive")

    assert paths is not None
    assert len(paths) == 8
    for path in paths:
        assert isinstance(path, tuple)
        assert len(path) == 3
        assert path[0] == "a34r"
        assert path[2] == "hsv-1"

def test_get_cumulative_distance(paper_comention_network_100_most_frequent):

    distance = "distance_npmi"
    paths = top_n_paths(paper_comention_network_100_most_frequent, "a34r", "hsv-1", n=10,
                        distance=distance, strategy="yen")

    assert paths is not None
    assert len(list(paths)) == 10
    for path in paths:
        cumulative_distance = get_cumulative_distance(paper_comention_network_100_most_frequent, path, distance=distance)
        expected_cumulative_distance = 0
        for i in range(1, len(path)):
            expected_cumulative_distance += paper_comention_network_100_most_frequent.edges[path[i-1], path[i]][distance]

        assert cumulative_distance == expected_cumulative_distance


@when("I compute a path cumulative distance using a non supported distance. The computation should fails.")
def test_get_cumulative_distance_error(paper_comention_network_100_most_frequent):
    
    with pytest.raises(KeyError):
        shortest_paths = list(
            nx.all_shortest_paths(paper_comention_network_100_most_frequent, "a34r", "hsv-1"))
    
        assert shortest_paths is not None
        for path in shortest_paths:
            distance = "non_supported_distance"
            get_cumulative_distance(paper_comention_network_100_most_frequent, path, distance=distance)

def test_top_n_tripaths(paper_comention_network_100_most_frequent):

    a = "a34r"
    b = "infection"
    c = "hsv-1"

    a_b_c_paths = top_n_tripaths(paper_comention_network_100_most_frequent, a, b,c, n=10,
                distance="distance_npmi", strategy="naive", intersecting=True)

    assert a_b_c_paths is not None
    a_b_c_paths = (set(a_b_c_paths[0]), set(a_b_c_paths[1]))

    a_b_paths = top_n_paths(paper_comention_network_100_most_frequent, a, b, n=10,
                        distance="distance_npmi", strategy="naive")

    b_c_paths = top_n_paths(paper_comention_network_100_most_frequent, b, c, n=10,
                            distance="distance_npmi", strategy="naive")

    assert a_b_c_paths == (set(a_b_paths), set(b_c_paths))

    #non intersecting
    a_b_c_paths = top_n_tripaths(paper_comention_network_100_most_frequent, a, b, c, n=10,
                                 distance="distance_npmi", strategy="naive", intersecting=False)

    assert a_b_c_paths is not None
    a_b_c_paths = (set(a_b_c_paths[0]), set(a_b_c_paths[1]))

    b_c_paths = {('infection', 'a34r', 'hsv-1')}

    assert a_b_c_paths == (set(a_b_paths), b_c_paths)
#top_n


