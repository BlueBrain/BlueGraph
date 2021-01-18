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

import pytest
from kganalytics.metrics import (compute_degree_centrality,
                                 compute_pagerank_centrality,
                                 compute_betweenness_centrality,
                                 detect_communities
                                 )

import pandas as pd

def test_compute_degree_centrality(paper_comention_network_100_most_frequent):

    weights = ["frequency"]
    degree_centrality = compute_degree_centrality(paper_comention_network_100_most_frequent, weights, 10)

    assert degree_centrality is not None
    assert type(degree_centrality) == dict
    assert len(degree_centrality.keys()) == 1
    assert len(degree_centrality[weights[0]].keys()) == 100


def test_compute_pagerank_centrality(paper_comention_network_100_most_frequent):

    weights = ["frequency"]
    pagerank_centrality = compute_pagerank_centrality(paper_comention_network_100_most_frequent,  weights, 10)

    assert pagerank_centrality is not None
    assert type(pagerank_centrality) == dict
    assert len(pagerank_centrality.keys()) == 1
    assert len(pagerank_centrality[weights[0]].keys()) == 100


def test_compute_betweenness_centrality(paper_comention_network_100_most_frequent):
    weights = ["distance_ppmi","distance_npmi"]
    betweenness_centrality = compute_betweenness_centrality(paper_comention_network_100_most_frequent, weights, 20)
    assert betweenness_centrality is not None
    assert type(betweenness_centrality) == dict
    assert len(betweenness_centrality.keys()) == 2
    assert len(betweenness_centrality[weights[0]].keys()) == 100

def test_compute_betweenness_centrality_node_attributes(paper_comention_network_100_most_frequent):

    _ = detect_communities(paper_comention_network_100_most_frequent, weight="frequency", set_attr="community")
    for node in paper_comention_network_100_most_frequent.nodes(data=True):
        assert "community" in node[1]
        assert isinstance(node[1], dict)
        assert isinstance(node[1]["community"], int)
