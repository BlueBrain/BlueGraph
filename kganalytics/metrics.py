# Copyright (c) 2020–2021, EPFL/Blue Brain Project
#
# Blue Graph is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blue Graph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Blue Graph. If not, see <https://choosealicense.com/licenses/lgpl-3.0/>.

"""Collection of utils for computing various network metrics."""
import networkx as nx

import community as community_louvain

from kganalytics.utils import top_n, keys_by_value


def compute_degree_centrality(graph, weights, print_top_n_nodes=None):
    """Compute degree centralities and add as node attrs."""
    degree_centrality = {}
    for w in weights:
        degree_centrality[w] = dict(graph.degree(weight=w))
        nx.set_node_attributes(
            graph,
            degree_centrality[w], "{}_{}".format("degree", w))
        if print_top_n_nodes is not None:
            print("Top n nodes by {}:".format(w))
            for entity in top_n(degree_centrality[w], print_top_n_nodes):
                print("\t{} ({})".format(entity, degree_centrality[w][entity]))
            print()
    return degree_centrality


def compute_pagerank_centrality(graph, weights, print_top_n_nodes=None):
    """Compute PageRank centralities and add as node attrs."""
    pagerank_centrality = {}
    for w in weights:
        pagerank_centrality[w] = nx.pagerank(graph, weight=w)
        nx.set_node_attributes(
            graph,
            pagerank_centrality[w], "{}_{}".format("pagerank", w))
        if print_top_n_nodes is not None:
            print("Top n nodes by {}:".format(w))
            for entity in top_n(pagerank_centrality[w], print_top_n_nodes):
                print("\t{} ({:.2f})".format(entity, pagerank_centrality[w][entity]))
            print()
    return pagerank_centrality


def compute_betweenness_centrality(graph, weights, print_top_n_nodes=None):
    """Compute PageRank centralities and add as node attrs."""
    betweenness_centrality = {}
    for w in weights:
        betweenness_centrality[w] = nx.betweenness_centrality(graph, weight=w)
        nx.set_node_attributes(
            graph,
            betweenness_centrality[w], "{}_{}".format("betweenness", w))
        if print_top_n_nodes is not None:
            print("Top n nodes by {}:".format(w))
            for entity in top_n(betweenness_centrality[w], print_top_n_nodes):
                print("\t{} ({})".format(
                    entity, betweenness_centrality[w][entity]))
            print()
    return betweenness_centrality


def detect_communities(graph, weight="frequency", set_attr=None):
    """Detect node communities using Louvain algo."""
    print("Detecting communities...")
    partition = community_louvain.best_partition(
        graph, weight=weight)
    modularity = community_louvain.modularity(
        partition, graph, weight=weight)
    print("Best network partition:")
    print("\t Number of communities:", len(set(partition.values())))
    print("\t Modularity:", modularity)

    if set_attr:
        nx.set_node_attributes(graph, partition, set_attr)
    return partition


def show_top_members(graph, partition, n):
    """Pretty-print top community members."""
    print("Top important community nodes: ")
    print("------------------------------------------------------------------")
    communitites = set(partition.values())
    for i, c in enumerate(communitites):
        members = keys_by_value(partition, c)
        top_members = top_n(
            {k: graph.nodes[k]["degree_frequency"] for k in members}, n)
        print("\tCommunity #{}: ".format(i + 1))
        for m in top_members:
            print("\t\t", m)
        print()


def compute_all_metrics(graph, degree_weights,
                        pagerank_weights=None,
                        betweenness_weights=None,
                        community_weights=None,
                        print_summary=False):
    """Compute all the metrics in one go."""
    print_top_n_nodes = None
    if print_summary:
        print_top_n_nodes = 10
        print("Computing degree centrality statistics....")
    compute_degree_centrality(graph, degree_weights, print_top_n_nodes)

    if pagerank_weights is None:
        pagerank_weights = degree_weights

    if print_summary:
        print("Computing PageRank centrality statistics....")

    compute_pagerank_centrality(
        graph, pagerank_weights, print_top_n_nodes)

    if betweenness_weights is None:
        betweenness_weights = degree_weights

    if print_summary and len(betweenness_weights) > 0:
        print("Computing betweenness centrality statistics....")

    compute_betweenness_centrality(
        graph, betweenness_weights, print_top_n_nodes)

    if community_weights is None:
        community_weights = degree_weights
    for w in community_weights:
        print(f"Using the '{w}' weight...")
        detect_communities(
            graph, weight=w, set_attr="community_{}".format(w))