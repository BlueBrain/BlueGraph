"""Set of utils for nodes and edge centralities."""
import networkx as nx

from analytics.utils import top_n


def compute_degree_centrality(graph, weights, print_top_n_nodes=None):
    """Compute degree centralities and save as node attrs."""
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
    """Compute PageRank centralities and save as node attrs."""
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
    """Compute PageRank centralities and save as node attrs."""
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
