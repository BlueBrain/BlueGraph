import pickle

import pandas as pd
import networkx as nx

from analytics.utils import top_n


def minimum_spanning_tree(graph, weight):
    """Compute the minimum spanning tree."""
    return nx.minimum_spanning_tree(graph, weight=weight)


def pretty_print_paths(paths, as_repr=False):
    """Pretty print a set of same source/target paths."""
    a = paths[0][0]
    b = paths[0][-1]
    a_repr = "{} <-> ".format(a)
    b_repr = " <-> {}".format(b)
    path_repr = [
        " <-> ".join(p[1:-1])
        for p in paths
    ]
    lines = ["{}{}{}".format(
        a_repr, " " * max(len(p) for p in path_repr), b_repr)]
    lines += ["{}{}".format(" " * len(a_repr), p) for p in path_repr]
    if as_repr:
        return "\n".join(lines)
    for l in lines:
        print(l)


def get_cumulative_weight(graph, path, weight):
    """Get cumulative weight along the path."""
    result = 0
    for i in range(1, len(path)):
        source = path[i - 1]
        target = path[i]
        result += graph.edges[source, target][weight]
    return result


def get_all_paths(graph, input_source, input_target,
                  weight, path_condition=None):
    """Get all shortest paths."""
    backup_edge = None
    path_ranking = {}

    if (input_source, input_target) in graph.edges():
        backup_edge  = {**graph.edges[input_source, input_target]}
        graph.remove_edge(input_source, input_target)
    try:
        shortest_paths = list(
            nx.all_shortest_paths(graph, input_source,
                                  input_target))
        path_ranking = {}
        for p in shortest_paths:
            if path_condition is None or path_condition(p):
                path_ranking[tuple(p[1:-1])] = get_cumulative_weight(
                    graph, p, weight=weight)
        return(path_ranking)

    except Exception as e:
        print(e)
        pass

    if backup_edge is not None:
        graph.add_edge(input_source, input_target, **backup_edge)

    return path_ranking


def top_n_paths(graph, a, b, n, weight, path_condition=None,
                pretty_print=False, pretty_repr=False, strategy="naive"):
    """Get top n shortest paths."""
    if strategy == "naive":
        path_ranks = get_all_paths(
            graph, a, b, path_condition=path_condition, weight=weight)
        path_ranks = {
            tuple([a] + [el for el in p] + [b]): r
            for p, r in path_ranks.items()
        }
        paths = top_n(path_ranks, n)
    elif strategy == "yen":
        generator = nx.shortest_simple_paths(
            graph, a, b, weight="distance_{}".format(weight))
        i = 0
        paths = []
        for path in generator:
            paths.append(path)
            i += 1
            if i == n:
                break

    if pretty_print or pretty_repr:
        r = pretty_print_paths(paths, as_repr=pretty_repr)
        return paths, r
    return paths


def single_shortest_path(graph, a, b, pretty_print=False):
    """Get the single shortest path."""
    path = list(nx.shortest_path(graph, a, b))
    if pretty_print:
        print(" -> ".join(path))
    return path


def top_n_tripaths(graph, a, b, c, n, intersecting=True, pretty_print=False, 
                   pretty_repr=False, weight=None, strategy="naive"):
    """Get top n shortest 'tripaths'."""
    def non_intersecting(path, reference_paths):
        core = set(path[1:-1])
        for p in reference_paths:
            if len(set(p[1:-1]).intersection(core)) > 0:
                return False
        return True

    a_b_paths = top_n_paths(
        graph, a, b, n, weight=weight,
        strategy=strategy)
    if not intersecting:
        b_c_paths = top_n_paths(
            graph, b, c, n,
            path_condition=lambda x: non_intersecting(x, a_b_paths),
            weight=weight,
            strategy=strategy)
    else:
        b_c_paths = top_n_paths(
            graph, b, c, n,
            weight=weight,
            strategy=strategy)
    path_ranking = {}

    if pretty_print:
        a_b_paths_repr = [
            " -> ".join(p[1:-1]) for p in a_b_paths
        ]

        b_c_paths_repr = [
            " -> ".join(p[1:-1]) for p in b_c_paths
        ]

        max_left = max([len(el) for el in a_b_paths_repr])
        max_right = max([len(el) for el in b_c_paths_repr])

        a_repr = "{} ->".format(a)
        b_repr = "-> {} ->".format(b)
        c_repr = "-> {}".format(c)
        print("{}{}{}{}{}".format(
            a_repr,
            " " * max_left,
            b_repr,
            " " * max_right,
            c_repr))
        for i in range(n):
            if i == len(a_repr):
                break
            print(
                " " * len(a_repr), a_b_paths_repr[i],
                " " * (max_left - len(a_b_paths_repr[i]) + len(b_repr)),
                b_c_paths_repr[i])
    return (a_b_paths, b_c_paths)
