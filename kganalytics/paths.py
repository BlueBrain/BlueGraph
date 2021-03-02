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

"""Utils for path search in co-occurrence networks."""
import networkx as nx

from kganalytics.utils import top_n


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
    for line in lines:
        print(line)


def pretty_print_tripaths(a, b, c, n, a_b_paths, b_c_paths):
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
        if i >= len(a_b_paths) and i >= len(b_c_paths):
            break
        left = a_b_paths_repr[i] if i < len(a_b_paths) else (" " * max_left)
        right = b_c_paths_repr[i] if i < len(b_c_paths) else (" " * max_right)
        print(
            " " * len(a_repr), left,
            " " * (max_left - len(left) + len(b_repr)),
            right)


def get_cumulative_distance(graph, path, distance="distance_npmi"):
    """Get cumulative distance score along the path."""
    result = 0
    for i in range(1, len(path)):
        source = path[i - 1]
        target = path[i]
        result += graph.edges[source, target][distance]
    return result


def single_shortest_path(graph, source, target, distance=None):
    """Get the single shortest path."""
    path = list(nx.shortest_path(graph, source, target, weight=distance))
    return path


def get_all_paths(graph, input_source, input_target,
                  distance, path_condition=None):
    """Get all shortest paths between a pair of nodes and their distance score.

    Parameters
    ----------
    graph : nx.Graph
        Input graph object
    input_source : str
        Source node ID
    input_target : str
        Target node ID
    distance : str
        The name of the attribute to use as the edge distance
    path_condition : func, optional
        Edge filtering function returning Boolean flag

    Returns
    -------
    path_ranking : dict
        Dictionary whose keys are paths (n-tuples of 2-tuples) and whose
        values are the cumulative distance scores of the corresponding paths.
    """
    backup_edge = None
    path_ranking = {}

    if (input_source, input_target) in graph.edges():
        backup_edge = {**graph.edges[input_source, input_target]}
        graph.remove_edge(input_source, input_target)
    try:
        shortest_paths = list(
            nx.all_shortest_paths(graph, input_source,
                                  input_target))
        path_ranking = {}
        for p in shortest_paths:
            if path_condition is None or path_condition(p):
                path_ranking[tuple(p[1:-1])] = get_cumulative_distance(
                    graph, p, distance=distance)

    except Exception as e:
        print(e)
        pass

    if backup_edge is not None:
        graph.add_edge(input_source, input_target, **backup_edge)

    return path_ranking


def top_n_paths(graph, source, target, n, distance=None,
                path_condition=None, strategy="naive"):
    """Get top n shortest indirect paths from the source node to the target node.

    The paths are indirect in the sense that, if there exists a direct edge
    from the source to the target node, it is discarded as the search result.
    Therefore, the result of this function consists of paths with two
    or more edges. Two search strategies are available: 'naive' and 'yen'.
    The naive strategy first finds the set of all shortest paths from the
    source to the target node, it then ranks them by the cumulative distance
    score and returns n best paths. The second strategy uses Yen's
    algorithm [1] for finding n shortest paths. The first naive strategy
    performs better for highly dense graphs (where every node is connected to
    almost every other node).


    1. Yen, Jin Y. "Finding the k shortest loopless paths in a network".
    Management Science 17.11 (1971): 712-716.

    Parameters
    ----------
    graph : nx.Graph
        Input graph object
    source : str
        Source node ID
    target : str
        Target node ID
    n : int
        Number of top paths to include in the result
    distance : str, optional
        The name of the attribute to use as the edge distance
    path_condition : func, optional
        Edge filtering function returning Boolean flag
    strategy : str, optional
        Path finding strategy: `naive` or `yen`. By default, `naive`.

    Returns
    -------
    paths : list
        List containing top n best paths according to the distance score
    """
    if n == 1:
        return [tuple(single_shortest_path(graph, source, target))]
    if strategy == "naive":
        path_ranks = get_all_paths(
            graph, source, target,
            path_condition=path_condition, distance=distance)

        path_ranks = {
            tuple([source] + [el for el in p] + [target]): r
            for p, r in path_ranks.items()
        }
        if len(path_ranks) == 0:
            raise ValueError("No undirect paths from '{}' to '{}' found".format(
                source, target))
        paths = top_n(path_ranks, n, smallest=True)
    elif strategy == "yen":
        generator = nx.shortest_simple_paths(
            graph, source, target, weight=distance)
        i = 0
        paths = []
        for path in generator:
            if path_condition is None or path_condition(path):
                paths.append(path)
                i += 1
            if i == n:
                break
    else:
        raise ValueError("Unknown path search strategy '{}'".format(
            strategy))

    return paths


def top_n_tripaths(graph, a, b, c, n, distance=None,
                   strategy="naive", intersecting=True):
    """Get top n shortest indirect 'tripaths' from A to C passing through B.

    Tripaths cosist of two path sets, from the node A to the node B and from
    the node B to the node C. These sets can be overlapping or not. If the sets
    are non-overlapping, all the nodes encountered on the paths from A to B
    are excluded from the search of paths from B to C.
    The paths are indirect in the sense that,
    if there exists a direct edge from the source to the target node,
    it is discarded as the search result. Therefore, the result of this
    function consists of paths with two or more edges.


    1. Yen, Jin Y. "Finding the k shortest loopless paths in a network".
    Management Science 17.11 (1971): 712-716.

    Parameters
    ----------
    graph : nx.Graph
        Input graph object
    a : str
        Source node ID
    b : str
        Intermediate node ID
    c : str
        Target node ID
    n : int
        Number of top paths to include in the result
    distance : str, optional
        The name of the attribute to use as the edge distance
    strategy : str, optional
        Path finding strategy: `naive` or `yen`. By default, `naive`
        (see details in `top_n_paths`).
    intersecting : bool, optional.
        Flag indicating whether the two sets of paths are allowed to
        intersect (to pass through the same nodes). By default True.

    Returns
    -------
    a_b_paths : list
        List containing top n best paths from A to B
    b_c_paths : list
        List containing top n best paths from B to C
    """
    def non_intersecting(path, reference_paths):
        core = set(path[1:-1])
        for p in reference_paths:
            if len(set(p[1:-1]).intersection(core)) > 0:
                return False
        return True

    a_b_paths = top_n_paths(
        graph, a, b, n, distance=distance, strategy=strategy)
    if not intersecting:
        b_c_paths = top_n_paths(
            graph, b, c, n,
            path_condition=lambda x: non_intersecting(x, a_b_paths),
            distance=distance,
            strategy=strategy)
    else:
        b_c_paths = top_n_paths(
            graph, b, c, n,
            distance=distance,
            strategy=strategy)

    return (a_b_paths, b_c_paths)


def top_n_nested_paths(graph, source, target, n, nested_n=None,
                       distance=None, strategy="naive", depth=1):
    """Find top n nested paths.

    Nested paths are found iteratively for each level of depth. For example,
    if `e1 <-> e2 <-> ... <-> eN` is a path on the current level of depth,
    then the function searches for paths between each consecutive pair of
    nodes (e1 and e2, e2 and e3, etc.).

    Parameters
    ----------
    graph : nx.Graph
        Input graph object
    source : str
        Source node ID
    target : str
        Target node ID
    n : int
        Number of top paths to include in the result
    nested_n : int
        Number of top paths to include in the result for the depth > 1
    distance : str, optional
        The name of the attribute to use as the edge distance
    strategy : str, optional
        Path finding strategy: `naive` or `yen`. By default, `naive`.
    depth : int, optional
        Number of interactions of the path search


    Returns
    -------
    current_paths : list
        List containing best nested paths according to the distance score
    """
    if nested_n is None:
        nested_n = n
    current_paths = [[source, target]]
    visited = set()
    for level in range(depth):
        new_paths = []
        for path in current_paths:
            for i in range(1, len(path)):
                s = path[i - 1]
                t = path[i]
                if (s, t) not in visited and (t, s) not in visited:
                    visited.add((s, t))
                    new_paths += top_n_paths(
                        graph, s, t, n if level == 0 else nested_n,
                        strategy=strategy, distance=distance)
        current_paths = new_paths
    return current_paths


def minimum_spanning_tree(graph, weight):
    """Compute the minimum spanning tree."""
    return nx.minimum_spanning_tree(graph, weight=weight)


def graph_from_paths(paths, source_graph=None):
    """Create a graph from a set of paths.

    Resulting graph contains nodes and edges from the input
    set  of paths. If the source graph is provided, the attributes
    of the selected nodes and edges are copied from the source graph.
    """
    nodes = set()
    edges = set()
    for p in paths:
        nodes.add(p[0])
        for i in range(1, len(p)):
            nodes.add(p[i])
            edges.add((p[i - 1], p[i]))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if source_graph is not None:
        # Graph are asumed to be hemogeneous
        attrs = source_graph.nodes[list(nodes)[0]].keys()
        for k in attrs:
            nx.set_node_attributes(
                graph, {n: source_graph.nodes[n][k] for n in nodes}, k)
        edge_attrs = source_graph.edges[list(edges)[0]].keys()
        for k in edge_attrs:
            nx.set_edge_attributes(
                graph, {e: source_graph.edges[e][k] for e in edges}, k)
    return graph


def top_neighbors(graph, node, n, weight):
    """Get top n neighbours of the specified node by weight."""
    neigbours = {}
    for neighbor in graph.neighbors(node):
        neigbours[neighbor] = graph.edges[node, neighbor][weight]
    return {
        el: neigbours[el] for el in top_n(neigbours, n)
    }
