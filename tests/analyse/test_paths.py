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
from bluegraph.core.analyse.paths import pretty_print_paths, pretty_print_tripaths
from bluegraph.backends.networkx import NXPathFinder, NXGraphProcessor
from bluegraph.backends.graph_tool import GTPathFinder, GTGraphProcessor
from bluegraph.backends.neo4j import Neo4jPathFinder, Neo4jGraphView, Neo4jGraphProcessor


def assert_undirected_edges_equal(edges1, edges2):
    all_edges = True
    for (s, t) in edges1:
        if (s, t) not in edges2 and (t, s) not in edges2:
            all_edges = False
            break
    if all_edges:
        for (s, t) in edges2:
            if (s, t) not in edges1 and (t, s) not in edges1:
                all_edges = False
                break
    assert(all_edges)


def benchmark_undirected_path_finder(finder):
    edges = finder.edges()
    assert_undirected_edges_equal(
        set(edges), set(
            [
                ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'),
                ('B', 'D'), ('B', 'C'), ('C', 'E'), ('D', 'E')
            ]))

    edges = finder.edges(properties=True)
    assert(len(edges) == 8)

    distance = finder.get_distance("A", "B", "distance")
    assert(int(distance) == 2)

    neighbors = finder.neighbors("A")
    assert(set(neighbors) == set(['B', 'C', 'D', 'E']))

    res = finder.top_neighbors("A", 10, "distance")
    assert({
        k: int(v) for k, v in res.items()
    } == {'D': 8, 'B': 2, 'C': 4, 'E': 2})

    # ------ Test single shortest paths ----------
    res = finder.shortest_path("A", "B")
    assert(res == ("A", "B"))
    res = finder.shortest_path(
        "A", "B", distance="distance")
    assert(res == ("A", "B"))
    res = finder.shortest_path(
        "A", "B", distance="distance", exclude_edge=True)
    assert(res == ("A", "C", "B"))

    res = finder.shortest_path("A", "D")
    assert(res == ("A", "D"))
    res = finder.shortest_path(
        "A", "D", distance="distance")
    assert(res == ("A", "B", "D"))

    # ------ Test all shortest paths ----------
    res = finder.all_shortest_paths(
        "A", "D")
    pretty_print_paths(res, as_repr=True)
    assert(res == [("A", "D")])
    res = finder.all_shortest_paths(
        "A", "D", exclude_edge=True)
    assert(set(res) == set([("A", "B", "D"), ("A", "E", "D")]))

    # ------ Test n shortest paths ----------
    try:
        res = finder.n_shortest_paths("A", "D", 3, distance="distance")
        assert(set(res) == set([("A", "B", "D"), ("A", "E", "D"), ("A", "D")]))

        res = finder.n_shortest_paths("A", "D", 4, distance="distance")
        assert(set(res) == set([("A", "B", "D"), ("A", "E", "D"), ("A", "D")]))

        res = finder.n_shortest_paths(
            "A", "D", 3, distance="distance", strategy="yen")
        assert(set(res) == set([("A", "B", "D"), ("A", "E", "D"), ("A", "D")]))

        res = finder.n_shortest_paths(
            "A", "D", 4, distance="distance", strategy="yen")
        assert(set(res) == set(
            [
                ("A", "C", "B", "D"),
                ("A", "B", "D"), ("A", "E", "D"), ("A", "D")
            ])
        )

        res = finder.n_shortest_paths(
            "A", "D", 3, distance="distance", exclude_edge=True)
        assert(set(res) == set([("A", "B", "D"), ("A", "E", "D")]))

        res = finder.n_shortest_paths(
            "A", "D", 4, distance="distance", strategy="yen", exclude_edge=True)
        assert(set(res) == set(
            [
                ("A", "C", "B", "D"),
                ("A", "B", "D"),
                ("A", "E", "D"),
                ("A", "E", "C", "B", "D")
            ])
        )

        res = finder.nested_shortest_path(
            "A", "B", 2, "distance", exclude_edge=True)
        assert(len(res) == 3)

        res = finder.n_nested_shortest_paths(
            "A", "B", top_level_n=5, nested_n=3, depth=2,
            distance="distance", exclude_edge=True)
        assert(len(res) == 8)
        res = finder.n_nested_shortest_paths(
            "A", "B", top_level_n=3, nested_n=2, depth=2,
            strategy="yen",
            distance="distance", exclude_edge=True)
        assert(len(res) == 15)
    except finder.NotImplementedError:
        pass

    # ------ Test tripaths ----------
    a_b, b_d = finder.shortest_tripath("A", "B", "D", distance="distance")
    assert(a_b == ("A", "B"))
    assert(b_d == ("B", "D"))
    pretty_print_tripaths("A", "B", "D", 1, [a_b], [b_d], as_repr=True)

    a_b, b_d = finder.shortest_tripath(
        "A", "B", "D", distance="distance", exclude_edge=True)
    assert(a_b == ('A', 'C', 'B'))
    assert(b_d == ('B', 'A', 'E', 'D'))

    res = finder.shortest_tripath(
        "A", "B", "D", distance="distance",
        exclude_edge=True, overlap=False)
    assert(a_b == ('A', 'C', 'B'))
    assert(b_d == ('B', 'A', 'E', 'D'))

    try:
        a_b, b_d = finder.n_shortest_tripaths(
            "A", "B", "D", 3, distance="distance")
        assert(set(a_b) == set([
            ('A', 'B'), ('A', 'C', 'B'), ('A', 'D', 'B')
        ]))
        assert(set(b_d) == set([('B', 'D'), ('B', 'A', 'D')]))

        pretty_print_tripaths("A", "B", "D", 3, a_b, b_d, as_repr=True)

        a_b, b_d = finder.n_shortest_tripaths(
            "A", "B", "D", 3, distance="distance", exclude_edge=True)
        assert(set(a_b) == set([('A', 'C', 'B'), ('A', 'D', 'B')]))
        assert(set(b_d) == set([('B', 'A', 'D')]))

        a_b, b_d = finder.n_shortest_tripaths(
            "A", "B", "D", 3, distance="distance", exclude_edge=True,
            overlap=False)
        assert(set(a_b) == set([('A', 'C', 'B'), ('A', 'D', 'B')]))
        assert(set(b_d) == set([('B', 'A', 'D')]))

        a_b, b_d = finder.n_shortest_tripaths(
            "A", "B", "D", 3,
            strategy="yen",
            distance="distance")
        assert(set(a_b) == set(
            [('A', 'B'), ('A', 'C', 'B'), ('A', 'E', 'D', 'B')]))
        assert(set(b_d) == set(
            [('B', 'D'), ('B', 'A', 'E', 'D'), ('B', 'C', 'E', 'D')]))

        a_b, b_d = finder.n_shortest_tripaths(
            "A", "B", "D", 3, strategy="yen",
            distance="distance", overlap=False)
        assert(set(a_b) == set(
            [('A', 'B'), ('A', 'C', 'B'), ('A', 'E', 'D', 'B')]))
        assert(set(b_d) == set([('B', 'D'), ('B', 'A', 'D')]))

    except finder.NotImplementedError:
        pass


def benchmark_directed_path_finder(finder):
    # ------ Test single shortest paths ----------
    res = finder.shortest_path("A", "B")
    assert(res == ("A", "C", "B"))
    res = finder.shortest_path(
        "A", "B", distance="distance")
    assert(res == ("A", "C", "B"))
    res = finder.shortest_path(
        "A", "B", distance="distance", exclude_edge=True)
    assert(res == ("A", "C", "B"))

    res = finder.shortest_path("B", "C")
    assert(res == ("B", "A", "C"))

    res = finder.all_shortest_paths(
        "A", "D")
    assert(res == [("A", "D")])
    res = finder.all_shortest_paths(
        "A", "D", exclude_edge=True)
    assert(set(res) == set([("A", "E", "D")]))


def test_nx_paths(path_test_graph):
    finder = NXPathFinder(path_test_graph, directed=False)
    benchmark_undirected_path_finder(finder)

    res = finder.minimum_spanning_tree("distance")
    mst = {
        ('A', 'E'), ('B', 'A'), ('B', 'C'), ('B', 'D')
    }
    assert_undirected_edges_equal(res.edges(), mst)
    finder.minimum_spanning_tree(
        "distance", write=True, write_property="MST")
    for s, t in finder.graph.edges():
        if (s, t) in mst or (t, s) in mst:
            assert(finder.graph.edges[s, t]["MST"] is True)
        else:
            assert(finder.graph.edges[s, t]["MST"] is False)

    finder = NXPathFinder(path_test_graph, directed=True)
    benchmark_directed_path_finder(finder)


def test_gt_paths(path_test_graph):
    finder = GTPathFinder(path_test_graph, directed=False)
    benchmark_undirected_path_finder(finder)

    res = finder.minimum_spanning_tree("distance")
    mst = {
        ('A', 'E'), ('A', 'B'), ('B', 'C'), ('B', 'D')
    }
    processor = GTGraphProcessor.from_graph_object(res)
    assert_undirected_edges_equal(set(processor.edges()), mst)
    finder.minimum_spanning_tree(
        "distance", write=True, write_property="MST")

    edges = finder.edges(properties=True)
    for s, t, attrs in edges:
        if (s, t) in mst or (t, s) in mst:
            assert(attrs["MST"] == 1)
        else:
            assert(attrs["MST"] == 0)

    finder = GTPathFinder(path_test_graph, directed=True)
    benchmark_directed_path_finder(finder)


def test_neo4j_paths(path_test_graph, neo4j_driver):
    finder = Neo4jPathFinder(
        pgframe=path_test_graph,
        driver=neo4j_driver,
        node_label="TestNode",
        edge_label="TestEdge", directed=False)
    benchmark_undirected_path_finder(finder)

    finder.minimum_spanning_tree(
        "distance", write=True,
        write_edge_label="MST_A", start_node="A")

    mst = {
        ('A', 'E'), ('A', 'B'), ('B', 'C'), ('B', 'D')
    }

    graph = Neo4jGraphView(
        finder.driver, "TestNode", "MST_A")

    assert_undirected_edges_equal(
        mst, Neo4jGraphProcessor.from_graph_object(graph).edges())

    finder.minimum_spanning_tree(
        "distance", write=True,
        write_edge_label="MST")
    graph = Neo4jGraphView(
        finder.driver, "TestNode", "MST")
    edges = Neo4jGraphProcessor.from_graph_object(graph).edges()
    assert(len(mst) == len(edges))

    visited = set()
    for s, t in edges:
        visited.add(s)
        visited.add(t)

    assert(visited == set(finder.nodes()))

    finder = Neo4jPathFinder(
        driver=neo4j_driver,
        node_label="TestNode",
        edge_label="TestEdge", directed=True)
    benchmark_directed_path_finder(finder)
