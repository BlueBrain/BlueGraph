from bluegraph.backends.networkx import NXPathFinder
from bluegraph.backends.graph_tool import GTPathFinder
from bluegraph.backends.neo4j import Neo4jPathFinder


def _benchmark_path_finder(finder):
    res = finder.top_neighbors("A", 10, "distance")
    assert(res == {'D': 8, 'B': 3, 'C': 3, 'E': 2})

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
    assert(res == ("A", "E", "D"))

    # ------ Test all shortest paths ----------
    res = finder.all_shortest_paths(
        "A", "D")
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
        assert(set(res) == set([
                ("A", "C", "B", "D"),
                ("A", "B", "D"), ("A", "E", "D"), ("A", "D")
            ])
        )

        res = finder.n_shortest_paths(
            "A", "D", 3, distance="distance", exclude_edge=True)
        assert(set(res) == set([("A", "B", "D"), ("A", "E", "D")]))

        res = finder.n_shortest_paths(
            "A", "D", 4, distance="distance", strategy="yen", exclude_edge=True)
        assert(set(res) == set([
                ("A", "C", "B", "D"),
                ("A", "B", "D"),
                ("A", "E", "D"),
                ("A", "C", "E", "D")
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

    a_b, b_d = finder.shortest_tripath(
        "A", "B", "D", distance="distance", exclude_edge=True)
    assert(a_b == ('A', 'C', 'B'))
    assert(b_d == ('B', 'A', 'E', 'D'))

    res = finder.shortest_tripath(
        "A", "B", "D", distance="distance",
        exclude_edge=True, overlap=False)
    assert(a_b == ('A', 'C', 'B'))
    assert(b_d == ('B', 'A', 'E', 'D'))

    a_b, b_d = finder.n_shortest_tripaths(
        "A", "B", "D", 3, distance="distance")
    assert(set(a_b) == set([
        ('A', 'B'), ('A', 'C', 'B'), ('A', 'D', 'B')
    ]))
    assert(set(b_d) == set([('B', 'D'), ('B', 'A', 'D')]))

    a_b, b_d = finder.n_shortest_tripaths(
        "A", "B", "D", 3, distance="distance", exclude_edge=True)
    assert(set(a_b) == set([('A', 'C', 'B'), ('A', 'D', 'B')]))
    assert(set(b_d) == set([('B', 'A', 'D')]))

    a_b, b_d = finder.n_shortest_tripaths(
        "A", "B", "D", 3, distance="distance", exclude_edge=True,
        overlap=False)
    assert(set(a_b) == set([('A', 'C', 'B'), ('A', 'D', 'B')]))
    assert(set(b_d) == set([('B', 'A', 'D')]))

    try:
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


def test_nx_paths(path_test_graph):
    finder = NXPathFinder(path_test_graph)
    _benchmark_path_finder(finder)

    res = finder.minimum_spanning_tree("distance")
    mst = {
        ('A', 'E'), ('A', 'B'), ('A', 'C'), ('B', 'D')
    }
    assert(set(res.edges()) == mst)
    finder.minimum_spanning_tree(
        "distance", write=True, write_property="MST")
    for e in finder.graph.edges():
        if e in mst:
            assert(finder.graph.edges[e]["MST"] is True)
        else:
            assert(finder.graph.edges[e]["MST"] is False)


def test_gt_paths(path_test_graph):
    finder = GTPathFinder(path_test_graph)
    _benchmark_path_finder(finder)

    res = finder.minimum_spanning_tree("distance")
    mst = {
        ('A', 'E'), ('A', 'B'), ('A', 'C'), ('E', 'D')
    }
    assert(set(GTPathFinder._get_edges(res)) == mst)
    finder.minimum_spanning_tree(
        "distance", write=True, write_property="MST")

    edges = GTPathFinder._get_edges(finder.graph, properties=True)
    for s, t, attrs in edges:
        if (s, t) in mst:
            assert(attrs["MST"] == 1)
        else:
            assert(attrs["MST"] == 0)


def test_neo4j_paths(path_test_graph, neo4j_driver, neo4j_test_node_label,
                     neo4j_test_edge_label):
    finder = Neo4jPathFinder(
        pgframe=path_test_graph,
        driver=neo4j_driver,
        node_label=neo4j_test_node_label,
        edge_label=neo4j_test_edge_label)
    _benchmark_path_finder(finder)

