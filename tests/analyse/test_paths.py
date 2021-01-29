from bluegraph.backends.networkx import NXPathFinder
from bluegraph.backends.graph_tool import GTPathFinder, get_edges


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
    res = finder.n_shortest_paths("A", "D", 3, distance="distance")
    assert(set(res) == set([("A", "B", "D"), ("A", "E", "D"), ("A", "D")]))

    res = finder.n_shortest_paths("A", "D", 4, distance="distance")
    assert(set(res) == set([("A", "B", "D"), ("A", "E", "D"), ("A", "D")]))

    try:
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
    except NotImplementedError:
        pass

    res = finder.n_shortest_paths(
        "A", "D", 3, distance="distance", exclude_edge=True)
    assert(set(res) == set([("A", "B", "D"), ("A", "E", "D")]))

    try:
        res = finder.n_shortest_paths(
            "A", "D", 4, distance="distance", strategy="yen", exclude_edge=True)
        assert(set(res) == set([
                ("A", "C", "B", "D"),
                ("A", "B", "D"),
                ("A", "E", "D"),
                ("A", "C", "E", "D")
            ])
        )
    except NotImplementedError:
        pass

    res = finder.nested_shortest_path(
        "A", "B", 2, "distance", exclude_edge=True)
    assert(len(res) == 3)

    res = finder.n_nested_shortest_paths(
        "A", "B", top_level_n=5, nested_n=3, depth=2,
        distance="distance", exclude_edge=True)
    assert(len(res) == 8)

    # ------ Test tripaths ----------
    # res = finder.shortest_tripath("A", "B", "D", distance="distance")
    # print(res)

    # res = finder.shortest_tripath(
    #     "A", "B", "D", distance="distance", exclude_edge=True)
    # print(res)

    # res = finder.shortest_tripath(
    #     "A", "B", "D", distance="distance", exclude_edge=True, overlap=False)
    # print(res)


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
    assert(set(get_edges(res)) == mst)
    finder.minimum_spanning_tree(
        "distance", write=True, write_property="MST")

    edges = get_edges(finder.graph, properties=True)
    for s, t, attrs in edges:
        if (s, t) in mst:
            assert(attrs["MST"] == 1)
        else:
            assert(attrs["MST"] == 0)
