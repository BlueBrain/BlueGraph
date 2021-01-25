from bluegraph.backends.networkx import NXPathFinder
from bluegraph.backends.graph_tool import GTPathFinder


def test_nx_paths(path_test_graph):
    finder = NXPathFinder(path_test_graph)

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

    # ------ Test tripaths ----------
    # res = finder.shortest_tripath("A", "B", "D", distance="distance")
    # print(res)

    # res = finder.shortest_tripath(
    #     "A", "B", "D", distance="distance", exclude_edge=True)
    # print(res)

    # res = finder.shortest_tripath(
    #     "A", "B", "D", distance="distance", exclude_edge=True, overlap=False)
    # print(res)


def test_gt_paths(path_test_graph):
    finder = GTPathFinder(path_test_graph)
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