import pandas as pd
from bluegraph import PandasPGFrame

from bluegraph.preprocess.generators import CooccurrenceGenerator


def test_generation_from_nodes():
    frame = PandasPGFrame()
    nodes = pd.DataFrame([
        {"@id": "a", "papers": {1, 2, 3, 4}},
        {"@id": "b", "papers": {1, 2, 3, 5}},
        {"@id": "c", "papers": {3, 4, 5, 6}},
    ]).set_index("@id")
    frame._nodes = nodes
    generator = CooccurrenceGenerator(frame)
    generator.generate_from_nodes(
        "papers",
        compute_statistics=["frequency"])
    edges = generator.generate_from_nodes(
        "papers", total_factor_instances=6,
        compute_statistics=["frequency", "npmi"],
        parallelize=True, cores=8)
    frame._edges = edges
    assert(edges.shape[0] == 3)
    props = frame.get_edge("a", "b")
    assert(props["frequency"] == 3)
    props = frame.get_edge("b", "c")
    assert(props["frequency"] == 2)
    props = frame.get_edge("a", "c")
    assert(props["frequency"] == 2)


def test_generation_from_edges():
    frame = PandasPGFrame()
    nodes = [
        "a",
        "b",
        "c",
        "P1",
        "P2",
        "P3"
    ]
    edges = [
        ("a", "P1"),
        ("a", "P2"),
        ("a", "P3"),
        ("b", "P2"),
        ("b", "P3"),
        ("c", "P1"),
        ("c", "P3")
    ]
    frame = PandasPGFrame(nodes=nodes, edges=edges)
    frame.add_edge_types({
        ("a", "P1"): "OccursIn",
        ("a", "P2"): "OccursIn",
        ("a", "P3"): "OccursIn",
        ("b", "P2"): "OccursIn",
        ("b", "P3"): "OccursIn",
        ("c", "P1"): "OccursIn",
        ("c", "P3"): "OccursIn",
    })
    generator = CooccurrenceGenerator(frame)
    edges = generator.generate_from_edges(
        edge_type="OccursIn",
        compute_statistics=["frequency", "npmi"])

    new_frame = frame.copy()
    new_frame._edges = edges
    assert(edges.shape[0] == 3)
    props = new_frame.get_edge("a", "b")
    assert(props["frequency"] == 2)
    props = new_frame.get_edge("b", "c")
    assert(props["frequency"] == 1)
    props = new_frame.get_edge("a", "c")
    assert(props["frequency"] == 2)

    frame.add_edge_properties(
        pd.DataFrame([
            ("a", "P1", {1, 2, 3}),
            ("a", "P2", {11, 22, 33}),
            ("a", "P3", {111, 222, 333}),
            ("b", "P2", {22, 33}),
            ("b", "P3", {222, 333}),
            ("c", "P1", {1, 3}),
            ("c", "P3", {111, 333}),
        ], columns=["@source_id", "@target_id", "paragraphs"]))

    def aggregate_paragraphs(data):
        return set(sum(data["paragraphs"].apply(list), []))
    edges = generator.generate_from_edges(
        "OccursIn",
        factor_aggregator=aggregate_paragraphs,
        compute_statistics=["frequency", "ppmi", "npmi"],
        parallelize=True, cores=8)
    new_frame = frame.copy()
    new_frame._edges = edges
    assert(edges.shape[0] == 3)
    props = new_frame.get_edge("a", "b")
    assert(props["frequency"] == 4)
    props = new_frame.get_edge("b", "c")
    assert(props["frequency"] == 1)
    props = new_frame.get_edge("a", "c")
    assert(props["frequency"] == 4)
