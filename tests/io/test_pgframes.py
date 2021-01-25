import pytest
import numpy as np
import pandas as pd

from bluegraph.core import PandasPGFrame


def test_pandas_pg_creation():
    """Test PandasPGFrames creation."""
    nodes = ["a", "b", "c", "d", "e", "f"]
    sources = ["a", "b", "a", "f", "c"]
    targets = ["b", "c", "e", "e", "f"]
    edges = list(zip(sources, targets))
    frame = PandasPGFrame(nodes=nodes, edges=edges)
    assert(frame.nodes() == nodes)
    assert(frame.edges() == edges)

    # add node/edge properties
    node_weight = pd.DataFrame({
        "@id": nodes,
        "weight": range(len(nodes))
    })
    # add node/edge properties
    edge_weight = pd.DataFrame({
        "@source_id": sources,
        "@target_id": targets,
        "weight": [(el + 1) * 10 for el in range(len(sources))]
    })

    frame.add_node_properties(node_weight)
    frame.add_edge_properties(edge_weight)
    assert(frame._nodes.index.name == "@id")
    assert(frame._edges.index.names == ["@source_id", "@target_id"])
    assert(set(frame._nodes.columns) == set(["weight"]))
    assert(set(frame._edges.columns) == set(["weight"]))

    frame.node_prop_as_numeric("weight")
    frame.edge_prop_as_numeric("weight")
    assert(frame.is_numeric_node_prop("weight"))
    assert(frame.is_numeric_edge_prop("weight"))
