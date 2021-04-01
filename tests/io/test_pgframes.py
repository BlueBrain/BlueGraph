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

import pytest
import numpy as np
import pandas as pd

from bluegraph.core import PandasPGFrame
from bluegraph.backends.neo4j.io import pgframe_to_neo4j, neo4j_to_pgframe
from bluegraph.backends.stellargraph.io import (pgframe_to_stellargraph,
                                                stellargraph_to_pgframe)


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

    PandasPGFrame.from_frames(
        nodes=frame._nodes, edges=frame._edges,
        node_prop_types=frame._node_prop_types,
        edge_prop_types=frame._edge_prop_types)
    new_frame = frame.copy()
    assert(new_frame.nodes() == frame.nodes())
    assert(new_frame.edges() == frame.edges())
    assert(new_frame._node_prop_types == frame._node_prop_types)
    assert(new_frame._edge_prop_types == frame._edge_prop_types)


def test_pandas_modification():
    frame = PandasPGFrame()
    frame.add_nodes(["a", "b", "c"])
    assert(frame.nodes() == ["a", "b", "c"])
    frame.add_edges([("a", "b"), ("b", "c"), ("c", "a")])
    assert(frame.edges() == [("a", "b"), ("b", "c"), ("c", "a")])
    frame.add_node_properties([
        {"@id": "a", "name": "A"},
        {"@id": "a", "name": "B"},
        {"@id": "a", "name": "C"}
    ], prop_type="category")
    assert(set(frame.node_properties()) == {"name"})
    frame.remove_node_properties("name")
    assert(set(frame.node_properties()) == set())

    frame.add_edge_properties([
        {"@source_id": "a", "@target_id": "b", "name": "X"},
        {"@source_id": "b", "@target_id": "c", "name": "Y"},
        {"@source_id": "c", "@target_id": "a", "name": "Z"},
    ], prop_type="category")
    assert(set(frame.edge_properties()) == {"name"})
    frame.remove_edge_properties("name")
    assert(set(frame.edge_properties()) == set())

    assert(frame.has_node_types() is False)
    assert(frame.has_edge_types() is False)
    frame.add_node_types({
        "a": "Person",
        "b": "Person",
        "c": "software"
    })
    assert(frame.has_node_types() is True)
    assert(set(frame.node_types()) == {"Person", "software"})
    frame.add_edge_types({
        ("a", "b"): "Person",
        ("b", "c"): "Person",
        ("c", "a"): "software"
    })
    assert(frame.has_edge_types() is True)
    assert(set(frame.edge_types()) == {"Person", "software"})

    frame.remove_nodes(["a"])
    assert("a" not in frame.nodes())
    frame.remove_nodes(["a"])
    frame.remove_edges([("b", "c")])
    assert(("b", "c") not in frame.edges())
    frame.remove_isolated_nodes()
    assert(frame.number_of_nodes() == 0)


def test_pandas_getters(random_pgframe):
    assert(
        len(random_pgframe.get_node_property_values("weight")) ==
        random_pgframe.number_of_nodes())
    assert(
        len(random_pgframe.get_edge_property_values("mi")) ==
        random_pgframe.number_of_edges())
    node = random_pgframe.nodes()[0]
    s, t = random_pgframe.edges()[0]
    assert("weight" in random_pgframe.get_node(node))
    assert(set(random_pgframe.get_edge(s, t).keys()) == {"mi", "distance"})
    assert(
        len(random_pgframe.to_triples(predicate_prop="mi")) ==
        random_pgframe.number_of_edges() + random_pgframe.number_of_nodes())
    json_repr = random_pgframe.to_json()
    assert("nodes" in json_repr)
    assert("edges" in json_repr)
    assert("node_property_types" in json_repr)
    assert("edge_property_types" in json_repr)
    nodes_to_include = list(range(10))
    edges_to_include = random_pgframe.edges()[:10]
    fr = random_pgframe.filter_nodes(nodes=nodes_to_include)
    assert(fr.shape[0] == 10)
    fr = random_pgframe.filter_edges(
        edges=edges_to_include)
    assert(fr.shape[0] == 10)

    new_frame = random_pgframe.subgraph(
        nodes=nodes_to_include)
    assert(new_frame.number_of_nodes() == 10)
    assert(new_frame.number_of_edges())


def test_neo4j_io(random_pgframe, neo4j_driver):
    pgframe_to_neo4j(
        random_pgframe, driver=neo4j_driver,
        node_label="TestIONode", edge_label="TestIOEdge")
    frame = neo4j_to_pgframe(
        driver=neo4j_driver,
        node_label="TestIONode", edge_label="TestIOEdge")
    assert(frame.number_of_nodes() == random_pgframe.number_of_nodes())
    assert(frame.number_of_edges() == random_pgframe.number_of_edges())


def test_stellargraph_io(random_pgframe):
    sg_object = pgframe_to_stellargraph(random_pgframe)
    frame = stellargraph_to_pgframe(sg_object)
    assert(frame.number_of_nodes() == random_pgframe.number_of_nodes())
    assert(frame.number_of_edges() == random_pgframe.number_of_edges())
