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
