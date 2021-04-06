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

from bluegraph.core.io import PandasPGFrame

from neo4j import GraphDatabase

import numpy as np

import pandas as pd

import pytest

import nltk


# Download nltk corpora used in tests
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')


# Neo4j credentials (should be moved to some config files or env vars)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j"


def generate_targets(nodes, s, density=0.2):
    edges = []
    for t in nodes:
        if s < t:
            edge = np.random.choice([0, 1], p=[1 - density, density])
            if edge:
                mi = np.random.normal(loc=0.5, scale=0.5)
                if mi < 0:
                    mi = 0
                elif mi > 1:
                    mi = 1
                edges.append([
                    s, t, mi, 1 / mi
                    if mi != 0 else np.inf
                ])
    return edges


@pytest.fixture(scope="module")
def random_pgframe():
    n_nodes = 50
    density = 0.3

    nodes = list(range(n_nodes))

    edges = sum(
        map(lambda x: generate_targets(nodes, x, density), nodes), [])
    edges = pd.DataFrame(
        edges, columns=["@source_id", "@target_id", "mi", "distance"])
    edges_df = edges.set_index(["@source_id", "@target_id"])

    frame = PandasPGFrame(nodes=nodes, edges=edges_df.index)

    frame.add_node_properties(
        pd.DataFrame({
            "@id": nodes,
            "weight": np.random.rand(n_nodes)
        }))
    frame.node_prop_as_numeric("weight")

    frame.add_edge_properties(edges_df["mi"])
    frame.edge_prop_as_numeric("mi")
    frame.add_edge_properties(edges_df["distance"])
    frame.edge_prop_as_numeric("distance")
    return frame


@pytest.fixture(scope="module")
def neo4j_driver():
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield driver
    cleanup_query = (
        "MATCH (n) "
        "WHERE any(l IN labels(n) WHERE l STARTS WITH 'Test') "
        "DETACH DELETE n"
    )
    session = driver.session()
    session.run(cleanup_query)
    session.close()


@pytest.fixture(scope="session")
def path_test_graph():
    nodes = ["A", "B", "C", "D", "E"]
    sources = ["B", "A", "A", "A", "B", "C", "C", "E"]
    targets = ["A", "C", "D", "E", "D", "B", "E", "D"]
    weights = [2, 4, 8, 2, 2, 3, 4, 3]
    edges = list(zip(sources, targets))
    frame = PandasPGFrame(nodes=nodes, edges=edges)
    edge_weight = pd.DataFrame({
        "@source_id": sources,
        "@target_id": targets,
        "distance": weights
    })
    frame.add_edge_properties(edge_weight)
    frame.edge_prop_as_numeric("distance")
    return frame


@pytest.fixture(scope="session")
def node_embedding_test_graph():
    nodes = [
        "Alice", "Bob", "Eric", "John", "Anna", "Laura", "Matt"
    ]
    age = [25, 9, 70, 42, 26, 35, 36]
    height = [180, 122, 173, 194, 172, 156, 177]
    weight = [75, 43, 68, 82, 70, 59, 81]
    sources = [
        "Alice", "Alice", "Bob", "Bob", "Bob", "Eric", "Anna", "Anna", "Matt"
    ]
    targets = [
        "Bob", "Eric", "Eric", "John", "Anna", "Anna", "Laura", "John", "John"
    ]
    weights = [1.0, 2.2, 0.3, 4.1, 1.5, 21.0, 1.0, 2.5, 7.5]
    edges = list(zip(sources, targets))
    frame = PandasPGFrame(nodes=nodes, edges=edges)

    # Add properties

    a = pd.DataFrame()
    frame.add_node_properties(
        {
            "@id": nodes,
            "age": age
        }, prop_type="numeric")
    frame.add_node_properties(
        {
            "@id": nodes,
            "height": height
        }, prop_type="numeric")
    frame.add_node_properties(
        {
            "@id": nodes,
            "weight": weight
        }, prop_type="numeric")

    edge_weight = pd.DataFrame({
        "@source_id": sources,
        "@target_id": targets,
        "distance": weights
    })
    frame.add_edge_properties(edge_weight, prop_type="numeric")
    return frame


@pytest.fixture(scope="session")
def node_embedding_prediction_test_graph():
    nodes = [
        "Marie", "Ivan", "Sarah", "Claire"
    ]
    age = [45, 10, 65, 38]
    height = [194, 122, 156, 177]
    weight = [82, 44, 59, 81]
    sources = [
        "Marie", "Marie", "Ivan", "Claire"
    ]
    targets = [
        "Ivan", "Sarah", "Claire", "Sarah"
    ]
    weights = [2.5, 11.0, 0.5, 2.5]
    edges = list(zip(sources, targets))
    frame = PandasPGFrame(nodes=nodes, edges=edges)

    # Add properties

    a = pd.DataFrame()
    frame.add_node_properties(
        {
            "@id": nodes,
            "age": age
        }, prop_type="numeric")
    frame.add_node_properties(
        {
            "@id": nodes,
            "height": height
        }, prop_type="numeric")
    frame.add_node_properties(
        {
            "@id": nodes,
            "weight": weight
        }, prop_type="numeric")

    edge_weight = pd.DataFrame({
        "@source_id": sources,
        "@target_id": targets,
        "distance": weights
    })
    frame.add_edge_properties(edge_weight, prop_type="numeric")
    return frame


@pytest.fixture(scope="session")
def community_test_graph():
    frame = PandasPGFrame.load_json(
        "tests/zachari_karate_club.json")

    sources = [s for s, _ in frame.edges()]
    targets = [t for _, t in frame.edges()]
    edge_weight = pd.DataFrame({
        "@source_id": sources,
        "@target_id": targets,
        "strength": [
            el if el > 0 else 0
            for el in np.random.normal(
                loc=0.5, scale=0.5, size=frame.number_of_edges())
        ]
    })
    frame.add_edge_properties(edge_weight, prop_type="numeric")

    return frame
