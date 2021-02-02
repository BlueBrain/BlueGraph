import pytest
import numpy as np
import pandas as pd

from bluegraph.core.io import PandasPGFrame


from neo4j import GraphDatabase

# Neo4j credentials (should be moved to some config files or env vars)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "admin"


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


@pytest.fixture(scope="session")
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
    query = "MATCH (n:TestNode) DETACH DELETE n"
    session = driver.session()
    session.run(query)
    session.close()


@pytest.fixture(scope="session")
def neo4j_test_node_label():
    return "TestNode"


@pytest.fixture(scope="session")
def neo4j_test_edge_label():
    return "TestEdge"


@pytest.fixture(scope="session")
def path_test_graph():
    nodes = ["A", "B", "C", "D", "E"]
    sources = ["A", "A", "A", "A", "B", "C", "C", "E"]
    targets = ["B", "C", "D", "E", "D", "B", "E", "D"]
    weights = [3, 3, 8, 2, 3, 3, 4, 3]
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

