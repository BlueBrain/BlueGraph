import math
import numpy as np

from neo4j import GraphDatabase


def pgframe_to_neo4j(pgframe=None, uri=None, username=None, password=None,
                     driver=None, node_label=None, edge_label=None,
                     batch_size=10000):
    """Write the property graph to the Neo4j databse."""
    if not driver:
        driver = GraphDatabase.driver(
            uri, auth=(username, password))

    if pgframe is None:
        return driver

    def execute(query):
        session = driver.session()
        session.run(query)
        session.close()

    def preprocess_value(v):
        if v == float("inf"):
            return "1.0 / 0.0"
        elif v == float("-inf"):
            return "-s1.0 / 0.0"
        return v

    # Create nodes
    # Split nodes into batches
    batches = np.array_split(
        pgframe._nodes.index, math.ceil(pgframe.number_of_nodes() / batch_size))
    # Run node creation queries for different batches
    for batch in batches:
        node_batch = pgframe._nodes.loc[batch]
        node_repr = []
        for index, properties in node_batch.to_dict("index").items():
            node_dict = [f"id: '{index}'"]
            for k, v in properties.items():
                quote = "'"
                if pgframe._node_prop_types[k] == "numeric":
                    quote = ""
                node_dict.append(f"{k}: {quote}{preprocess_value(v)}{quote}")
            node_repr.append("{" + ", ".join(node_dict) + "}")

        query = (
        f"""
        WITH [{", ".join(node_repr)}] AS batch
        UNWIND batch as individual
        CREATE (n:{node_label})
        SET n += individual
        """)
        execute(query)

    # Create edges
    # Split edges into batches
    batches = np.array_split(
        pgframe._edges.index, math.ceil(pgframe.number_of_edges() / batch_size))
    for batch in batches:
        edge_batch = pgframe._edges.loc[batch]
        edge_repr = []
        for (s, t), properties in edge_batch.to_dict("index").items():
            edge_dict = [f"source: '{s}'", f"target: '{t}'"]
            edge_props = []
            for k, v in properties.items():
                quote = "'"
                if pgframe._edge_prop_types[k] == "numeric":
                    quote = ""
                edge_props.append(f"{k}: {quote}{preprocess_value(v)}{quote}")
            edge_dict.append(f"props: {{ {', '.join(edge_props)} }}")
            edge_repr.append("{" + ", ".join(edge_dict) + "}")
        query = (
        f"""
        WITH [{", ".join(edge_repr)}] AS batch
        UNWIND batch as individual
        MATCH (n:{node_label} {{id: individual["source"]}}), (m:{node_label} {{id: individual["target"]}})
        CREATE (n)-[r:{edge_label}]->(m)
        SET r += individual["props"]
        """)
        execute(query)
    return driver


def neo4j_to_pgframe(driver, node_label, edge_label):
    pass