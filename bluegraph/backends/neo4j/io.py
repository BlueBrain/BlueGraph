import math
import numpy as np

from neo4j import GraphDatabase

from bluegraph.core.io import GraphProcessor


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


class Neo4jGraphProcessor(GraphProcessor):

    def __init__(self, pgframe=None, uri=None, username=None, password=None,
                 driver=None, node_label=None, edge_label=None):
        if node_label is None:
            raise Neo4jGraphProcessor.ProcessorException(
                "Cannot initialize a Neo4jMetricProcessor: "
                "node label must be specified")
        if edge_label is None:
            raise Neo4jGraphProcessor.ProcessorException(
                "Cannot initialize a Neo4jMetricProcessor: "
                "edge label must be specified")
        if driver is None:
            self.driver = GraphDatabase.driver(
                uri, auth=(username, password))
        else:
            self.driver = driver
        self.node_label = node_label
        self.edge_label = edge_label
        if pgframe is not None:
            self._generate_graph(
                pgframe, driver=driver,
                node_label=node_label, edge_label=edge_label)

    @classmethod
    def from_graph_object(cls, graph_object):
        """Instantiate a MetricProcessor directly from a Graph object."""
        raise Neo4jGraphProcessor.ProcessorException(
            "Neo4jMetricProcessor cannot be initialized from a graph object")

    @staticmethod
    def _generate_graph(pgframe, driver=None,
                        node_label=None, edge_label=None, directed=True):
        return pgframe_to_neo4j(
            pgframe=pgframe,
            driver=driver, node_label=node_label,
            edge_label=edge_label)

    def execute(self, query):
        session = self.driver.session()
        response = session.run(query)
        result = response.data()
        session.close()
        return result

    def _generate_pgframe(self, node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        return neo4j_to_pgframe(self.driver, self.node_label, self.edge_label)
