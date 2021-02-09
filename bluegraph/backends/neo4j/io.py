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

    @staticmethod
    def generate_driver(pgframe=None, uri=None, username=None,
                        password=None, driver=None, node_label=None,
                        edge_label=None):
        if driver is None:
            driver = GraphDatabase.driver(
                uri, auth=(username, password))
        if pgframe is not None:
            Neo4jGraphProcessor._generate_graph(
                pgframe, driver=driver,
                node_label=node_label, edge_label=edge_label)
        return driver

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
        self.driver = self.generate_driver(
            pgframe=pgframe, uri=uri, username=username,
            password=password, driver=driver, node_label=node_label,
            edge_label=edge_label)
        self.node_label = node_label
        self.edge_label = edge_label

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


class Neo4jGraphView(object):

    def __init__(self, driver, node_label,
                 edge_label, nodes_to_exclude=None,
                 edges_to_exclude=None):
        self.driver = driver
        self.node_label = node_label
        self.edge_label = edge_label
        self.nodes_to_exclude = nodes_to_exclude if nodes_to_exclude else []
        self.edges_to_exclude = edges_to_exclude if edges_to_exclude else []

    def execute(self, query):
        session = self.driver.session()
        response = session.run(query)
        result = response.data()
        session.close()
        return result

    def _get_nodes_query(self, return_ids=False):
        nodes_exclude_statement = ""
        if len(self.nodes_to_exclude) > 0:
            nodes_repr = ", ".join([
                f"\"{node}\"" for node in self.nodes_to_exclude
            ])
            nodes_exclude_statement =\
                f"NOT n.id IN [{nodes_repr}]"

        if len(nodes_exclude_statement) > 0:
            nodes_exclude_statement =\
                "WHERE " + nodes_exclude_statement

        if return_ids:
            return_statement = "RETURN id(n) as id"
        else:
            return_statement = "RETURN n.id as node_id, properties(n) as node"

        node_query = (
            f"MATCH (n:{self.node_label}) {nodes_exclude_statement} " +
            return_statement
        )
        return node_query

    def _get_edge_query(self, source, target):
        return (
            f"MATCH (start:{self.node_label} {{id: '{source}'}})-"
            f"[r:{self.edge_label}]-"
            f"(end:{self.node_label} {{id: '{target}'}})\n"
            "RETURN properties(r) as edge"
        )

    def _get_edges_query(self, distance=None, return_ids=False,
                         single_direction=False):
        edges_exclude_statement = ""
        edges_exceptions = []

        if len(self.nodes_to_exclude) > 0:
            nodes_repr = ", ".join([
                f"\"{node}\"" for node in self.nodes_to_exclude
            ])
            edges_exceptions.append(
                f"NOT n.id IN [{nodes_repr}] " +
                f"AND NOT m.id IN [{nodes_repr}]"
            )

        if len(self.edges_to_exclude) > 0:
            edges_exceptions += [
                f"NOT (n.id=\"{source}\" AND m.id=\"{target}\") "
                for (source, target) in self.edges_to_exclude
            ]

        if len(edges_exceptions) > 0:
            edges_exclude_statement =\
                "WHERE " + " AND ".join(edges_exceptions)

        # generate node/edge queries
        distance_selector = (
            f", r.{distance} as distance"
            if distance else ""
        )

        if return_ids:
            return_statement =\
                f"RETURN id(n) AS source, id(m) AS target {distance_selector}"
        else:
            return_statement = (
                "RETURN n.id as source_id, m.id as target_id, "
                "properties(r) as edge"
            )

        arrow = ">" if single_direction else ""
        edge_query = (
            f"MATCH (n:{self.node_label})-[r:{self.edge_label}]-{arrow}"
            f"(m:{self.node_label}) "
            f"{edges_exclude_statement} {return_statement}"
        )
        return edge_query

    def get_projection_query(self, distance=None, node_properties=None):
        if node_properties is None:
            node_properties = []

        if len(self.nodes_to_exclude) == 0 and len(self.edges_to_exclude) == 0:
            # generate node/edge projection
            distance_selector = (
                f"       properties: '{distance}',\n"
                if distance else ""
            )
            node_prop_repr = ""
            if len(node_properties) > 0:
                prop_list = ", ".join([
                    f"'{prop}'" for prop in node_properties])
                node_projection = (
                    "{\n"
                    f"    {self.node_label}: {{\n"
                    f"      label: '{self.node_label}',\n"
                    f"      properties: [{prop_list}]\n"
                    "   }\n"
                    "  }"
                )
            else:
                node_projection = f"'{self.node_label}'"

            selector = (
                f"  nodeProjection: {node_projection},\n"
                "  relationshipProjection: {\n"
                f"    Edge: {{\n"
                f"      type: '{self.edge_label}',\n{distance_selector}"
                f"      orientation: 'UNDIRECTED'\n"
                "    }\n"
                "  }"
            )
        else:
            node_query = self._get_nodes_query(return_ids=True)
            edge_query = self._get_edges_query(distance, return_ids=True)
            selector = (
                f"  nodeQuery: '{node_query}',\n"
                f"  relationshipQuery: '{edge_query}'"
            )
        return selector

    def _generate_st_match_query(self, source, target):
        return (
            f"MATCH (start:{self.node_label} {{id: '{source}'}}), "
            f"(end:{self.node_label} {{id: '{target}'}})\n"
        )

