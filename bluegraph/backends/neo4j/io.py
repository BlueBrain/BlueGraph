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
import math
import numpy as np
import numbers
import pandas as pd

from neo4j import GraphDatabase

from bluegraph.core.io import GraphProcessor, PandasPGFrame


def execute(driver, query):
    session = driver.session()
    response = session.run(query)
    result = response.data()
    session.close()
    return result


def generate_neo4j_driver(uri=None, username=None,
                          password=None, driver=None):
    if driver is not None:
        return driver
    elif uri is not None and\
            username is not None and password is not None:
        driver = GraphDatabase.driver(
            uri, auth=(username, password))
    return driver


def preprocess_value(v):
    if v == float("inf"):
        return "1.0 / 0.0"
    elif v == float("-inf"):
        return "-s1.0 / 0.0"
    return v


def safe_node_id(index):
    if isinstance(index, str):
        return index.replace("\'", "\\'")
    return index


def _generate_property_repr(properties, prop_types=None):
    if prop_types is None:
        prop_types = {}
        for k, v in properties.items():
            prop_types[k] = (
                "numeric"
                if isinstance(v, numbers.Number) else "category"
            )
    props = []
    for k, v in properties.items():
        if k not in ["@id", "@type"]:
            quote = "'"
            if prop_types[k] == "numeric" and not math.isnan(v):
                quote = ""
            props.append("{}: {}{}{}".format(
                k, quote, preprocess_value(v), quote))
    return props


def pgframe_to_neo4j(pgframe=None, uri=None, username=None, password=None,
                     driver=None, node_label=None, edge_label=None,
                     directed=True, batch_size=10000):
    """Write the property graph to the Neo4j databse."""
    driver = generate_neo4j_driver(uri, username, password, driver)

    if pgframe is None:
        return Neo4jGraphView(
            driver, node_label, edge_label, directed=directed)

    # Create nodes
    # Split nodes into batches
    batches = np.array_split(
        pgframe._nodes.index, math.ceil(pgframe.number_of_nodes() / batch_size))
    # Run node creation queries for different batches
    for batch in batches:
        node_batch = pgframe._nodes.loc[batch]
        node_repr = []
        for index, properties in node_batch.to_dict("index").items():
            node_id = safe_node_id(index)
            node_dict = ["id: '{}'".format(node_id)]
            node_dict += _generate_property_repr(
                properties, pgframe._node_prop_types)
            node_repr.append("{" + ", ".join(node_dict) + "}")

        query = (
        f"""
        WITH [{", ".join(node_repr)}] AS batch
        UNWIND batch as individual
        CREATE (n:{node_label})
        SET n += individual
        """)
        execute(driver, query)

    # Create edges
    # Split edges into batches
    batches = np.array_split(
        pgframe._edges.index, math.ceil(pgframe.number_of_edges() / batch_size))
    for batch in batches:
        edge_batch = pgframe._edges.loc[batch]
        edge_repr = []
        for (s, t), properties in edge_batch.to_dict("index").items():
            edge_dict = [
                "source: '{}'".format(safe_node_id(s)),
                "target: '{}'".format(safe_node_id(t))
            ]
            edge_props = []
            for k, v in properties.items():
                quote = "'"
                if pgframe._edge_prop_types[k] == "numeric":
                    quote = ""
                edge_props.append(f"{k}: {quote}{preprocess_value(v)}{quote}")
            edge_dict.append("props: {{{}}}".format(
                ', '.join(_generate_property_repr(
                    properties, pgframe._edge_prop_types))))
            edge_repr.append("{" + ", ".join(edge_dict) + "}")
        query = (
        f"""
        WITH [{", ".join(edge_repr)}] AS batch
        UNWIND batch as individual
        MATCH (n:{node_label} {{id: individual["source"]}}), (m:{node_label} {{id: individual["target"]}})
        CREATE (n)-[r:{edge_label}]->(m)
        SET r += individual["props"]
        """)
        execute(driver, query)

    return Neo4jGraphView(driver, node_label, edge_label, directed=directed)


def neo4j_to_pgframe(uri=None, username=None, password=None,
                     driver=None, node_label=None, edge_label=None,
                     node_prop_types=None, edge_prop_types=None):
    driver = generate_neo4j_driver(uri, username, password, driver)
    # Get nodes and their properties
    query = (
        f"MATCH (n:{node_label}) RETURN n as node"
    )
    result = execute(driver, query)

    nodes_frame = pd.DataFrame([record["node"] for record in result]).rename(
        columns={"id": "@id"})
    nodes_frame["@id"] = nodes_frame["@id"].apply(str)
    nodes_frame = nodes_frame.set_index("@id")

    # Get edges and their properties
    query = (
        f"""MATCH (n:{node_label})-[r:{edge_label}]->(m:{node_label})
        RETURN n.id as source_id, m.id as target_id, properties(r) as edge
        """
    )
    result = execute(driver, query)
    edges_frame = pd.DataFrame([
        {
            **record["edge"],
            "@source_id": record["source_id"],
            "@target_id": record["target_id"]
        }
        for record in result
    ])
    edges_frame["@source_id"] = edges_frame["@source_id"].apply(str)
    edges_frame["@target_id"] = edges_frame["@target_id"].apply(str)
    edges_frame = edges_frame.set_index(["@source_id", "@target_id"])

    return PandasPGFrame.from_frames(
        nodes=nodes_frame, edges=edges_frame,
        node_prop_types=node_prop_types, edge_prop_types=edge_prop_types)


class Neo4jGraphProcessor(GraphProcessor):
    """Neo4j graph processor.

    The provided interface allows to communicate with an
    instance of the Neo4j database, populate it with the
    input PGFrames and read out PGFrames from the database.
    """

    def __init__(self, pgframe=None, uri=None, username=None, password=None,
                 driver=None, node_label=None, edge_label=None, directed=True):
        if node_label is None:
            raise Neo4jGraphProcessor.ProcessorException(
                "Cannot initialize a Neo4jMetricProcessor: "
                "node label must be specified")
        if edge_label is None:
            raise Neo4jGraphProcessor.ProcessorException(
                "Cannot initialize a Neo4jMetricProcessor: "
                "edge label must be specified")

        self.driver = generate_neo4j_driver(
            uri=uri, username=username,
            password=password, driver=driver)

        self.directed = directed

        if pgframe is not None:
            Neo4jGraphProcessor._generate_graph(
                pgframe, driver=driver,
                node_label=node_label, edge_label=edge_label)

        self.node_label = node_label
        self.edge_label = edge_label

    @classmethod
    def from_graph_object(cls, graph_veiw):
        """Instantiate a MetricProcessor directly from a Graph object."""
        processor = cls(
            driver=graph_veiw.driver,
            node_label=graph_veiw.node_label,
            edge_label=graph_veiw.edge_label,
            directed=graph_veiw.directed)
        return processor

    def _yeild_node_property(self, new_property):
        """Return dictionary containing the node property values."""
        pass

    def _write_node_property(self, new_property, property_name):
        """Write node property values to the graph."""
        pass

    def _get_adjacency_matrix(self, nodes, weight=None):
        weight_expr = f"r.{weight}" if weight is not None else 1
        query = (
            f"MATCH (n:{self.node_label})\n"
            "WITH n\n"
            "ORDER by n.id\n"
            "WITH COLLECT(n) AS nodes\n"
            "UNWIND nodes AS n1\n"
            "UNWIND nodes AS n2\n"
            f"OPTIONAL MATCH (n1)-[r:{self.edge_label}]->(n2)\n"
            f"WITH n1, n2, CASE WHEN r is null THEN 0 ELSE {weight_expr} END AS overlap\n"
            "ORDER BY n1.id, n2.id\n"
            "RETURN n1.id as node_id, COLLECT(overlap) as adjacency\n"
            "ORDER BY n1.id"
        )
        result = execute(self.driver, query)
        result_dict = {
            record["node_id"]: record["adjacency"]
            for record in result
        }
        return [result_dict[str(n)] for n in nodes]

    def _get_node_property_values(self, prop, nodes):
        query = (
            f"MATCH (n:{self.node_label})\n"
            f"RETURN n.id as node_id, n.{prop} as prop"
        )
        result = execute(self.driver, query)
        result_dict = {
            record["node_id"]: record["prop"]
            for record in result
        }
        return [
            result_dict[str(n)]
            for n in nodes
        ]

    @staticmethod
    def _generate_graph(pgframe, driver=None,
                        node_label=None, edge_label=None, directed=True):
        return pgframe_to_neo4j(
            pgframe=pgframe,
            driver=driver, node_label=node_label,
            edge_label=edge_label)

    def _get_identity_view(self):
        return Neo4jGraphView(
            self.driver, self.node_label, self.edge_label,
            directed=self.directed)

    def execute(self, query):
        return execute(self.driver, query)

    def _generate_pgframe(self, node_prop_types=None, edge_prop_types=None,
                          node_filter=None, edge_filter=None):
        """Get a new pgframe object from the wrapped graph object."""
        return neo4j_to_pgframe(
            driver=self.driver,
            node_label=self.node_label,
            edge_label=self.edge_label,
            node_prop_types=node_prop_types,
            edge_prop_types=edge_prop_types)

    @staticmethod
    def _is_directed(graph):
        return graph.directed

    def nodes(self, properties=False):
        graph = self._get_identity_view()
        query = graph._get_nodes_query()
        result = graph.execute(query)
        nodes = []
        for record in result:
            n = record["node_id"]
            if properties:
                props = record["node"]
                nodes.append((n, props))
            else:
                nodes.append(n)
        return nodes

    def get_node(self, node):
        query = (
            f"MATCH (n:{self.node_label} {{id: '{node}'}}) "
            "RETURN properties(n) as props"
        )
        result = self.execute(query)
        properties = {}
        for record in result:
            properties = record["props"]
        del properties["id"]
        return properties

    def remove_node(self, node):
        query = (
            f"MATCH (n:{self.node_label} {{id: '{node}'}}) "
            "DETACH DELETE n"
        )
        self.execute(query)

    def rename_nodes(self, node_mapping):
        query = (
            "MATCH {} \n".format(", ".join(
                [
                    "(`{}`:{} {{id: '{}'}})".format(
                        k, self.node_label, k)
                    for k in node_mapping.keys()
                ])
            ) + "\n".join([
                "SET `{}`.id = '{}'".format(k, v)
                for k, v in node_mapping.items()
            ])
        )
        self.execute(query)

    def set_node_properties(self, node, properties):
        property_repr = ", ".join(
            _generate_property_repr(properties))
        query = (
            f"""MATCH (n:{self.node_label} {{id: '{node}'}})
            SET n = {{}}
            SET n.id = '{node}'
            SET n += {{ {property_repr} }}
            """
        )
        result = self.execute(query)

    def add_edge(self, source, target, properties=None):
        property_repr = ""
        if properties is not None:
            property_repr = "{" + ", ".join(
                _generate_property_repr(properties)) + "}"
        query = (
            f"""MATCH (n:{self.node_label} {{id: '{source}'}}),
                (m:{self.node_label} {{id: '{target}'}})
            CREATE (n)-[r:{self.edge_label} {property_repr}]->(m)
            """
        )
        result = self.execute(query)

    def set_edge_properties(self, source, target, properties):
        property_repr = ", ".join(
            _generate_property_repr(properties))
        query = (
            f"MATCH (n:{self.node_label} {{id: '{source}'}})-"
            f"[r:{self.edge_label}]-(m:{self.node_label} {{id: '{target}'}}) "
            "SET r = {} "
            f"SET r += {{ {property_repr} }}"
        )
        result = self.execute(query)

    def remove_edge(self, source, target):
        query = (
            f"MATCH (n:{self.node_label} {{id: '{source}'}})-"
            f"[r:{self.edge_label}]-(m:{self.node_label} {{id: '{target}'}}) "
            "DELETE r"
        )
        result = self.execute(query)

    def edges(self, properties=False):
        graph = self._get_identity_view()
        query = graph._get_edges_query(single_direction=True)
        result = graph.execute(query)
        edges = []
        for record in result:
            s = record["source_id"]
            t = record["target_id"]
            if properties:
                props = record["edge"]
                edges.append((s, t, props))
            else:
                edges.append((s, t))
        return edges

    def get_edge(self, source, target):
        query = (
            f"MATCH (n:{self.node_label} {{id: '{source}'}})-"
            f"[r:{self.edge_label}]-(m:{self.node_label} {{id: '{target}'}}) "
            "RETURN properties(r) as props"
        )
        result = self.execute(query)
        for record in result:
            properties = record["props"]
        return properties

    def neighbors(self, node_id):
        """Get neighors of the node."""
        query = (
            f"MATCH (n:{self.node_label} {{id: '{node_id}'}})-"
            f"[r:{self.edge_label}]-(m:{self.node_label})\n"
            "RETURN m.id as neighor"
        )
        result = self.execute(query)
        return [record["neighor"] for record in result]

    def subgraph(self, nodes_to_include=None, edges_to_include=None,
                 nodes_to_exclude=None, edges_to_exclude=None):
        """Get a node/edge induced subgraph."""
        return Neo4jGraphView(
            self.driver, self.node_label, self.edge_label,
            nodes_to_exclude=nodes_to_exclude,
            edges_to_exclude=edges_to_exclude,
            directed=self.directed)


class Neo4jGraphView(object):
    """Neo4j persistent graph view.

    This interface allows to create virtual views of a
    persistent Neo4j graph. Such views are defined given a
    node and edge labels which define subsets of nodes and edges
    to consider. They also allow to filter specific nodes and edges,
    as well as selecting whether the underlying graph is considered to
    be directed or undirected. The main goal of creating such a graph
    view is to be able to automatically generate queries operating on the
    subgraph defined by the view.

    TODO: make methods public
    """
    def __init__(self, driver, node_label,
                 edge_label, nodes_to_exclude=None,
                 edges_to_exclude=None, directed=True):
        self.driver = driver
        self.node_label = node_label
        self.edge_label = edge_label
        self.nodes_to_exclude = nodes_to_exclude if nodes_to_exclude else []
        self.edges_to_exclude = edges_to_exclude if edges_to_exclude else []
        self.directed = directed

    def execute(self, query):
        return execute(self.driver, query)

    def _clear(self):
        node_match = self._get_nodes_query(no_return=True)
        query = f"{node_match} DETACH DELETE n"
        self.execute(query)

    def _get_nodes_query(self, return_ids=False, no_return=False):
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
        if no_return:
            return_statement = ""
        else:
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
                         single_direction=False, no_return=False):
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

        if no_return:
            return_statement = ""
        else:
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

            orientation = 'NATURAL' if self.directed else 'UNDIRECTED'

            selector = (
                f"  nodeProjection: {node_projection},\n"
                "  relationshipProjection: {\n"
                f"    Edge: {{\n"
                f"      type: '{self.edge_label}',\n{distance_selector}"
                f"      orientation: '{orientation}'\n"
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
            "MATCH (start:{} {{id: '{}'}}), ".format(
                self.node_label, safe_node_id(source)) +
            "(end:{} {{id: '{}'}})\n".format(
                self.node_label, safe_node_id(target))
        )
