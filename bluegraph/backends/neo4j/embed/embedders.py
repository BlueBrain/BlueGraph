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
import time
import json
import warnings

import pandas as pd

from bluegraph.core.embed.embedders import (GraphElementEmbedder,
                                            DEFAULT_EMBEDDING_DIMENSION)

from ..io import Neo4jGraphView, pgframe_to_neo4j


NEO4J_NODE_EMBEDDING_CALLS = {
    "fastrp": "gds.fastRP",
    "node2vec": "gds.alpha.node2vec",
    "graphsage": "gds.beta.graphSage"
}


DEFAULT_NEO4j_PARAMS = {
    "embeddingDimension": DEFAULT_EMBEDDING_DIMENSION,
}

NEO4j_PARAMS = {
    "fastrp": [
        "embeddingDimension",
        "iterationWeights",
        "normalizationStrength"
    ],
    "node2vec": [
        "walkLength",
        "walksPerNode",
        "windowSize",
        "walkBufferSize",
        "inOutFactor",
        "returnFactor",
        "negativeSamplingRate",
        "centerSamplingFactor",
        "contextSamplingExponent",
        "embeddingDimension",
        "initialLearningRate",
        "minLearningRate",
        "iterations"
    ],
    "graphsage": [
        "embeddingDimension",
        "activationFunction",
        "sampleSizes",
        # "featureProperties":,
        "projectedFeatureDimension",
        "batchSize",
        "tolerance",
        "learningRate",
        "epochs",
        "maxIterations",
        "searchDepth",
        "negativeSampleWeight",
    ]
}


def _generate_param_repr(params):
    param_repr = (
       ",\n".join([f"  {k}: {v}" for k, v in params.items()])
    )
    return param_repr


class Neo4jNodeEmbedder(GraphElementEmbedder):

    _transductive_models = ["node2vec", "fastrp"]
    _inductive_models = ["graphsage"]

    @staticmethod
    def _generate_graph(pgframe=None, uri=None, username=None,
                        password=None, driver=None,
                        node_label=None, edge_label=None):
        """Generate backend-specific graph object."""
        return pgframe_to_neo4j(
            pgframe=pgframe, uri=uri, username=username, password=password,
            driver=driver, node_label=node_label, edge_label=edge_label)

    def _dispatch_model_params(self, **kwargs):
        """Dispatch training parameters."""
        params = {}
        for k, v in kwargs.items():
            if k not in NEO4j_PARAMS[self.model_name]:
                warnings.warn(
                    f"StellarGraphNodeEmbedder's model '{self.model_name}' "
                    f"does not support the training parameter '{k}', "
                    "the parameter will be ignored",
                    GraphElementEmbedder.FittingWarning)
            else:
                params[k] = v

        for k, v in DEFAULT_NEO4j_PARAMS.items():
            if k not in params:
                params[k] = v

        return params

    def _fit_transductive_embedder(self, train_graph,
                                   write=False, write_property=None):
        """Fit transductive embedder (no model, just embeddings)."""
        edge_weight = self.graph_configs["edge_weight"]
        if edge_weight is not None and\
           self.model_name == "node2vec":
            warnings.warn(
                "Weighted node2vec embedding for Neo4j graphs "
                "is not implemented: computing the unweighted version",
                GraphElementEmbedder.FittingWarning)
            edge_weight = None

        node_edge_selector = train_graph.get_projection_query(edge_weight)
        weight_setter = (
            f"  relationshipWeightProperty: '{edge_weight}',\n"
            if edge_weight else ""
        )
        if write:
            query = (
                f"CALL {NEO4J_NODE_EMBEDDING_CALLS[self.model_name]}.write({{\n" +
                f"{node_edge_selector},\n{weight_setter}"
                f"    writeProperty: '{write_property}',\n{_generate_param_repr(self.params)}"
                "})\n"
                "YIELD computeMillis"
            )
            train_graph.execute(query)
        else:
            query = (
                f"CALL {NEO4J_NODE_EMBEDDING_CALLS[self.model_name]}.stream({{\n" +
                f"{node_edge_selector},\n{weight_setter}{_generate_param_repr(self.params)}"
                "})\n"
                "YIELD nodeId, embedding\n"
                "RETURN gds.util.asNode(nodeId).id AS node_id, embedding"
            )
            result = train_graph.execute(query)
            embedding = {
                record["node_id"]: record["embedding"]
                for record in result
            }
            return embedding

    def _fit_inductive_embedder(self, train_graph):
        """Fit inductive embedder (predictive model and embeddings)."""
        edge_weight = self.graph_configs["edge_weight"]
        node_edge_selector = train_graph.get_projection_query(
            edge_weight, node_properties=self.graph_configs["feature_props"])
        weight_setter = (
            f"  relationshipWeightProperty: '{edge_weight}',\n"
            if edge_weight else ""
        )

        # Train the model
        model_id = "GraphSAGEModel"

        # 1. Check if the model exists in the catalog
        query = (
            f"CALL gds.beta.model.exists('{model_id}') YIELD exists  "
            "RETURN exists"
        )
        result = train_graph.execute(query)
        model_exists = [record["exists"]for record in result][0]

        if model_exists:
            # drop the model
            query = (
                f"CALL gds.beta.model.drop('{model_id}')\n"
                "YIELD modelInfo, creationTime"
            )
            train_graph.execute(query)

        if self.graph_configs["feature_props"] is None:
            feature_props = []
        else:
            feature_props = self.graph_configs["feature_props"]
        featureProperties = "[{}]".format(
            ", ".join(f"'{p}'" for p in feature_props))

        weight_setter = (
            f"  relationshipWeightProperty: '{edge_weight}',\n"
            if edge_weight else ""
        )

        train_query = (
            f"CALL {NEO4J_NODE_EMBEDDING_CALLS[self.model_name]}.train({{\n"
            f"{node_edge_selector},\n{weight_setter}"
            f"  modelName: '{model_id}',\n"
            f"  featureProperties: {featureProperties},\n{weight_setter}{_generate_param_repr(self.params)}"
            "})\n"
        )
        train_graph.execute(train_query)
        return model_id

    def _predict_embeddings(self, graph,
                            write=False, write_property=None):
        node_edge_selector = graph.get_projection_query(
            self.graph_configs["edge_weight"],
            node_properties=self.graph_configs["feature_props"])
        if write:
            query = (
                f"CALL {NEO4J_NODE_EMBEDDING_CALLS[self.model_name]}.write({{\n"
                f"{node_edge_selector},"
                f"  modelName: '{self._embedding_model}',\n"
                f"  writeProperty: '{write_property}'\n"
                "})\n"
                "YIELD computeMillis"
            )
            graph.execute(query)
        else:
            query = (
                f"CALL {NEO4J_NODE_EMBEDDING_CALLS[self.model_name]}.stream({{\n" +
                f"{node_edge_selector},"
                f"  modelName: '{self._embedding_model}'"
                "})\n"
                "YIELD nodeId, embedding\n"
                "RETURN gds.util.asNode(nodeId).id AS node_id, embedding"
            )
            result = graph.execute(query)
            embedding = {
                record["node_id"]: record["embedding"]
                for record in result
            }
            return embedding

    @staticmethod
    def _save_predictive_model(model, path):
        with open(path, "w") as f:
            json.dump({"catalogueModelId": model}, f)

    @staticmethod
    def _load_predictive_model(path):
        with open(path, "r") as f:
            data = json.load(f)
            return data["catalogueModelId"]

    def fit_model(self, pgframe=None, uri=None, username=None, password=None,
                  driver=None, node_label=None, edge_label=None,
                  graph_view=None, write=False,
                  write_property=False):
        """Train specified model on the provided graph."""
        if write:
            if write_property is None:
                raise GraphElementEmbedder.FittingException(
                    "Fitting has the write option set to True, "
                    "the write property name for saving embedding vectors "
                    "must be specified")

        if graph_view is None:
            train_graph = self._generate_graph(
                pgframe=pgframe, uri=uri, username=username,
                password=password, driver=driver,
                node_label=node_label, edge_label=edge_label)
        else:
            train_graph = graph_view

        if self.model_name in self._transductive_models:
            embeddings = self._fit_transductive_embedder(
                train_graph, write=write,
                write_property=write_property)
            if embeddings:
                embeddings = pd.DataFrame(
                    embeddings.items(), columns=["@id", "embedding"])
                embeddings = embeddings.set_index("@id")
        elif self.model_name in self._inductive_models:
            self._embedding_model = self._fit_inductive_embedder(
                train_graph)
            embeddings = self._predict_embeddings(
                train_graph,
                write=write, write_property=write_property)
            embeddings = pd.DataFrame(
                embeddings.items(), columns=["@id", "embedding"])
            embeddings = embeddings.set_index("@id")
        return embeddings

    def predict_embeddings(self, pgframe=None, uri=None, username=None,
                           password=None, driver=None, node_label=None,
                           edge_label=None, graph_view=None,
                           write=False, write_property=False):
        """Predict embeddings of out-sample elements."""
        if write:
            if write_property is None:
                raise GraphElementEmbedder.PredictionException(
                    "Prediction has the write option set to True, "
                    "the write property name for saving embedding vectors "
                    "must be specified")

        if self._embedding_model is None:
            raise GraphElementEmbedder.PredictionException(
                "Embedder does not have a predictive model")

        if graph_view is None:
            graph = self._generate_graph(
                pgframe=pgframe, uri=uri, username=username,
                password=password, driver=driver,
                node_label=node_label, edge_label=edge_label)
        else:
            graph = graph_view

        node_embeddings = self._predict_embeddings(
            graph, write=write, write_property=write_property)
        if node_embeddings:
            node_embeddings = pd.DataFrame(
                node_embeddings.items(), columns=["@id", "embedding"])
            node_embeddings = node_embeddings.set_index("@id")

        return node_embeddings
