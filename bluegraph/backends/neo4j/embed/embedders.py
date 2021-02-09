import time
import warnings

import pandas as pd

from bluegraph.core.embed.embedders import (ElementEmbedder,
                                            DEFAULT_EMBEDDING_DIMENSION)

from ..io import Neo4jGraphProcessor, Neo4jGraphView


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


class Neo4jNodeEmbedder(ElementEmbedder):

    _transductive_models = ["node2vec", "fastrp"]
    _inductive_models = ["graphsage"]

    @staticmethod
    def _generate_graph(pgframe=None, uri=None, username=None,
                        password=None, driver=None,
                        node_label=None, edge_label=None):
        """Generate backend-specific graph object."""
        driver = Neo4jGraphProcessor.generate_driver(
            pgframe=pgframe, uri=uri, username=username,
            password=password, driver=driver,
            node_label=node_label, edge_label=edge_label)

        return Neo4jGraphView(driver, node_label, edge_label)

    @staticmethod
    def _dispatch_training_params(model_name, defaults, **kwargs):
        """Dispatch training parameters."""
        params = {}
        for k, v in kwargs.items():
            if k not in NEO4j_PARAMS[model_name]:
                warnings.warn(
                    f"StellarGraphNodeEmbedder's model '{model_name}' "
                    f"does not support the training parameter '{k}', "
                    "the parameter will be ignored",
                    ElementEmbedder.FittingWarning)
            else:
                params[k] = v

        for k, v in DEFAULT_NEO4j_PARAMS.items():
            if k not in params:
                params[k] = v

        return params

    @staticmethod
    def _fit_transductive_embedder(train_graph, params, model_name,
                                   edge_weight=None,
                                   write=False, write_property=None):
        """Fit transductive embedder (no model, just embeddings)."""
        node_edge_selector = train_graph.get_projection_query(edge_weight)
        weight_setter = (
            f"  relationshipWeightProperty: '{edge_weight}',\n"
            if edge_weight else ""
        )
        if write:
            query = (
                f"CALL {NEO4J_NODE_EMBEDDING_CALLS[model_name]}.write({{\n" +
                f"{node_edge_selector},\n{weight_setter}"
                f"    writeProperty: '{write_property}',\n{_generate_param_repr(params)}"
                "})\n"
                "YIELD computeMillis"
            )
            train_graph.execute(query)
        else:
            query = (
                f"CALL {NEO4J_NODE_EMBEDDING_CALLS[model_name]}.stream({{\n" +
                f"{node_edge_selector},\n{weight_setter}{_generate_param_repr(params)}"
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

    @staticmethod
    def _fit_inductive_embedder(train_graph, params, model_name,
                                feature_props=None, edge_weight=None):
        """Fit inductive embedder (predictive model and embeddings)."""
        node_edge_selector = train_graph.get_projection_query(
            edge_weight, node_properties=feature_props)
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

        if feature_props is None:
            feature_props = []
        featureProperties = "[{}]".format(
            ", ".join(f"'{p}'" for p in feature_props))

        weight_setter = (
            f"  relationshipWeightProperty: '{edge_weight}',\n"
            if edge_weight else ""
        )

        train_query = (
            f"CALL {NEO4J_NODE_EMBEDDING_CALLS[model_name]}.train({{\n"
            f"{node_edge_selector},\n{weight_setter}"
            f"  modelName: '{model_id}',\n"
            f"  featureProperties: {featureProperties},\n{weight_setter}{_generate_param_repr(params)}"
            "})\n"
        )
        train_graph.execute(train_query)
        return model_id

    def _predict_embeddings(self, graph, edge_weight=None,
                            write=False, write_property=None):
        node_edge_selector = graph.get_projection_query(
            edge_weight, node_properties=self._graph_configs["feature_props"])
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
        pass

    @staticmethod
    def _load_predictive_model(path):
        pass

    def fit_model(self, pgframe=None, uri=None, username=None, password=None,
                  driver=None, node_label=None, edge_label=None,
                  edge_weight=None, write=False,
                  write_property=False, **kwargs):
        """Train specified model on the provided graph."""
        if edge_weight is not None and self.model_name == "node2vec":
            warnings.warn(
                "Weighted node2vec embedding for Neo4j graphs "
                "is not implemented: computing the unweighted version",
                ElementEmbedder.FittingWarning)
            edge_weight = None

        if write:
            if write_property is None:
                raise ElementEmbedder.FittingException(
                    "Fitting has the write option set to True, "
                    "the write property name for saving embedding vectors "
                    "must be specified")

        train_graph = self._generate_graph(
            pgframe=pgframe, uri=uri, username=username,
            password=password, driver=driver,
            node_label=node_label, edge_label=edge_label)

        params = self._dispatch_training_params(
            self.model_name, self.default_params, **kwargs)

        if self.model_name in self._transductive_models:
            embeddings = self._fit_transductive_embedder(
                train_graph, params, self.model_name,
                edge_weight=edge_weight, write=write,
                write_property=write_property)
            if embeddings:
                embeddings = pd.DataFrame(
                    embeddings.items(), columns=["@id", "embedding"])
                embeddings = embeddings.set_index("@id")
        elif self.model_name in self._inductive_models:
            self._embedding_model = self._fit_inductive_embedder(
                train_graph, params, self.model_name,
                feature_props=self._graph_configs["feature_props"],
                edge_weight=edge_weight)
            embeddings = self._predict_embeddings(
                train_graph, write=write, write_property=write_property)
            embeddings = pd.DataFrame(
                embeddings.items(), columns=["@id", "embedding"])
            embeddings = embeddings.set_index("@id")
        return embeddings

    def predict_embeddings(self, pgframe=None, uri=None, username=None,
                           password=None, driver=None, node_label=None,
                           edge_label=None, edge_weight=None,
                           write=False, write_property=False,
                           **kwargs):
        """Predict embeddings of out-sample elements."""
        if write:
            if write_property is None:
                raise ElementEmbedder.PredictionException(
                    "Prediction has the write option set to True, "
                    "the write property name for saving embedding vectors "
                    "must be specified")

        if self._embedding_model is None:
            raise ElementEmbedder.PredictionException(
                "Embedder does not have a predictive model")

        graph = self._generate_graph(
            pgframe=pgframe, uri=uri, username=username,
            password=password, driver=driver,
            node_label=node_label, edge_label=edge_label)

        node_embeddings = self._predict_embeddings(
            graph, edge_weight=edge_weight,
            write=write, write_property=write_property, **kwargs)
        if node_embeddings:
            node_embeddings = pd.DataFrame(
                node_embeddings.items(), columns=["@id", "embedding"])
            node_embeddings = node_embeddings.set_index("@id")

        return node_embeddings

    def save(self, path, compress=True):
        """Save the embedder."""
        pass

    @staticmethod
    def load(path):
        """Load a dumped embedder."""
        pass
