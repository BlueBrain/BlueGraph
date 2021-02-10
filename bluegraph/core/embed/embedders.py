from abc import ABC, abstractmethod

import os
import pickle
import re
import shutil
import json

import pandas as pd

from bluegraph.exceptions import BlueGraphException, BlueGraphWarning


DEFAULT_EMBEDDING_DIMENSION = 64


class ElementEmbedder(ABC):
    """Abstract class for a node/edge embedder."""

    @property
    def _transductive_models(self):
        raise NotImplementedError

    @property
    def _inductive_models(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _generate_graph(self, pgframe):
        """Generate backend-specific graph object."""
        pass

    @staticmethod
    @abstractmethod
    def _dispatch_training_params(model_name, defaults, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def _fit_transductive_embedder(train_graph, params, model_name):
        """Fit transductive embedder (no model, just embeddings)."""
        pass

    @staticmethod
    @abstractmethod
    def _fit_inductive_embedder(train_graph, params, model_name):
        """Fit inductive embedder (predictive model and embeddings)."""
        pass

    @abstractmethod
    def _predict_embeddings(self, graph, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def _save_predictive_model(model, path):
        pass

    @staticmethod
    @abstractmethod
    def _load_predictive_model(path):
        pass

    def __init__(self, model_name, directed=False, include_type=False,
                 feature_props=None, feature_vector_prop=None, default_params=None):
        """Initialize StellarGraphEmbedder."""
        if model_name.lower() not in self._transductive_models and\
           model_name.lower() not in self._inductive_models:
            raise ElementEmbedder.InvalidModelException(
                f"Embedding model '{model_name.lower()}' is not implemented "
                f"for {self.__class__.__name__}")

        self.model_name = model_name.lower()

        # Default training parameters
        self.default_params = default_params

        self._embedding_model = None

        self._graph_configs = {
            "directed": directed,
            "include_type": include_type,
            "feature_props": feature_props,
            "feature_vector_prop": feature_vector_prop
        }

    def info(self):
        """Print embedder info."""
        title = f"'{self.__class__.__name__}' info"
        print(title)
        print("=" * len(title))

        model_type = (
            'transductive'
            if self.model_name in self._transductive_models
            else 'inductive'
        )

        trained = "True" if self._embedding_model else "False"
        model_trained = (
            f"\nTrained for prediction: {trained}"
            if model_type == "inductive"
            else ""
        )
        print(
            f"Model name: '{self.model_name}' ("
            f"{model_type}){model_trained}")
        print("Graph configurations: ")
        print(json.dumps(self._graph_configs, indent="     "))
        print("Default model parameters: ")
        print(json.dumps(self.default_params, indent="     "))

    def fit_model(self, pgframe, **kwargs):
        """Fit the embedding model."""
        train_graph = self._generate_graph(
            pgframe, self._graph_configs)

        params = self._dispatch_training_params(
            self.model_name, self.default_params, **kwargs)

        if self.model_name in self._transductive_models:
            embeddings = self._fit_transductive_embedder(
                train_graph, params, self.model_name)

            if not isinstance(embeddings, pd.DataFrame):
                embeddings = pd.DataFrame(
                    {"embedding": embeddings.tolist()},
                    index=train_graph.nodes())
        elif self.model_name in self._inductive_models:
            self._embedding_model = self._fit_inductive_embedder(
                train_graph, params, self.model_name)
            embeddings = self._predict_embeddings(
                train_graph, **kwargs)
            embeddings = pd.DataFrame(
                embeddings.items(), columns=["@id", "embedding"])
            embeddings = embeddings.set_index("@id")
        return embeddings

    def predict_embeddings(self, pgframe, **kwargs):
        """Predict embeddings of out-sample elements."""
        if self._embedding_model is None:
            raise ElementEmbedder.PredictionException(
                "Embedder does not have a predictive model")

        input_graph = self._generate_graph(pgframe, self._graph_configs)

        node_embeddings = self._predict_embeddings(input_graph, **kwargs)
        node_embeddings = pd.DataFrame(
            node_embeddings.items(), columns=["@id", "embedding"])
        node_embeddings = node_embeddings.set_index("@id")

        return node_embeddings

    def save(self, path, compress=True, save_graph=False):
        """Save the embedder."""
        # backup the model
        model_backup = self._embedding_model

        # remove model for pickling
        self._embedding_model = None

        # create a dir
        if not os.path.isdir(path):
            os.mkdir(path)

        # pickle picklable part of the embedder
        with open(os.path.join(path, "emb.pkl"), "wb") as f:
            pickle.dump(self, f)

        # save the predictive model
        if model_backup is not None:
            self._save_predictive_model(
                model_backup, os.path.join(path, "model"))

        self._embedding_model = model_backup

        if compress:
            shutil.make_archive(path, 'zip', path)
            shutil.rmtree(path)

    @staticmethod
    def load(path):
        """Load a dumped embedder."""
        decompressed = False
        if re.match(r"(.+)\.zip", path):
            # decompress
            shutil.unpack_archive(
                path,
                extract_dir=re.match(r"(.+)\.zip", path).groups()[0])
            path = re.match(r"(.+)\.zip", path).groups()[0]
            decompressed = True

        with open(os.path.join(path, "emb.pkl"), "rb") as f:
            embedder = pickle.load(f)
        embedder._embedding_model = embedder._load_predictive_model(
            os.path.join(path, "model"))
        if decompressed:
            shutil.rmtree(path)

        return embedder

    class InvalidModelException(BlueGraphException):
        """Exception class for invalid model names."""
        pass

    class FittingException(BlueGraphException):
        """Exception class for fitting errors."""
        pass

    class FittingWarning(BlueGraphWarning):
        """Exception class for fitting errors."""
        pass

    class PredictionException(BlueGraphException):
        """Exception class for fitting errors."""
        pass


class GraphEmbedder(ABC):
    """Abstract class for a graph embedder."""
    pass
