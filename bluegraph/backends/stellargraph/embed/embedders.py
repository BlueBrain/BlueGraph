import os
import pickle
import re
import shutil

import numpy as np
import pandas as pd
import stellargraph as sg
from scipy.spatial import cKDTree

from tensorflow.keras.models import load_model

from bluegraph.core import NodeEmbedder, DEFAULT_PARAMS

from .ml_utils import (dispatch_node_generator,
                       dispatch_training_params,
                       fit_embedder,
                       fit_attri_embedder)


class StellarGraphNodeEmbedder(NodeEmbedder):
    """Embedder for StellarGraph library."""

    def __init__(self, model_name, default_params=None):
        """Initialize StellarGraphEmbedder."""
        self.model_name = model_name

        # Default training parameters
        if default_params is None:
            default_params = DEFAULT_PARAMS
        self.default_params = default_params

        self._graph = None
        self._embedding_model = None
        self.embeddings = None

    @staticmethod
    def _generate_graph(pgframe, directed=True, include_type=True,
                        feature_prop=None):
        """Convert the PGFrame to a StellarGraph object."""
        feature_array = None
        if include_type:
            nodes = {}
            for t in pgframe.node_types():
                index = pgframe.nodes(typed_by=t)
                if feature_prop:
                    feature_array = np.array(
                        pgframe.get_node_property_values(
                            feature_prop,
                            typed_by=t).to_list())
                nodes[t] = sg.IndexedArray(feature_array, index=index)
        else:
            if feature_prop:
                feature_array = np.array(
                    pgframe.get_node_property_values(
                        feature_prop).to_list())
            nodes = sg.IndexedArray(feature_array, index=pgframe.nodes())

        if pgframe.number_of_edges() > 0:
            edges = pgframe.edges(
                raw_frame=True,
                include_index=True,
                filter_props=lambda x: (x == "@type") if include_type else False,
                rename_cols={'@source_id': 'source', "@target_id": "target"})
        else:
            edges = pd.DataFrame(columns=["source", "target"])

        if directed:
            graph = sg.StellarDiGraph(
                nodes=nodes,
                edges=edges,
                edge_type_column="@type" if include_type else None)
        else:
            graph = sg.StellarGraph(
                nodes=nodes,
                edges=edges,
                edge_type_column="@type" if include_type else None)
        return graph

    def fit_model(self, **kwargs):
        """Fit the model."""
        params = dispatch_training_params(self.default_params, **kwargs)

        if self._graph is None:
            raise ValueError(
                "Graph is not specified, use 'set_graph' before"
                " fitting the model"
            )

        kg_algos = ["ComplEx", "DistMult"]
        attri_algos = ["attri2vec", "GraphSAGE"]

        if self.model_name in kg_algos:
            embeddings = fit_embedder(
                self._graph, params, self.model_name)
            self.embeddings = pd.DataFrame(
                {"embedding": embeddings.tolist()}, index=self._graph.nodes())
        elif self.model_name in attri_algos:
            self._embedding_model = fit_attri_embedder(
                self._graph, params, self.model_name)
            self.embeddings = self.predict_embeddings(self._graph)
        else:
            raise ValueError(
                "Unknown embedding model '{}'.".format(self.model_name))

    def predict_embeddings(self, graph, batch_size=None, num_samples=None):
        """Predict embedding for out-of-sample elements."""
        if self._embedding_model is None:
            raise ValueError(
                "Embedder does not have a predictive model")
        if batch_size is None:
            batch_size = self.default_params["batch_size"]
        if num_samples is None:
            num_samples = self.default_params["num_samples"]
        node_generator = dispatch_node_generator(
            graph, self.model_name, batch_size, num_samples)
        node_embeddings = self._embedding_model.predict(node_generator)
        return pd.DataFrame(
            {"embedding": node_embeddings.tolist()}, index=graph.nodes())

    def save(self, path, compress=True, save_graph=False):
        """Save the embedder."""
        # backup the graph and the model
        graph_backup = self._graph
        model_backup = self._embedding_model

        # remove them for pickling
        if save_graph is False:
            self._graph = None
        self._embedding_model = None

        # create a dir
        if not os.path.isdir(path):
            os.mkdir(path)

        # pickle picklable part of the embedder
        with open(os.path.join(path, "emb.pkl"), "wb") as f:
            pickle.dump(self, f)

        # save the predictive model (using tensorflow)
        if model_backup is not None:
            model_backup.save(os.path.join(path, "model"))

        self._graph = graph_backup
        self._embedding_model = model_backup

        if compress:
            shutil.make_archive(path, 'zip', path)
            shutil.rmtree(path)

    @staticmethod
    def load(path):
        """Load a dumped embedder."""
        decompressed = False
        if re.match("(.+)\.zip", path):
            # decompress
            shutil.unpack_archive(
                path,
                extract_dir=re.match("(.+)\.zip", path).groups()[0])
            path = re.match("(.+)\.zip", path).groups()[0]
            decompressed = True

        with open(os.path.join(path, "emb.pkl"), "rb") as f:
            embedder = pickle.load(f)
        embedder._embedding_model = load_model(
            os.path.join(path, "model"),
            compile=False)

        if decompressed:
            shutil.rmtree(path)

        return embedder

    def get_similar_nodes(self, node_id, number=10, node_subset=None):
        """Get N most similar entities."""
        embeddings = self.embeddings
        if node_subset is not None:
            # filter embeddings
            embeddings = self.embeddings.loc[node_subset]
        if embeddings.shape[0] < number:
            number = embeddings.shape[0]
        search_vec = self.embeddings.loc[node_id]["embedding"]
        matrix = np.matrix(embeddings["embedding"].to_list())
        closest_indices = cKDTree(matrix).query(search_vec, k=number)[1]
        return embeddings.index[closest_indices].to_list()
