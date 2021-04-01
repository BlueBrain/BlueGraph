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
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import os
import re
import pickle
import shutil

from .similarity import SimilarityProcessor


class ElementClassifier(ABC):
    """Interface for graph element classification models.

    It wraps a predictive classification model provided by the user
    and a set of configs that allow the user to fit the model
    and make predictions on the input PGFrames. Its main goal is to
    hide the details on converting element (node or edge) properties
    into data tables that can be provided to the predictive model.
    """
    def __init__(self, model, feature_vector_prop=None, feature_props=None,
                 **kwargs):
        self.model = model
        self.feature_vector_prop = feature_vector_prop
        self.feature_props = feature_props

    def _concatenate_feature_props(self, pgframe, nodes):
        if self.feature_props is None or len(self.feature_props) == 0:
            raise ValueError
        return pgframe.nodes(
            raw_frame=True).loc[nodes, self.feature_props].to_numpy()

    def _get_node_features(self, pgframe, nodes):
        if self.feature_vector_prop:
            features = pgframe.get_node_property_values(
                self.feature_vector_prop, nodes=nodes).tolist()
        else:
            features = self._concatenate_feature_props(pgframe, nodes)
        return np.array(features)

    @abstractmethod
    def _generate_train_elements(self, pgfame, elements=None):
        pass

    @abstractmethod
    def _generate_predict_elements(self, pgfame, elements=None):
        pass

    @abstractmethod
    def _generate_train_labels(self, pgframe, elements, label_prop=None):
        pass

    @abstractmethod
    def _generate_data_table(self, pgframe, elements):
        pass

    def fit(self, pgframe, train_elements=None, labels=None, label_prop=None,
            **kwargs):
        train_elements = self._generate_train_elements(
            pgframe, train_elements, **kwargs)
        labels = self._generate_train_labels(
            pgframe, train_elements, label_prop) if labels is None else labels
        data = self._generate_data_table(pgframe, train_elements)
        self.model.fit(data, labels)

    def predict(self, pgframe, predict_elements=None):
        predict_elements = self._generate_predict_elements(
            pgframe, predict_elements)
        data = self._generate_data_table(pgframe, predict_elements)
        return self.model.predict(data)


class Preprocessor(ABC):
    """Preprocessor inferface for EmbeddingPipeline."""

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass


class Embedder(ABC):
    """Embedder inferface for EmbeddingPipeline."""

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def fit_model(self):
        pass


class EmbeddingPipeline(object):

    def __init__(self, preprocessor=None, embedder=None,
                 embedding_table=None,
                 similarity_processor=None):
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.embedding_table = embedding_table
        self.similarity_processor = similarity_processor

    def is_transductive(self):
        return self.embedder is None or self.embedder._embedding_model is None

    def is_inductive(self):
        return self.embedder is not None and self.embedder._embedding_model is not None

    def run_fitting(self, data):
        # Encode
        if self.preprocessor is not None:
            self.preprocessor.fit(data)
            train_data = self.preprocessor.transform(data)
        else:
            train_data = data
        # Train the embedder
        self.embedding_table = self.embedder.fit_model(train_data)
        # Create a similarity processor
        vectors =\
            self.embedding_table["embedding"].tolist()
        self.similarity_processor._initialize_model(vectors)
        self.similarity_processor.add(vectors, self.embedding_table.index)
        self.similarity_processor.index = self.embedding_table.index

    def run_prediction(self, data):
        pass

    def retrieve_embeddings(self, indices):
        if self.embedding_table is not None:
            return self.embedding_table.loc[indices]["embedding"].tolist()
        else:
            return [
                el.tolist()
                for el in self.similarity_processor.get_vectors(indices)
            ]

    def get_similar_points(self, indices, k=10):
        return self.similarity_processor.get_similar_points(
            existing_indices=indices, k=k)

    @classmethod
    def load(cls, path, embedder_interface=None, embedder_ext="pkl"):
        """Load a dumped embedding pipeline."""
        decompressed = False
        if re.match(r"(.+)\.zip", path):
            # decompress
            shutil.unpack_archive(
                path,
                extract_dir=re.match(r"(.+)\.zip", path).groups()[0])
            path = re.match(r"(.+)\.zip", path).groups()[0]
            decompressed = True

        # Load the encoder
        encoder = None
        with open(os.path.join(path, "encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)

        # Load the model
        embedder = None
        extension = f".{embedder_ext}" if embedder_ext else ""
        if os.path.isfile(os.path.join(path, f"embedder{extension}")):
            if embedder_interface is None:
                with open(os.path.join(path, f"embedder{extension}"), "rb") as f:
                    embedder = pickle.load(f)
            else:
                embedder = embedder_interface.load(
                    os.path.join(path, "embedder.zip"))

        # Load the embedding table
        embedding_table = None
        if os.path.isfile(os.path.join(path, "vectors.pkl")):
            embedding_table = pd.read_pickle(
                os.path.join(path, "vectors.pkl"))

        # Load the similarity processor
        similarity_processor = SimilarityProcessor.load(
            os.path.join(path, "similarity.pkl"),
            os.path.join(path, "index.faiss"))

        pipeline = cls(
            preprocessor=encoder,
            embedder=embedder,
            embedding_table=embedding_table,
            similarity_processor=similarity_processor)

        if decompressed:
            shutil.rmtree(path)

        return pipeline

    def save(self, path, compress=False):

        if not os.path.isdir(path):
            os.mkdir(path)

        # Save the encoder
        with open(os.path.join(path, "encoder.pkl"), "wb") as f:
            pickle.dump(self.preprocessor, f)

        # Save the embedding model
        self.embedder.save(
            os.path.join(path, "embedder"), compress=True)

        # Save the embedding table
        self.embedding_table.to_pickle(
            os.path.join(path, "vectors.pkl"))

        # Save the similarity processor
        if self.similarity_processor is not None:
            self.similarity_processor.export(
                os.path.join(path, "similarity.pkl"),
                os.path.join(path, "index.faiss"))

        if compress:
            shutil.make_archive(path, 'zip', path)
            shutil.rmtree(path)
