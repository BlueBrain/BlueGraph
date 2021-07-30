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

from bluegraph.exceptions import BlueGraphException, BlueGraphWarning
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
        """Initialize the classifier."""
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
        """Fit the classifier."""
        train_elements = self._generate_train_elements(
            pgframe, train_elements, **kwargs)
        labels = self._generate_train_labels(
            pgframe, train_elements, label_prop) if labels is None else labels
        data = self._generate_data_table(pgframe, train_elements)
        self.model.fit(data, labels)

    def predict(self, pgframe, predict_elements=None):
        """Run prediction on the input graph."""
        predict_elements = self._generate_predict_elements(
            pgframe, predict_elements)
        data = self._generate_data_table(pgframe, predict_elements)
        return self.model.predict(data)


class EmbeddingPipeline(object):
    """Data structure for stacking embedding pipelines.

    In this context, an embedding pipeline consists of the
    following steps:

    1) preprocess
    2) embedd
    3) build a similarity index
    """

    def __init__(self, preprocessor=None, embedder=None,
                 similarity_processor=None):
        """Initilize the pipeline."""
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.similarity_processor = similarity_processor

    def is_transductive(self):
        """Return flag indicating if the embedder is transductive."""
        if self.embedder is None:
            return self.preprocessor is None
        else:
            return (
                self.embedder._embedding_model is None
            )

    def is_inductive(self):
        """Return flag indicating if the embedder is inductive."""
        if self.embedder is None:
            return self.preprocessor is not None
        else:
            return self.embedder._embedding_model is not None

    def run_fitting(self, data, index=None, preprocessor_kwargs=None,
                    embedder_kwargs=None):
        """Run fitting of the pipeline components."""
        if preprocessor_kwargs is None:
            preprocessor_kwargs = {}
        if embedder_kwargs is None:
            embedder_kwargs = {}

        # Train the encoder
        if self.preprocessor is not None:
            self.preprocessor.fit(data)
            train_data = self.preprocessor.transform(
                data, **preprocessor_kwargs)
        else:
            train_data = data

        # Train the embedder
        if not self.embedder:
            vectors = train_data
            if index is not None:
                index = pd.Index(index)
        else:
            embedding_table = self.embedder.fit_model(
                train_data, **embedder_kwargs)
            vectors = embedding_table["embedding"].tolist()
            index = embedding_table.index

        # Build the similarity processor from obtained vectors
        self.similarity_processor.add(vectors, index)
        self.similarity_processor.index = index

    def run_prediction(self, data, preprocessor_kwargs=None,
                       embedder_kwargs=None, data_indices=None,
                       add_to_index=False):
        """Run prediction using the pipeline components."""
        if preprocessor_kwargs is None:
            preprocessor_kwargs = {}
        if embedder_kwargs is None:
            embedder_kwargs = {}

        # Encode the data
        if self.preprocessor is not None:
            transformed_data = self.preprocessor.transform(
                data, **preprocessor_kwargs)
        else:
            transformed_data = data

        # Embed
        if not self.embedder:
            vectors = transformed_data
        else:
            embedding_table = self.embedder.predict_embeddings(
                transformed_data, **embedder_kwargs)
            vectors = embedding_table["embedding"].tolist()

        # Add to index if specified
        if add_to_index is True:
            if data_indices is None:
                raise SimilarityProcessor.SimilarityException(
                    "Parameter 'add_to_index' is set to True, "
                    "'data_indices' must be specified")
            self.similarity_processor.add(vectors, data_indices)

        return vectors

    def generate_embedding_table(self):
        """Generate embedding table from similarity index."""
        index = self.similarity_processor.index
        pairs = [
            (ind, self.similarity_processor._model.reconstruct(i))
            for i, ind in enumerate(index)
        ]
        return pd.DataFrame(
            pairs, columns=["@id", "embedding"]).set_index("@id")

    def get_index(self):
        """Get index of existing points."""
        return self.similarity_processor.index

    def retrieve_embeddings(self, indices):
        """Get embedding vectors for the input indices."""
        if self.similarity_processor is None:
            raise EmbeddingPipeline.EmbeddingPipelineException(
                "Similarity processor object is None, cannot "
                "retrieve embedding vectors")
        else:

            return [
                el.tolist() if el is not None else None
                for el in self.similarity_processor.get_vectors(indices)
            ]

    def get_similar_points(self, vectors=None,
                           existing_indices=None, k=10,
                           preprocessor_kwargs=None,
                           embedder_kwargs=None):
        """Get top most similar points for the input indices."""
        return self.similarity_processor.get_similar_points(
            vectors=vectors,
            existing_indices=existing_indices, k=k)

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

        # Load the similarity processor
        similarity_processor = SimilarityProcessor.load(
            os.path.join(path, "similarity.pkl"),
            os.path.join(path, "index.faiss"))

        pipeline = cls(
            preprocessor=encoder,
            embedder=embedder,
            similarity_processor=similarity_processor)

        if decompressed:
            shutil.rmtree(path)

        return pipeline

    def save(self, path, compress=False):
        """Save the pipeline."""
        if not os.path.isdir(path):
            os.mkdir(path)

        # Save the encoder
        with open(os.path.join(path, "encoder.pkl"), "wb") as f:
            pickle.dump(self.preprocessor, f)

        # Save the embedding model
        if self.embedder:
            self.embedder.save(
                os.path.join(path, "embedder"), compress=True)
        else:
            with open(os.path.join(path, "embedder.pkl"), "wb") as f:
                pickle.dump(self.preprocessor, f)

        # Save the similarity processor
        if self.similarity_processor is not None:
            self.similarity_processor.export(
                os.path.join(path, "similarity.pkl"),
                os.path.join(path, "index.faiss"))

        if compress:
            shutil.make_archive(path, 'zip', path)
            shutil.rmtree(path)

    class EmbeddingPipelineException(BlueGraphException):
        """Pipeline exception class."""

        pass

    class EmbeddingPipelineWarning(BlueGraphWarning):
        """Pipeline warning class."""

        pass
