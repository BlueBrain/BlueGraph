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
import pandas as pd

import os
import re
import pickle
import shutil

from bluegraph.exceptions import BlueGraphException, BlueGraphWarning
from .similarity import SimilarityProcessor


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

    def run_fitting(self, data, point_ids=None, preprocessor_kwargs=None,
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
            if point_ids is not None:
                point_ids = pd.Index(point_ids)
        else:
            embedding_table = self.embedder.fit_model(
                train_data, **embedder_kwargs)
            vectors = embedding_table["embedding"].tolist()
            point_ids = embedding_table.index

        # Build the similarity processor from obtained vectors
        self.similarity_processor.add(vectors, point_ids)
        self.similarity_processor.point_ids = point_ids

    def run_prediction(self, data, preprocessor_kwargs=None,
                       embedder_kwargs=None, data_point_ids=None,
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
            if data_point_ids is None:
                raise SimilarityProcessor.SimilarityException(
                    "Parameter 'add_to_index' is set to True, "
                    "'data_point_ids' must be specified")
            self.similarity_processor.add(vectors, data_point_ids)

        return vectors

    def generate_embedding_table(self):
        """Generate embedding table from similarity index."""
        point_ids = self.similarity_processor.point_ids
        pairs = [
            (ind, self.similarity_processor.index.reconstruct(i))
            for i, ind in enumerate(point_ids)
        ]
        return pd.DataFrame(
            pairs, columns=["@id", "embedding"]).set_index("@id")

    def get_point_ids(self):
        """Get index of existing points."""
        return self.similarity_processor.point_ids

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

    def get_neighbors(self, vectors=None,
                      existing_points=None, k=10,
                      preprocessor_kwargs=None,
                      embedder_kwargs=None):
        """Get top most similar points for the input indices."""
        return self.similarity_processor.get_neighbors(
            vectors=vectors,
            existing_points=existing_points, k=k)

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
