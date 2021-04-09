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
"""Collection of property graph encoders."""
from abc import ABC, abstractmethod

import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from .utils import (Word2VecModel,
                    _get_encoder_type,
                    _generate_type_repr)
from bluegraph.core.utils import normalize_to_set
from bluegraph.core.io import PandasPGFrame
from bluegraph.exceptions import BlueGraphException


class SemanticPGEncoder(ABC):
    """Abstract class for semantic property graph encoder.

    The encoder provides a wrapper for multiple heterogeneous
    models for encoding various node/edge properties (of different
    data types) into numerical vectors. It supports three types of
    properties: categorical properties, text properties and
    numerical properties.

    TODO: Make it concrete by allowing to specify custom encoding
    models for different property types (?)
    """

    # ----------- Abstract methods ----------

    @staticmethod
    @abstractmethod
    def _create_pgframe(nodes=None, edges=None):
        """Create a PGFrame object."""
        pass

    @staticmethod
    @abstractmethod
    def _concatenate_features(features):
        """Concatenate arrays of numerical features."""
        pass

    @abstractmethod
    def _fit_encoder(self, frame, prop, encoder_type="category"):
        """Fit an encoder for the provided property."""
        pass

    @abstractmethod
    def _apply_encoder(self, frame, prop, encoder, encoder_type="category"):
        """Apply the specified encoder to the provided property."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the encoder to the file."""
        pass

    @abstractmethod
    def load(self, path):
        """Load the encoder from the file."""
        pass

    @abstractmethod
    def _infer_prop_encoder_type(self, prop, is_edge=False, element_type=None):
        pass

    # ----------- Concrete methods ----------

    def __init__(self,  node_properties=None, edge_properties=None,
                 heterogeneous=False, drop_types=False,
                 encode_types=False, edge_features=False,
                 categorical_encoding="multibin",
                 text_encoding="tfidf",
                 text_encoding_max_dimension=128,
                 missing_numeric="drop",
                 imputation_strategy="mean",
                 standardize_numeric=True):
        """Initialize an encoder.

        Parameters
        ----------
        node_properties : list, optional
        node_properties : list, optional
        heterogeneous : bool, optional
            Flag indicating if the feature space is heterogeneous accross
            different node/edge types. False by default.
        encode_types : bool, optional
            Flag indicating if node/edge types should be included in
            the generated features. False by default.
        categorical_encoding : str, optional
            Strategy for encoding categorical values (possible input
            'multibin', ..). By default is Multilabel binarizer.
        text_encoding : str, optional
            Strategy for encoding textual values (possible input 'tfidf',
            'word2vec'). By default is TfIdf.
        missing_numeric : str, optional.
            Strategy for treating properties with missing numeric values.
            By default such columns are dropped.
        imputation_strategy : str, optional
        standardize_numeric : str, optional
            Flag indicating if numerical values should be standardized
        """
        self.heterogeneous = heterogeneous
        self.drop_types = drop_types
        self.encode_types = encode_types
        self.edge_features = edge_features
        self.categorical_encoding = categorical_encoding
        self.text_encoding = text_encoding
        self.text_encoding_max_dimension = text_encoding_max_dimension
        self.missing_numeric = missing_numeric
        self.imputation_strategy = imputation_strategy
        self.standardize_numeric = standardize_numeric

        self._node_encoders = {}
        if node_properties is not None:
            if self.heterogeneous:
                if not isinstance(node_properties, dict):
                    raise SemanticPGEncoder.EncodingException(
                        "Encoder is heterogeneous, specified node properties "
                        "should be a dictionary whose keys are node types "
                        "and whose values are sets of properties to encode.")
                for t, props in node_properties.items():
                    self._node_encoders[t] = {}
                    for p in props:
                        self._node_encoders[t][p] = None
            else:
                for p in node_properties:
                    self._node_encoders[p] = None
        self._edge_encoders = {}
        if edge_properties is not None:
            if self.heterogeneous:
                if not isinstance(edge_properties, dict):
                    raise SemanticPGEncoder.EncodingException(
                        "Encoder is heterogeneous, specified edge properties "
                        "should be a dictionary whose keys are edge types "
                        "and whose values are sets of properties to encode.")
                for t, props in edge_properties.items():
                    self._edge_encoders[t] = {}
                    for p in props:
                        self._edge_encoders[t][p] = None
            else:
                for p in edge_properties:
                    self._edge_encoders[p] = None

    def info(self):
        if self.heterogeneous:
            node_properties = {}
            for k, v in self._node_encoders.items():
                node_properties[k] = list(v.keys())
            edge_properties = None
            if self.edge_features:
                edge_properties = {}
                for k, v in self._edge_encoders.items():
                    edge_properties[k] = list(v.keys())
        else:
            node_properties = list(self._node_encoders.keys())
            edge_properties = (
                list(self._edge_encoders.keys())
                if self.edge_features else None
            )
        info = {
            "heterogeneous": self.heterogeneous,
            "drop_types": self.drop_types,
            "encode_types": self.encode_types,
            "edge_features": self.edge_features,
            "categorical_encoding": self.categorical_encoding,
            "text_encoding": self.text_encoding,
            "text_encoding_max_dimension": self.text_encoding_max_dimension,
            "missing_numeric": self.missing_numeric,
            "imputation_strategy": self.imputation_strategy,
            "standardize_numeric": self.standardize_numeric,
            "node_properties": node_properties,
            "edge_properties": edge_properties
        }
        return info

    def fit(self, pgframe):
        """Fit encoders for node and edge properties."""
        if self.heterogeneous:
            for node_type in pgframe.node_types():
                node_type_repr = _generate_type_repr(node_type)
                self._node_encoders[node_type_repr] = {}

                for prop in self._node_encoders[node_type]:
                    self._node_encoders[
                        node_type_repr][prop] = self._fit_encoder(
                            pgframe.nodes(
                                typed_by=node_type, raw_frame=True), prop,
                        _get_encoder_type(pgframe, prop))
            if self.edge_features:
                for edge_type in pgframe.edge_types():
                    edge_type_repr = _generate_type_repr(edge_type)
                    self._edge_encoders[edge_type_repr] = {}

                    for prop in self._edge_encoders[edge_type]:
                        self._edge_encoders[
                            edge_type_repr][prop] = self._fit_encoder(
                                pgframe.edges(
                                    typed_by=edge_type, raw_frame=True), prop,
                                _get_encoder_type(pgframe, prop, is_edge=True))
        else:
            for prop in self._node_encoders:
                self._node_encoders[prop] = self._fit_encoder(
                    pgframe.nodes(raw_frame=True), prop,
                    _get_encoder_type(pgframe, prop))

            if self.edge_features:
                for prop in self._edge_encoders:
                    self._edge_encoders[prop] = self._fit_encoder(
                        pgframe.edges(raw_frame=True), prop,
                        _get_encoder_type(pgframe, prop, is_edge=True))

    def transform(self, pgframe):
        """Transform the input PGFrame."""
        transformed_pgframe = self._create_pgframe(
            nodes=pgframe.nodes(), edges=pgframe.edges())

        if not self.drop_types:
            if pgframe.has_node_types():
                transformed_pgframe.assign_node_types(
                    pgframe.get_node_typing())
            if pgframe.has_edge_types():
                transformed_pgframe.assign_edge_types(
                    pgframe.get_edge_typing())

        def _aggregate_encoding_of(frame, encoder, aggregation_handler,
                                   is_edge=False, element_type=None):
            if prop not in frame.columns:
                frame[prop] = math.nan

            encoded_prop = self._apply_encoder(
                frame, prop, encoder, self._infer_prop_encoder_type(
                    prop, is_edge, element_type=element_type))
            if encoded_prop is not None:
                if prop == "@type":
                    encoded_prop = encoded_prop.rename(
                        columns={"@type": "encoded_type"})
                aggregation_handler(encoded_prop)

        if self.heterogeneous:
            # Encode nodes
            for node_type in pgframe.node_types():
                node_type_repr = _generate_type_repr(node_type)
                for prop, encoder in self._node_encoders[node_type_repr].items():
                    _aggregate_encoding_of(
                        pgframe.nodes(typed_by=node_type, raw_frame=True),
                        encoder,
                        transformed_pgframe.add_node_properties,
                        element_type=node_type)

            # Encode edges
            for edge_type in pgframe.edge_types():
                edge_type_repr = _generate_type_repr(edge_type)
                if edge_type_repr in self._edge_encoders:
                    for prop, encoder in self._edge_encoders[edge_type_repr].items():
                        _aggregate_encoding_of(
                            pgframe.edges(typed_by=edge_type, raw_frame=True),
                            encoder,
                            transformed_pgframe.add_edge_properties,
                            True,
                            element_type=edge_type)
        else:
            # Encode nodes
            for prop, encoder in self._node_encoders.items():
                _aggregate_encoding_of(
                    pgframe.nodes(raw_frame=True),
                    encoder,
                    transformed_pgframe.add_node_properties)
            # Encode edges
            for prop, encoder in self._edge_encoders.items():
                _aggregate_encoding_of(
                    pgframe.edges(raw_frame=True),
                    encoder,
                    transformed_pgframe.add_edge_properties,
                    True)

        # Aggregate feature vectors constructed for different properties
        # into a single large vector
        transformed_pgframe.aggregate_node_properties(
            self._concatenate_features, into="features")

        if self.edge_features:
            transformed_pgframe.aggregate_edge_properties(
                self._concatenate_features, into="features")

        return transformed_pgframe

    def fit_transform(self, pgframe):
        """Fit the encoder and transform the input PGFrame."""
        self.fit(pgframe)
        return self.transform(pgframe)


class ScikitLearnPGEncoder(SemanticPGEncoder):
    """Scikit-learn-based in-memory property graph encoder.

    The encoder provides a wrapper for multiple heterogeneous
    models for encoding various node/edge properties of different
    data types into numerical vectors. It supports the following
    encoders:

    - for categorical properties: MultiLabelBinarizer
    - for text properties: TfIdf, word2vec
    - for numerical properties: standard scaler
    """

    # ---------------- Implementation of abstract methods --------------
    @staticmethod
    def _create_pgframe(nodes=None, edges=None):
        return PandasPGFrame(nodes=nodes, edges=edges)

    @staticmethod
    def _concatenate_features(features):
        non_empty_vectors = features[features.notna()].to_list()
        if len(non_empty_vectors) == 0:
            return math.nan
        return np.concatenate(non_empty_vectors)

    def _fit_encoder(self, frame, prop, encoder_type="category"):
        """Create and fit an encoder according to the property type."""
        encoder = None
        if encoder_type == "text":
            if self.text_encoding == "tfidf":
                encoder = self._fit_tfidf(
                    frame[prop], max_dim=self.text_encoding_max_dimension)
            elif self.text_encoding == "word2vec":
                encoder = self._fit_word2vec(frame[prop])
        elif encoder_type == "category":
            encoder = self._fit_multibin(frame[prop])
        elif encoder_type == "numeric":
            if self.standardize_numeric:
                encoder = self._fit_standard_scaler(
                    frame[prop],
                    missing_numeric=self.missing_numeric,
                    imputation_strategy=self.imputation_strategy)
        return encoder

    def _apply_encoder(self, frame, prop, encoder, encoder_type="category"):
        """Apply the input encoder to the property."""
        vectors = None
        if encoder_type == "category":
            vectors = encoder.transform(frame[prop].apply(normalize_to_set))
        elif encoder_type == "text":
            column = frame[prop].copy()
            column[column.isna()] = " "
            vectors = encoder.transform(column).todense()
        elif encoder_type == "numeric" and encoder is not None:
            vectors = encoder.transform(
                np.reshape(frame[prop].to_list(), (frame[prop].shape[0], 1)))
        if vectors is not None:
            df = pd.DataFrame(columns=[prop], index=frame.index)
            df[prop] = vectors.tolist()
            return df

    def save(self, path):
        """Save the encoder to the file."""
        pass

    def load(self, path):
        """Load the encoder from the file."""
        pass

    def _infer_prop_encoder_type(self, prop, is_edge=False, element_type=None):
        encoder = None

        if element_type is None:
            if not is_edge and prop in self._node_encoders:
                encoder = self._node_encoders[prop]
            elif prop in self._edge_encoders:
                encoder = self._edge_encoders[prop]
        else:
            if not is_edge and prop in self._node_encoders[element_type]:
                encoder = self._node_encoders[element_type][prop]
            elif prop in self._edge_encoders[element_type]:
                encoder = self._edge_encoders[element_type][prop]

        if encoder:
            if isinstance(encoder, MultiLabelBinarizer):
                return "category"
            elif isinstance(encoder, TfidfVectorizer) or\
                    isinstance(encoder, Word2VecModel):
                return "text"
            elif isinstance(encoder, StandardScaler) or\
                    isinstance(encoder, Pipeline):
                return "numeric"

    # ----------------------- Custom methods ----------------

    @staticmethod
    def _fit_multibin(series, transform=False):
        encoder = MultiLabelBinarizer()
        encoder.fit(series.apply(normalize_to_set))
        return encoder

    @staticmethod
    def _fit_tfidf(series, transform=False, max_dim=None):
        corpus = series[series.apply(lambda x: isinstance(x, str))]
        encoder = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word',
            stop_words='english',
            max_features=max_dim)
        encoder.fit(corpus)
        return encoder

    @staticmethod
    def _fit_word2vec(series, transform=False):
        corpus = series[series.apply(lambda x: isinstance(x, str))]
        encoder = Word2VecModel()
        encoder.fit(corpus)
        return encoder

    @staticmethod
    def _fit_standard_scaler(series, transform=False,
                             missing_numeric="drop",
                             imputation_strategy="mean"):
        imputer = None
        if series.isna().any():
            if missing_numeric == "drop":
                return None
            elif missing_numeric == "impute":
                imputer = SimpleImputer(
                    missing_values=math.nan,
                    strategy=imputation_strategy)
        scaler = StandardScaler()
        if imputer is not None:
            encoder = Pipeline([('imputer', imputer), ('scaler', scaler)])
        else:
            encoder = scaler

        encoder.fit(
            np.reshape(series.to_list(), (series.shape[0], 1)))
        return encoder

    class EncodingException(BlueGraphException):
        pass
