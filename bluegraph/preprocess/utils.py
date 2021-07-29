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
"""Collection of utils."""
import string
import collections

import numpy as np
import nltk
from nltk.corpus import stopwords

from scipy import sparse

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer

from bluegraph.core.utils import Preprocessor


def _get_encoder_type(pgframe, prop, is_edge=False):
    encoder_type = "category"
    if is_edge is False:
        if pgframe.is_numeric_node_prop(prop):
            encoder_type = "numeric"
        elif pgframe.is_text_node_prop(prop):
            encoder_type = "text"
    else:
        if pgframe.is_numeric_edge_prop(prop):
            encoder_type = "numeric"
        elif pgframe.is_text_edge_prop(prop):
            encoder_type = "text"
    return encoder_type


def _generate_type_repr(element_type):
    if not isinstance(element_type, collections.Iterable) or\
       isinstance(element_type, str):
        element_type_repr = element_type
    else:
        element_type_repr = tuple(sorted(element_type))
    return element_type_repr


def tokenize_text(text):
    """Tokenize text."""
    tokens = nltk.word_tokenize(text)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in tokens]

    stop_words = set(stopwords.words('english'))
    words = [
        w
        for w in words
        if w not in stop_words and len(w) > 0]
    return words


class TfIdfEncoder(Preprocessor):
    """Wrapper around tf-idf providing Preprocessor interface."""

    def __init__(self, params=None):
        """Initialize TfIdfTermEncoder."""
        if params is None:
            params = {}
        self.model = TfidfVectorizer(**params)

    def info(self):
        """Get model info."""
        return self.model.get_params()

    def fit(self, data):
        """Fit the model."""
        self.model.fit(data)

    def transform(self, data):
        """Transform the input data."""
        return np.array(self.model.transform(data).todense())


class Doc2VecEncoder(Preprocessor):
    """Wrapper around doc2vec providing Preprocessor interface."""

    def __init__(self, size=64, window=6, min_count=1, workers=4):
        """Initialize a model."""
        self.model = None
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def info(self):
        """Get model info."""
        return {
            "dm": self.model.dm,
            "vector_size": self.model.vector_size,
            "window": self.model.window,
            "alpha": self.model.alpha,
            "min_alpha": self.model.min_alpha,
            "min_count": self.model.vocabulary.min_count,
            "max_vocab_size": self.model.vocabulary.max_vocab_size,
            "sample": self.model.vocabulary.sample,
            "epochs": self.model.epochs,
            "hs": self.model.hs,
            "negative": self.model.negative,
            "ns_exponent": self.model.ns_exponent,
            "dm_concat": self.model.dm_concat,
            "dm_tag_count": self.model.dm_tag_count,
            "dbow_words": self.model.dbow_words
        }

    def fit(self, corpus):
        """Fit a word2vec model."""
        tokenized_corpus = [
            TaggedDocument(tokenize_text(text), [i])
            for i, text in enumerate(corpus)
        ]
        self.model = Doc2Vec(
            tokenized_corpus, vector_size=self.size, window=self.window,
            min_count=self.min_count, workers=self.workers)

    def transform(self, input):
        """Transform a text into a vector."""
        tokenized_input = [
            tokenize_text(text)
            for i, text in enumerate(input)
        ]
        result = []
        for s in tokenized_input:
            result.append(self.model.infer_vector(s))
        return sparse.csc_matrix(result)
