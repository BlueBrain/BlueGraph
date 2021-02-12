"""Collection of utils."""
import string
import collections
import math

import nltk
from nltk.corpus import stopwords

from scipy import sparse

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split


def str_to_set(s):
    """Parse string representation of a set."""
    if s[0] == "{":
        s = s[1:-1]
        return set([t.strip()[1:-1] for t in s.split(",")])
    return s


def is_nan(value):
    """Check if the value is nan."""
    if isinstance(value, float):
        return math.isnan(value)
    else:
        return False


def _aggregate_values(values):
    value_set = set()
    for el in values:
        if isinstance(el, set):
            value_set.update(el)
        elif not is_nan(el):
            value_set.add(el)
    if len(value_set) == 1:
        return list(value_set)[0]
    elif len(value_set) == 0:
        return math.nan
    return value_set


def element_has_type(element_type, query_type):
    if not isinstance(element_type, set):
        element_type = {element_type}
    if not isinstance(query_type, set):
        query_type = {query_type}
    return query_type.issubset(element_type)


def normalize_to_set(x):
    """Normalize the value to a set."""
    if isinstance(x, set):
        return x
    elif isinstance(x, float) and math.isnan(x):
        return set()
    else:
        return {x}


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


class Word2VecModel(object):
    """Wrapper around word2vec providing scikit-learn like interface."""

    def __init__(self, size=64, window=6, min_count=1, workers=4):
        """Initialize a model."""
        self._model = None
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def fit(self, corpus):
        """Fit a word2vec model."""
        tokenized_corpus = [
            TaggedDocument(tokenize_text(text), [i])
            for i, text in enumerate(corpus)
        ]
        self._model = Doc2Vec(
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
            result.append(self._model.infer_vector(s))
        return sparse.csc_matrix(result)

    def fit_transform(self, input):
        """Fit and transform the text."""
        pass


def graph_train_test_split(pgframe, test_size=None, random_state=42):
    train_test_split(
        pgframe._nodes, binarized_types,
        test_size=test_size, random_state=42)