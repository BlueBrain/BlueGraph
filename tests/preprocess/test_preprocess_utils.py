from bluegraph.preprocess.utils import Doc2VecEncoder, TfIdfEncoder
import pandas as pd


def test_tfidf(random_text_corpus):
    encoder = TfIdfEncoder({"max_features": 100})
    encoder.fit(random_text_corpus)
    encoder.transform(random_text_corpus)
    encoder.info()


def test_word2vec(random_text_corpus):
    encoder = Doc2VecEncoder(size=10, window=5)
    encoder.fit(random_text_corpus)
    encoder.transform(random_text_corpus)
    encoder.info()
