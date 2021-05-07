from bluegraph.preprocess.utils import Doc2VecEncoder
import pandas as pd
import random

from nltk.corpus import words


def test_word2vec():
    corpus = pd.DataFrame(
        [
            (i, ' '.join(random.sample(words.words(), 20)))
            for i in range(100)
        ],
        columns=["@id", "desc"]
    )
    # size=64, window=6, min_count=1, workers=4
    encoder = Doc2VecEncoder(size=10, window=5)
    encoder.fit(corpus)
    encoder.transform(corpus)
