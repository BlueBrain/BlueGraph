"""Model parameters and default parameter values for node embedders."""

from bluegraph.core.embed.embedders import DEFAULT_EMBEDDING_DIMENSION

NEO4j_PARAMS = {
    "fastrp": [
        "embeddingDimension",
        "iterationWeights",
        "normalizationStrength"
    ],
    "node2vec": [
        "walkLength",
        "walksPerNode",
        "windowSize",
        "walkBufferSize",
        "inOutFactor",
        "returnFactor",
        "negativeSamplingRate",
        "centerSamplingFactor",
        "contextSamplingExponent",
        "embeddingDimension",
        "initialLearningRate",
        "minLearningRate",
        "iterations"
    ],
    "graphsage": [
        "embeddingDimension",
        "activationFunction",
        "sampleSizes",
        "projectedFeatureDimension",
        "batchSize",
        "tolerance",
        "learningRate",
        "epochs",
        "maxIterations",
        "searchDepth",
        "negativeSampleWeight",
    ]
}


DEFAULT_NEO4j_PARAMS = {
    "embeddingDimension": DEFAULT_EMBEDDING_DIMENSION,
}


STELLARGRAPH_PARAMS = {
    "transductive": [
        "embedding_dimension",
        "batch_size",
        "negative_samples",
        "epochs",
        "length",
        "num_samples",
        "number_of_walks",
        "random_walk_p",
        "random_walk_q",
        "num_powers"
    ],
    "inductive": [
        "embedding_dimension",
        "length",
        "number_of_walks",
        "batch_size",
        "epochs",
        "num_samples",
        "clusters",  # number of random clusters
        "clusters_q"  # number of clusters to combine for each mini-batch
    ]
}


DEFAULT_STELLARGRAPH_PARAMS = {
    "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
    "batch_size": 20,
    "negative_samples": 10,
    "epochs": 5,
    "length": 5,  # maximum length of a random walk
    "number_of_walks": 4,  # number of random walks per root node
    "num_samples": [10, 5],
    "random_walk_p": 0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    "random_walk_q": 2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    "clusters": 2,
    "clusters_q": 1,
    "num_powers": 10
}


GENSIM_PARAMS = {
    "poincare": [
        "epochs",
        "size",
        "alpha",
        "negative",
        "workers",
        "epsilon",
        "regularization_coeff",
        "burn_in",
        "burn_in_alpha",
        "init_range",
        "dtype",
        "seed"
    ]
}


DEFAULT_GENSIM_PARAMS = {
    "size": 64,
    "epochs": 50
}