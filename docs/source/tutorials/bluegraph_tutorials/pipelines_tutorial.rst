.. _pipeline_tutorial:


Creating and running embedding pipelines
========================================


``bluegraph`` allows to create emebedding pipelines (using the
``EmbeddingPipeline`` class) that represent a useful wrapper around a
sequence of steps necessary to produce embeddings and compute point
similarities. In the examples below we create a pipeline for encoding text properties of nodes into feature vectors, producing ``attri2vec`` node embeddings and computing their similarity based on two different similarity backends. The source notebook can be found `here <https://github.com/BlueBrain/BlueGraph/blob/master/examples/notebooks/Create%20and%20run%20embedding%20pipelines.ipynb>`_.



.. code:: ipython3

    from bluegraph.core import PandasPGFrame
    
    from bluegraph.preprocess.encoders import ScikitLearnPGEncoder
    from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
    from bluegraph.downstream.similarity import (SimilarityProcessor,
                                                 FaissSimilarityIndex,
                                                 ScikitLearnSimilarityIndex,
                                                 SimilarityIndex)
    from bluegraph.downstream import EmbeddingPipeline

Example 1: creating pipeline trainable with ``run_fitting``
-----------------------------------------------------------

We first create an encoder object that will be used in our pipeline to
encode node property ``definition`` using a TfIdf encoder.

.. code:: ipython3

    definition_encoder = ScikitLearnPGEncoder(
        node_properties=["definition"],
        text_encoding_max_dimension=512,
        text_encoding="tfidf")

We then create an embedder object that can compute node embeddings for
input graphs using ``attri2vec`` node embedding technique.

.. code:: ipython3

    D = 128
    params = {
        "length": 5,
        "number_of_walks": 10,
        "epochs": 5,
        "embedding_dimension": D
    }
    attri2vec_embedder = StellarGraphNodeEmbedder(
        "attri2vec", feature_vector_prop="features", edge_weight="npmi", **params)

Next, we create a similarity processor based of Faiss indices that
allows us to perform fast search for nearest neighbors according to our
embedding vectors. We set our similarity measure to *cosine similarity*.

**Note:** in the code below we use the ``SimilarityProcessor`` interface
and not ``NodeSimilarityProcessor``, as we have done it in previous
tutorials. We use this lower abstraction level interface, because the
``EmbeddingPipeline`` is designed to work with any embedding models (not
only node embedding models).

.. code:: ipython3

    similarity_processor = SimilarityProcessor(
        FaissSimilarityIndex(
            similarity="cosine", dimension=D, n_segments=5))

And finally we create a pipeline object that stacks all the
above-mentioned elements.

.. code:: ipython3

    attri2vec_pipeline = EmbeddingPipeline(
        preprocessor=definition_encoder,
        embedder=attri2vec_embedder,
        similarity_processor=similarity_processor)

Now, let us load the training graph from the provided example dataset.

.. code:: ipython3

    graph = PandasPGFrame.load_json("../data/cooccurrence_graph.json")

We run the fitting process, which given the input data performs the
following steps: 1. fits the encoder 2. transforms the data 3. fits the
embedder 4. produces the embedding table 5. fits the similarity index

.. code:: ipython3

    attri2vec_pipeline.run_fitting(graph)



We can save our pipeline to the file system as follows:

.. code:: ipython3

    attri2vec_pipeline.save(
        "../data/attri2vec_test_model",
        compress=True)


We can launch prediction of the unseen graph nodes using our pipeline as
follows (in this case we use the same graph). As an output, we obtain
embedding vectors produced by the model.

.. code:: ipython3

    vectors = attri2vec_pipeline.run_prediction(graph)

Example 2: creating manually trained pipeline
---------------------------------------------

In the previous example we used ``FaissSimilarityIndex`` and the backend
for our nearest neighbors search. ``Faiss`` indices are updatable and
allow us to add new points to the index at any point. Therefore, we were
able to create an ‘untrained’ pipeline stacking preprocessor, embedder
and empty similarity index. We then run all the training steps at once
by using ``run_fitting``. As the result, vectors output by the embedder
were added to the index, once they were produced.

However, in some cases, similarity indices are static and the set of
vectors on which they are built must be provided at the creation time.
Consider the following example.

We would like to use
`BallTree <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html>`__
index implemented in ``scikit-learn`` and provided by ``bluegraph``\ ’s
``ScikitLearnSimilarityIndex``. In the cell below we try to initialize
this index without initial vectors on which it must be built.

.. code:: ipython3

    try:
        sklearn_similarity_processor = SimilarityProcessor(
            ScikitLearnSimilarityIndex(
                similarity="poincare", dimension=D,
                index_type="ballktree", leaf_size=10)
        )
    except SimilarityIndex.SimilarityException as e:
        print("Caught the following error: ")
        print(e)


.. parsed-literal::

    Caught the following error: 
    Initial vectors must be provied (scikit learn indices are not updatable) 


This means that we cannot create an initially empty similarity index and
let our pipeline fill it with vectors once the embedder has output the
them. What we can do instead is run encoding and embedding manually, as
follows:

.. code:: ipython3

    transformed_graph = definition_encoder.fit_transform(graph)
    embedding = attri2vec_embedder.fit_model(transformed_graph)


.. parsed-literal::

    link_classification: using 'ip' method to combine node embeddings into edge embeddings


We now can create a similarity index on the produced embedding vectors.

.. code:: ipython3

    sklearn_similarity_processor = SimilarityProcessor(
        ScikitLearnSimilarityIndex(
            similarity="poincare", dimension=D,
            initial_vectors=embedding["embedding"].tolist(),
            index_type="ballktree", leaf_size=10))

And, finally, stack our steps into a pipeline that can be dumped and
re-used as in the previous example.

.. code:: ipython3

    attri2vec_sklearn_pipeline = EmbeddingPipeline(
        preprocessor=definition_encoder,
        embedder=attri2vec_embedder,
        similarity_processor=sklearn_similarity_processor)
