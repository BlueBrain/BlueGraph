==========
Blue Graph
==========

|Travis_badge|

Unifying Python framework for graph analytics and co-occurrence analysis.


.. image:: examples/figures/BBP_Blue_Graph_banner.jpg
  :width: 300
  :alt: BlueGraph banner


About
-----

BlueGraph is a Python framework that consolidates graph analytics capabilities from different graph processing backends. It provides the following set of interfaces:

- preprocessing and co-occurrence analysis API providing semantic property encoders and co-occurrence graph generators;
- graph analytics API providing interfaces for computing graph metrics, performing path search and community detection;
- representation learning API for applying various graph embedding techniques;
- representation learning downstream tasks API allowing the user to perform node classification, similarity queries, link prediction.


Using the built-in `PGFrame` data structure (currently, `pandas <https://pandas.pydata.org/>`_-based implementation is available) for representing property graphs, it provides a backend-agnostic API supporting the following in-memory and persistent graph backends:

- `NetworkX <https://networkx.org/>`_ (for the analytics API)
- `graph-tool <https://graph-tool.skewed.de/>`_ (for the analytics API)
- `Neo4j <https://neo4j.com/>`_ (for the analytics and representation learning API);
- `StellarGraph <https://stellargraph.readthedocs.io/en/stable/>`_ (for the representation learning API).


This repository originated from the Blue Brain effort on building a COVID-19-related knowledge graph from the `CORD-19 <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_ dataset. 


:code:`bluegraph` package
-------------------------

BlueGraph's API is built upon 4 main packages:

- `bluegraph.core` providing the exchange data structure for graph representation that serves as the input to graph processors based on different backends (`PGFrame`), as well as basic interfaces for different graph analytics and embedding classes (`MetricProcessor`, `PathFinder`, `CommunityDetector`, `GraphElementEmbedder`, etc).
- `bluegraph.backends` is a package that collects implementation of various graph processing and analytics interfaces for different graph backends (for example, `NXPathFinder` for path search capabilities provided by NetworkX, `Neo4jCommunityDetector` for community detection methods provided by Neo4j, etc).
- `bluegraph.preprocess` is a package that contains utils for preprocessing property graphs (e.g. `SemanticPGEncoder` for encoding node/edge properties as numerical vectors, `CooccurrenceGenerator` for generation and analysis of co-occurrence relations in PGFrames.)
- `bluegraph.downstream` is a package that provides a set of utils for various downstream tasks based on vector representations of graphs and graph elements (for example, `NodeSimilarityProcessor` for building and querying node similarity indices based on vector representation of nodes, `EdgePredictor` for predicting true and false edges of the graph based on vector representation of its nodes, `EmbeddingPipeline` for stacking pipelines of graph preprocessing, embedding, similarity index building, etc).

Main components of BlueGraph's API are illustrated in the following diagram:

.. image:: examples/figures/README_BlueGraph_components.png
  :width: 300
  :alt: BlueGraph components


:code:`cord19kg` package
----------------------------

The `cord19kg` package contains a set of tools for interactive exploration and analysis of the `CORD-19 <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_ dataset using the co-occurrence analysis of the extracted named entities. It includes data preparation and curation helpers, tools for generation and analysis of co-occurrence graphs. Moreover, it provides several interactive mini-applications (based on `JupyterDash <https://github.com/plotly/jupyter-dash>`_ and `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/>`_) for Jupyter notebooks allowing the user to interactively perform:

- entity curation;
- graph visualization and analysis;
- dataset saving/loading from `Nexus <https://bluebrainnexus.io/>`_.


:code:`services` package
------------------------

Collects services included as a part of BlueGraph. Currently, only a mini-service for retrieving embedding vectors and similarity computation is included as a part of this repository (see embedder service specific `README <https://github.com/BlueBrain/BlueGraph/tree/bluegraph_design/services/embedder>`_).


Installation
------------

It is recommended to use a virtual environment such as `venv <https://docs.python.org/3.6/library/venv.html>`_  or `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

If you want to use `graph-tool` as a backend, you need to manually install the library, as it is not an oridinary Python library, but a wrapper around a C++ library (please, see `graph-tool installation instructions <https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#native-installation>`_).

The same holds for the Neo4j backend: in order to use it, the database should be installed and started (please, see `Neo4j installation instructions <https://neo4j.com/docs/operations-manual/current/installation/>`_). Typically, the Neo4j-based interfaces provided by BlueGraph require the database uri (the bolt port), username and password to be provided.


Finally, if you want to use the `bluegraph.downstream.similarity` module for building similarity indices (on embedder nodes, for example), you should install the Facebook Faiss library separately. Similarly to `graph-tool`, it is not a pure Python library and it cannot be simply installed by running `pip install`. Please, see `Faiss installation instructions <https://github.com/facebookresearch/faiss/blob/master/INSTALL.md>`_ (`conda` and `conda-forge` installation available).


Development version supporting all the backends can be installed from the source by cloning the current repository, i.e. running the following commands:
::

    git clone https://github.com/BlueBrain/BlueGraph.git
    cd BlueGraph
    pip install .[all]

You can also install a single backend by running the following commands.

::

    git clone https://github.com/BlueBrain/BlueGraph.git
    cd BlueGraph
    pip install .[<backend>]


Where `<backend>` has one of the following values `networkx`,  `graph-tool`, `neo4j`,  `stellargraph`.



Getting started
---------------
The `examples directory <https://github.com/BlueBrain/BlueGraph/tree/bluegraph_design/examples>`_ contains a set of Jupyter notebooks providing tutorials and usecases for BlueGraph.

To get started with property graph data structure `PGFrame` provided by BlueGraph, get an example of semantic property encoding, see the `PGFrames and sematic encoding tutorial <https://github.com/BlueBrain/BlueGraph/blob/bluegraph_design/examples/PGFrames%20and%20sematic%20encoding%20tutorial.ipynb>`_ notebook.

To get familiar with the ideas behind the co-occurrence analysis and the graph analytics interface provided by BlueGraph we recommend to run the following example notebooks: 

- `Literature exploration (PGFrames + in-memory analytics tutorial) <https://github.com/BlueBrain/BlueGraph/blob/bluegraph_design/examples/Literature%20exploration%20(PGFrames%20%2B%20in-memory%20analytics%20tutorial).ipynb>`_  illustrates how to use BlueGraphs's analytics API for in-memory graph backends based on the `NetworkX` and the `graph-tool` libraries.
- `NASA keywords (PGFrames + Neo4j analytics tutorial) <https://github.com/BlueBrain/BlueGraph/blob/bluegraph_design/examples/NASA%20keywords%20(PGFrames%20%2B%20Neo4j%20analytics%20tutorial).ipynb>`_ illustrates how to use the Neo4j-based analytics API for persistent property graphs.

`Embedding and downstream tasks tutorial <https://github.com/BlueBrain/BlueGraph/blob/bluegraph_design/examples/Embedding%20and%20downstream%20tasks%20tutorial.ipynb>`_ starts from the co-occurrence graph generation example and guides the user through the graph representation learning and all it's downstream tasks including node similarity queries, node classification, edge prediction and embedding pipeline building.

Finally, `Create and push embedding pipeline into Nexus.ipynb <https://github.com/BlueBrain/BlueGraph/blob/bluegraph_design/examples/Create%20and%20push%20embedding%20pipeline%20into%20Nexus.ipynb>`_ illustrates how embedding pipelines can be created and pushed to `Nexus <https://bluebrainnexus.io/>`_ and
`Embedding service API <https://github.com/BlueBrain/BlueGraph/blob/bluegraph_design/services/embedder/examples/notebooks/Embedding%20service%20API.ipynb>`_ shows how embedding service that retrieves the embedding pipelines from Nexus can be used.

Getting started with cord19kg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :code:`cord19kg` packages provides `examples <https://github.com/BlueBrain/BlueBrainGraph/tree/refactoring/cord19kg/examples>`_ of CORD-19-specific co-occurrence analysis. 

We recommend starting from the `Co-occurrence analysis tutorial <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Co-occurrence%20analysis%20tutorial.ipynb>`_ notebook providing a simple starting example.

The `Topic-centered co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Glucose%20is%20a%20risk%20facor%20for%20COVID-19%20(3000%20papers).ipynb>`_ notebook provides a full analysis pipeline on the selection of 3000 articles obtained by searching the CORD-19 dataset using the query *"Glucose is a risk factor for COVID-19"* (the search is performed using `BlueBrainSearch <https://github.com/BlueBrain/Search>`_).

The `Nexus-hosted co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Nexus-hosted%20topic-centered%20analysis%20(3000%20papers).ipynb>`_ notebook provides an example for the previously mentioned 3000-article dataset, where datasets corresponding to different analysis steps can be saved and loaded to and from a `Blue Brain Nexus <https://bluebrainnexus.io/>`_ project.

Finally, the :code:`generate_10000_network.py` `script <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/generate_10000_network.py>`_ allows the user to generate the co-occurrence networks for 10'000 most frequent entities extracted from the entire CORD-19v47 database (based on paper- and paragraph- level entity co-occurrence). To run the script, simply execute :code:`python generate_10000_network.py` from the examples folder.

Note that the generated networks are highly dense (contain a large number of edges, for example, ~44M edges for the paper-based network), and the process of their generation, even if parallelized, is highly costly.

Licensing
---------
- Blue Graph is distributed under the Apache 2 license.
- Included example scripts and notebooks (`BlueGraph/examples <https://github.com/BlueBrain/BlueGraph/tree/bluegraph_design/examples>`_ and `BlueGraph/cord19kg/examples <https://github.com/BlueBrain/BlueBrainGraph/tree/master/cord19kg/examples>`_) are distributed under the 3-Clause BSD License.
- Data files provided in the repository are distributed under the X license.

Aknowledgements
---------------

This project has received funding from the EPFL Blue Brain Project (funded by the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology).

.. |Travis_badge| image:: https://travis-ci.com/BlueBrain/BlueBrainGraph.svg?branch=master
    :target: https://travis-ci.com/BlueBrain/BlueBrainGraph

COPYRIGHT 2020–2021, Blue Brain Project/EPFL
