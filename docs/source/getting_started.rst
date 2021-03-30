Getting Started
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