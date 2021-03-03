========================
:code:`cord19kg` package
========================

The package contains a set of tools for interactive exploration and analysis of the `CORD-19 <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_ dataset using the co-occurrence analysis of the extracted named entities. It includes data preparation and curation helpers, tools for generation and analysis of co-occurrence networks. Moreover, it provides several mini-applications (based on `JupyterDash <https://github.com/plotly/jupyter-dash>`_ and `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/>`_) for Jupyter notebooks allowing the user to interactively perform:

- entity curation;
- network visualization and analysis;
- dataset saving/loading from `Nexus <https://bluebrainnexus.io/>`_.


.. _installation:

Installation
------------

To install the :code:`cord19kg` package alongside with all its dependencies, run

.. code-block::

  pip install bluegraph[cord19kg]


Interactive applications
------------------------

Curation app
^^^^^^^^^^^^

The entity curation app allows to view the input data table with entity occurrences, their occurrence frequency, type, etc. It also allows to link the data to the ontology (provided the input linking table), filter entities by their name, frequency and type. The following figure illustrates a snippet of the curation app:

.. image:: ./examples/figures/curation_app.png
  :width: 800
  :alt: Curation app snippet

Graph visualization app
^^^^^^^^^^^^^^^^^^^^^^^

The graph visualization app allows to view the generated co-occurrence graphs as minimal spanning trees, perform visual inspection of its nodes and edges, as well as perform visual analytics:

- filter nodes and edges;
- associate node sizes and edge thinkness values with different node/edge statistics;
- group and filter nodes by different criteria (entity type, communitities detected using co-occurrence frequency and mutual information);
- search for paths between different entities.

The application provides a set of interactive capabilities for examining the data associated to the nodes and edges of the underlying graphs. For example, given a selected node, it allows the user to:

- access the definition of the corresponding entity from the provided ontology linking;
- view the papers that mention the entity;
- inspect raw entities that ontology linking has mapped to a given entity;
- access the set of nearest neighbors with the highest mutual information scores.

It also allows to edit the underlying graph objects or their visualization displayed in the app.

The following figure illustrates a snippet of the curation app:

.. image:: ./examples/figures/graph_vis_app.png
  :width: 800
  :alt: Visualization app snippet


Examples and tutorials
----------------------

To be able to run examples and tutorials, please, install the :code:`cord19kg` package (see the installation instructions above).

The :code:`cord19kg` packages provides `examples <https://github.com/BlueBrain/BlueBrainGraph/tree/refactoring/cord19kg/examples>`_ of CORD-19-specific co-occurrence analysis. We recommend starting from the `Co-occurrence analysis tutorial <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Co-occurrence%20analysis%20tutorial.ipynb>`_ notebook providing a simple starting example (including a small guide to the visualization app interface).

The `Topic-centered co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Glucose%20is%20a%20risk%20facor%20for%20COVID-19%20(3000%20papers).ipynb>`_ notebook provides a full analysis pipeline on the selection of 3000 articles obtained by searching the CORD-19 dataset using the query *"Glucose is a risk factor for COVID-19"* (the search is performed using `BlueBrainSearch <https://github.com/BlueBrain/BlueBrainSearch>`_).

The :code:`generate_10000_networks.py` `script <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/generate_10000_network.py>`_ allows the user to generate the co-occurrence networks for 10'000 most frequent entities extracted from the entire CORD-19v47 database (based on paper- and paragraph- level entity co-occurrence). To run the script, simply execute :code:`python generate_10000_networks.py` from the examples folder.

Finally, the `Nexus-hosted co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Nexus-hosted%20topic-centered%20analysis%20(3000%20papers).ipynb>`_ notebook provides an example for the previously mentioned 3000-article dataset, where datasets corresponding to different analysis steps can be saved and loaded to and from a `Blue Brain Nexus <https://bluebrainnexus.io/>`_ project.

Note that the generated networks are highly dense (contain a large number of edges, for example, ~44M edges for the paper-based network), and the process of their generation, even if parallelized, is highly costly.
