====================================================
COVID-19 co-occurrence graph generation and analysis
====================================================

Interactive exploration and analysis of the `CORD-19 <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_ dataset using co-occurrence graphs of named entities. 

About
-----

This repository contains a collection of tools and datasets for the co-occurrence analysis of the `CORD-19 <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_ (v47). The analysis pipeline implemented by the Blue Brain Project consists of the following steps:

- Semantic Literature Search using `BlueBrain/Search <https://github.com/BlueBrain/Search>`_
- Named Entity Recognition using `BlueBrain/Search <https://github.com/BlueBrain/Search>`_
- Entity Linking and Curation (using the `NCIt ontology <https://ncithesaurus.nci.nih.gov/ncitbrowser/>`_)
- Co-occurrence Graph Generation and Analysis

The following usecases are provided in this repository:

**Topic centered co-occurrence analysis** consisting of the following steps (steps 3-5 can be reproduced using the `Glucose is a risk facor for COVID-19 (3000 papers).ipynb <https://github.com/BlueBrain/BlueGraph/blob/master/cord19kg/examples/notebooks/Glucose%20is%20a%20risk%20facor%20for%20COVID-19%20(3000%20papers).ipynb>`_):

1. Semantic Literature Search: the 3000 most relevant papers are selected using the query *'Glucose as a risk factor in COVID-19'*.

        - The meta-data for the selected 3000 papers can be found `here <https://github.com/BlueBrain/BlueGraph/blob/master/cord19kg/examples/data/Glucose_risk_3000_paper_meta_data.csv>`_.
        - The semantic search can be reproduced using `this external notebook <https://github.com/BlueBrain/Search-Graph-Examples>`__.

2. Named Entity Recognition: named entities of interest are extracted. Entity types of interest include "cell compartment", "cell type", "chemical", "symptom / disease", "drug", "organ / system", "organism", "biological process / pathway", and "protein".

        - The dataset with extracted entities for the selected 3000 papers can be found `here <https://github.com/BlueBrain/BlueGraph/blob/master/cord19kg/examples/data/Glucose_risk_3000_papers.csv.zip>`_.
        - The entity extraction can be reproduced using `this external notebook <https://github.com/BlueBrain/Search-Graph-Examples>`_.
       
3. Entity Linking and Curation: extracted entities are linked to the NCIt ontology terms, their types are augmented with the information from the corresponding NCIt entity classes. The resulting entities can be further curated using the provided interactive curation app.

        - The dataset with the ontology linking for the extracted entities can be found `here <https://github.com/BlueBrain/BlueGraph/blob/master/cord19kg/examples/data/NCIT_ontology_linking_3000_papers.csv.zip>`_.
        - The dataset with the mapping of the NCIt entity classes to the entity types of interest `here <https://github.com/BlueBrain/BlueGraph/blob/master/cord19kg/examples/data/NCIT_type_mapping.json>`_.
        
4. Co-occurrence Graph Generation: paper- and paragraph-based co-occurrences of the top 1500 most frequent entities are used to build a knowledge graphs whose nodes represent entities and whose links represent entity co-occurrences. The links are weighted using raw co-occurrence frequencies and mutual-information-based scores: positive pointwise mutual information (PPMI) and normalized point-wise mutual information (NPMI).

5. Co-occurrence Graph Analysis: the generated graphs can be interactively analysed and explored using node centrality measures (PageRank, weighted degree), community partitions, shortest paths search, minimum spanning trees.

..
            - Link to the ontology linking model and data
            - Link to the notebook for generating ontology Linking model and data
            - Add links to MyBinder

**Co-occurrence analysis of the entire CORD-19v47 dataset** consists of the previously described steps 2-5 (i.e. no prior literature search is performed) and can be reproduced using the provided `generate_10000_network.py <https://github.com/BlueBrain/BlueGraph/blob/master/cord19kg/examples/generate_10000_network.py>`_ script. In this usecase, 10'000 most frequent entities are used to build the co-occurrence graphs (note that the generated networks are highly dense and contain a large number of edges, for example, ~44M edges for the paper-based network, and the process of their generation, even if parallelized, is highly costly).

       - The dataset containing extracted and linked entities for the entire CORD-19v47 dataset can be found `here <https://github.com/BlueBrain/BlueGraph/blob/master/cord19kg/examples/data/CORD_19_v47_occurrence_top_10000.json.zip>`__.
       - The script outputs generated co-occurrence graphs based on paper-/paragraph-level co-occurrence and their minimum spanning trees (based on the NPMI distance score) and stores them as JSON. In addition, it computes centrality measures and community partition and stores them as node attributes. 


:code:`cord19kg` package description
-------------------------------------

The package contains a set of tools for interactive exploration and analysis of the `CORD-19 <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_ dataset using the co-occurrence analysis of the extracted named entities. It includes data preparation and curation helpers, tools for generation and analysis of co-occurrence networks. Moreover, it provides several mini-applications (based on `JupyterDash <https://github.com/plotly/jupyter-dash>`_ and `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/>`_) for Jupyter notebooks allowing the user to interactively perform:

- entity curation;
- network visualization and analysis;
- dataset saving/loading from `Nexus <https://bluebrainnexus.io/>`_.


.. _installation:

Installation
------------

To install the :code:`cord19kg` package alongside with all its dependencies, run

.. code-block::

  pip install .[cord19kg]


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

The :code:`cord19kg` packages provides `examples <https://github.com/BlueBrain/BlueBrainGraph/tree/refactoring/cord19kg/examples>`_ of the CORD-19-specific co-occurrence analysis. We recommend starting from the `Co-occurrence analysis tutorial <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Co-occurrence%20analysis%20tutorial.ipynb>`_ notebook providing a simple starting example of a small data sample.

The `Topic-centered co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Glucose%20is%20a%20risk%20facor%20for%20COVID-19%20(3000%20papers).ipynb>`_ notebook provides a full analysis pipeline on the selection of 3000 articles obtained by searching the CORD-19 dataset using the query *"Glucose is a risk factor for COVID-19"* (the search is performed using `BlueSearch <https://github.com/BlueBrain/BlueBrainSearch>`_).

The :code:`generate_10000_networks.py` `script <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/generate_10000_network.py>`_ allows the user to generate the co-occurrence networks for 10'000 most frequent entities extracted from the entire CORD-19v47 database (based on paper- and paragraph- level entity co-occurrence). To run the script, simply execute :code:`python generate_10000_networks.py` from the examples folder. Note that the generated networks are highly dense (contain a large number of edges, for example, ~44M edges for the paper-based network), and the process of their generation, even if parallelized, is highly costly.



