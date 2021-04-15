.. _intro_pgframe_tutorial:


Intro to PGFrames and semantic encoding
=============================================

This tutorial will help you to get started with property graph data structure `PGFrame` provided by BlueGraph, get an example of semantic property encoding. The source notebook can be found `here <https://github.com/BlueBrain/BlueGraph/blob/master/examples/notebooks/PGFrames%20and%20sematic%20encoding%20tutorial.ipynb>`_.

.. code:: ipython3

    import random
    
    import numpy as np
    import pandas as pd
    
    from nltk.corpus import words

.. code:: ipython3

    from bluegraph import PandasPGFrame
    from bluegraph.preprocess import ScikitLearnPGEncoder
    from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder

Example 1: small property graph
-------------------------------

Intialize a ``PandasPGFrame`` given a node and edge list.

.. code:: ipython3

    nodes = ["Alice", "Bob", "Eric", "John", "Anna", "Laura", "Matt"]
    
    sources = [
        "Alice", "Alice", "Bob", "Bob", "Bob", "Eric", "Anna", "Anna", "Matt"
    ]
    targets = [
        "Bob", "Eric", "Eric", "John", "Anna", "Anna", "Laura", "John", "John"
    ]
    edges = list(zip(sources, targets))
    
    frame = PandasPGFrame(nodes=nodes, edges=edges)

Get nodes and edges as lists.

.. code:: ipython3

    frame.nodes()




.. parsed-literal::

    ['Alice', 'Bob', 'Eric', 'John', 'Anna', 'Laura', 'Matt']



.. code:: ipython3

    frame.edges()




.. parsed-literal::

    [('Alice', 'Bob'),
     ('Alice', 'Eric'),
     ('Bob', 'Eric'),
     ('Bob', 'John'),
     ('Bob', 'Anna'),
     ('Eric', 'Anna'),
     ('Anna', 'Laura'),
     ('Anna', 'John'),
     ('Matt', 'John')]



Add properties to nodes and edges. Here, all the properties have type
``numeric``. Other available types are: ``categorical`` and ``text``.

.. code:: ipython3

    age = [25, 9, 70, 42, 26, 35, 36]
    frame.add_node_properties(
        {
            "@id": nodes,
            "age": age
        }, prop_type="numeric")
    
    height = [180, 122, 173, 194, 172, 156, 177]
    frame.add_node_properties(
        {
            "@id": nodes,
            "height": height
        }, prop_type="numeric")
    
    weight = [75, 43, 68, 82, 70, 59, 81]
    frame.add_node_properties(
        {
            "@id": nodes,
            "weight": weight
        }, prop_type="numeric")
    
    
    weights = [1.0, 2.2, 0.3, 4.1, 1.5, 21.0, 1.0, 2.5, 7.5]
    edge_weight = pd.DataFrame({
        "@source_id": sources,
        "@target_id": targets,
        "distance": weights
    })
    frame.add_edge_properties(edge_weight, prop_type="numeric")

Get nodes and edges as dataframes.

.. code:: ipython3

    frame.nodes(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>age</th>
          <th>height</th>
          <th>weight</th>
        </tr>
        <tr>
          <th>@id</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Bob</th>
          <td>9</td>
          <td>122</td>
          <td>43</td>
        </tr>
        <tr>
          <th>Eric</th>
          <td>70</td>
          <td>173</td>
          <td>68</td>
        </tr>
        <tr>
          <th>Anna</th>
          <td>26</td>
          <td>172</td>
          <td>70</td>
        </tr>
        <tr>
          <th>Matt</th>
          <td>36</td>
          <td>177</td>
          <td>81</td>
        </tr>
        <tr>
          <th>Alice</th>
          <td>25</td>
          <td>180</td>
          <td>75</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    frame.edges(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>distance</th>
        </tr>
        <tr>
          <th>@source_id</th>
          <th>@target_id</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Bob</th>
          <th>John</th>
          <td>4.1</td>
        </tr>
        <tr>
          <th>Anna</th>
          <th>John</th>
          <td>2.5</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">Bob</th>
          <th>Anna</th>
          <td>1.5</td>
        </tr>
        <tr>
          <th>Eric</th>
          <td>0.3</td>
        </tr>
        <tr>
          <th>Alice</th>
          <th>Bob</th>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Example 2: Random graph with a given density
--------------------------------------------

In this example we will generate a small random graph given a specified
density value (i.e. ratio of edges realized of all possible edges
between distinct pairs of nodes).

Create a PandasPGFrame
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    N = 70  # number of nodes
    density = 0.1  # density value

.. code:: ipython3

    # Helper functions for graph generation
    
    def generate_targets(nodes, s, density=0.2):
        edges = []
        for t in nodes:
            if s < t:
                edge = np.random.choice([0, 1], p=[1 - density, density])
                if edge:
                    
                    edges.append([s, t])
        return edges
    
    
    def random_pgframe(n_nodes, density):
        nodes = list(range(n_nodes))
    
        edges = sum(
            map(lambda x: generate_targets(nodes, x, density), nodes), [])
        edges = pd.DataFrame(
            edges, columns=["@source_id", "@target_id"])
        edges_df = edges.set_index(["@source_id", "@target_id"])
        frame = PandasPGFrame(nodes=nodes, edges=edges_df.index)
        return frame

.. code:: ipython3

    graph_frame = random_pgframe(N, density)

Get nodes and edges as dataframes.

.. code:: ipython3

    graph_frame.nodes(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
        </tr>
        <tr>
          <th>@id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>15</th>
        </tr>
        <tr>
          <th>27</th>
        </tr>
        <tr>
          <th>36</th>
        </tr>
        <tr>
          <th>68</th>
        </tr>
        <tr>
          <th>11</th>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    graph_frame.edges(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
        </tr>
        <tr>
          <th>@source_id</th>
          <th>@target_id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="2" valign="top">16</th>
          <th>63</th>
        </tr>
        <tr>
          <th>58</th>
        </tr>
        <tr>
          <th>25</th>
          <th>52</th>
        </tr>
        <tr>
          <th>23</th>
          <th>59</th>
        </tr>
        <tr>
          <th>25</th>
          <th>43</th>
        </tr>
      </tbody>
    </table>
    </div>



Add node and edge types
~~~~~~~~~~~~~~~~~~~~~~~

Here we generate random types for nodes and edges.

.. code:: ipython3

    types = ["Apple", "Orange", "Carrot"]
    node_types = {
        n: np.random.choice(types, p=[0.5, 0.4, 0.1])
        for n in range(N)
    }

.. code:: ipython3

    graph_frame.add_node_types(node_types)

.. code:: ipython3

    graph_frame.nodes(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>@type</th>
        </tr>
        <tr>
          <th>@id</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>14</th>
          <td>Apple</td>
        </tr>
        <tr>
          <th>64</th>
          <td>Apple</td>
        </tr>
        <tr>
          <th>18</th>
          <td>Carrot</td>
        </tr>
        <tr>
          <th>50</th>
          <td>Carrot</td>
        </tr>
        <tr>
          <th>20</th>
          <td>Orange</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    types = ["isFriend", "isEnemy"]
    edge_types = {
        e: np.random.choice(types, p=[0.8, 0.2])
        for e in graph_frame.edges()
    }

.. code:: ipython3

    graph_frame.add_edge_types(edge_types)

.. code:: ipython3

    graph_frame.edges(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>@type</th>
        </tr>
        <tr>
          <th>@source_id</th>
          <th>@target_id</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>67</th>
          <th>68</th>
          <td>isFriend</td>
        </tr>
        <tr>
          <th>41</th>
          <th>66</th>
          <td>isEnemy</td>
        </tr>
        <tr>
          <th>16</th>
          <th>30</th>
          <td>isFriend</td>
        </tr>
        <tr>
          <th>17</th>
          <th>37</th>
          <td>isFriend</td>
        </tr>
        <tr>
          <th>21</th>
          <th>31</th>
          <td>isFriend</td>
        </tr>
      </tbody>
    </table>
    </div>



Add node and edge properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We add node properties of different data types (``numeric``,
``categorical``, ``text``) randomly.

.. code:: ipython3

    weight = pd.DataFrame(
        [
            (n, np.random.normal(loc=35, scale=5))
            for n in graph_frame.nodes()
        ], 
        columns=["@id", "weight"]
    )

.. code:: ipython3

    graph_frame.add_node_properties(weight, prop_type="numeric")

.. code:: ipython3

    colors = ["red", "green", "blue"]

.. code:: ipython3

    colors = pd.DataFrame(
        [
            (n, np.random.choice(colors))
            for n in graph_frame.nodes()
        ], 
        columns=["@id", "color"]
    )

.. code:: ipython3

    graph_frame.add_node_properties(colors, prop_type="category")

.. code:: ipython3

    desc = pd.DataFrame(
        [
            (n, ' '.join(random.sample(words.words(), 20)))
            for n in graph_frame.nodes()
        ], 
        columns=["@id", "desc"]
    )

.. code:: ipython3

    graph_frame.add_node_properties(desc, prop_type="text")

.. code:: ipython3

    graph_frame.nodes(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>@type</th>
          <th>weight</th>
          <th>color</th>
          <th>desc</th>
        </tr>
        <tr>
          <th>@id</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>13</th>
          <td>Orange</td>
          <td>40.386831</td>
          <td>blue</td>
          <td>cutterhead amanuenses Kashubian Alchornea skin...</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Carrot</td>
          <td>29.168627</td>
          <td>blue</td>
          <td>probe menorrhoeic hemicephalous comart gander ...</td>
        </tr>
        <tr>
          <th>29</th>
          <td>Apple</td>
          <td>35.391697</td>
          <td>blue</td>
          <td>teruncius tetanoid unsovereign carpocarpal unr...</td>
        </tr>
        <tr>
          <th>10</th>
          <td>Apple</td>
          <td>37.038171</td>
          <td>green</td>
          <td>balloter preceding scabies lengthways lotase o...</td>
        </tr>
        <tr>
          <th>18</th>
          <td>Carrot</td>
          <td>32.094158</td>
          <td>green</td>
          <td>oiled sphericle relationism neostriatum molehi...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    graph_frame._node_prop_types




.. parsed-literal::

    {'@type': 'category', 'weight': 'numeric', 'color': 'category', 'desc': 'text'}



We add edge properties of different data types (``numeric``,
``categorical``, ``text``) randomly.

.. code:: ipython3

    years = pd.DataFrame(
        [
            (s, t, np.random.randint(0, 20))
            for s, t in graph_frame.edges()
        ], 
        columns=["@source_id", "@target_id", "n_years"]
    )

.. code:: ipython3

    graph_frame.add_edge_properties(years, prop_type="numeric")

.. code:: ipython3

    shapes = ["dashed", "dotted", "solid"]
    shapes = pd.DataFrame(
        [
            (s, t, np.random.choice(shapes))
            for s, t, in graph_frame.edges()
        ], 
        columns=["@source_id", "@target_id", "shapes"]
    )

.. code:: ipython3

    graph_frame.add_edge_properties(shapes, prop_type="category")

.. code:: ipython3

    desc = pd.DataFrame(
        [
            (s, t, ' '.join(random.sample(words.words(), 20)))
            for s, t, in graph_frame.edges()
        ], 
        columns=["@source_id", "@target_id", "desc"]
    )

.. code:: ipython3

    graph_frame.add_edge_properties(desc, prop_type="text")

.. code:: ipython3

    graph_frame.edges(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>@type</th>
          <th>n_years</th>
          <th>shapes</th>
          <th>desc</th>
        </tr>
        <tr>
          <th>@source_id</th>
          <th>@target_id</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>8</th>
          <th>13</th>
          <td>isFriend</td>
          <td>14</td>
          <td>dotted</td>
          <td>preconize Berycidae shopmaid tanyard topi piac...</td>
        </tr>
        <tr>
          <th>18</th>
          <th>34</th>
          <td>isFriend</td>
          <td>4</td>
          <td>dashed</td>
          <td>Sterope undermusic lorn sorbefacient Sabbatize...</td>
        </tr>
        <tr>
          <th>21</th>
          <th>30</th>
          <td>isFriend</td>
          <td>12</td>
          <td>dashed</td>
          <td>octadic teleozoic elderberry confirm stigmario...</td>
        </tr>
        <tr>
          <th>1</th>
          <th>69</th>
          <td>isFriend</td>
          <td>10</td>
          <td>solid</td>
          <td>leptocephalia Anglist uncorresponding parafloc...</td>
        </tr>
        <tr>
          <th>25</th>
          <th>66</th>
          <td>isEnemy</td>
          <td>12</td>
          <td>dashed</td>
          <td>Iswara myodynamia barken black timoneer defloc...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    graph_frame._edge_prop_types




.. parsed-literal::

    {'@type': 'category',
     'n_years': 'numeric',
     'shapes': 'category',
     'desc': 'text'}



Perform semantic encoding of properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BlueGraph allows to convert node/edge properties of different data types
into numerical vectors.

**NB:** If nltk error occurs, run the following code (the ‘words’ corpus
needs to be downloaded for semantic encoding of text properties):

::

   import nltk
   nltk.download('words')

Create a encoder object for homogeneous encoding (properties of all the
nodes (edges) are encoded with feature vectors of the same length
independently of their type).

.. code:: ipython3

    hom_encoder = ScikitLearnPGEncoder(
        node_properties=["weight", "color", "desc"],
        edge_properties=["n_years", "shapes", "desc"],
        edge_features=True,
        heterogeneous=False,
        encode_types=True,
        drop_types=True,
        text_encoding="tfidf",
        standardize_numeric=True)

.. code:: ipython3

    transformed_frame = hom_encoder.fit_transform(graph_frame)

.. code:: ipython3

    transformed_frame.nodes(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>features</th>
        </tr>
        <tr>
          <th>@id</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>25</th>
          <td>[-0.9693465349258025, 0.0, 1.0, 0.0, 0.0, 0.0,...</td>
        </tr>
        <tr>
          <th>59</th>
          <td>[1.0407324935966866, 0.0, 1.0, 0.0, 0.0, 0.0, ...</td>
        </tr>
        <tr>
          <th>40</th>
          <td>[0.22089544697164212, 0.0, 1.0, 0.0, 0.0, 0.0,...</td>
        </tr>
        <tr>
          <th>12</th>
          <td>[1.521323313308059, 0.0, 0.0, 1.0, 0.0, 0.0, 0...</td>
        </tr>
        <tr>
          <th>62</th>
          <td>[-1.2547871487822837, 0.0, 0.0, 1.0, 0.0, 0.0,...</td>
        </tr>
      </tbody>
    </table>
    </div>



We can inspect encoding models for different node and edge properties
created by BlueGraph.

.. code:: ipython3

    hom_encoder._node_encoders




.. parsed-literal::

    {'weight': StandardScaler(),
     'color': MultiLabelBinarizer(),
     'desc': TfidfVectorizer(max_features=128, stop_words='english', sublinear_tf=True)}



.. code:: ipython3

    transformed_frame.edges(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>features</th>
        </tr>
        <tr>
          <th>@source_id</th>
          <th>@target_id</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>54</th>
          <th>57</th>
          <td>[-0.2198738883485877, 1.0, 0.0, 0.0, 0.0, 0.0,...</td>
        </tr>
        <tr>
          <th>29</th>
          <th>40</th>
          <td>[-0.7501579720128285, 0.0, 1.0, 0.0, 0.0, 0.0,...</td>
        </tr>
        <tr>
          <th>2</th>
          <th>14</th>
          <td>[0.48717155653706673, 0.0, 0.0, 1.0, 0.0, 0.0,...</td>
        </tr>
        <tr>
          <th>15</th>
          <th>49</th>
          <td>[-1.6339647781198965, 0.0, 0.0, 1.0, 0.0, 0.0,...</td>
        </tr>
        <tr>
          <th>18</th>
          <th>33</th>
          <td>[0.3104101953156531, 0.0, 0.0, 1.0, 0.0, 0.0, ...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    hom_encoder._edge_encoders




.. parsed-literal::

    {'n_years': StandardScaler(),
     'shapes': MultiLabelBinarizer(),
     'desc': TfidfVectorizer(max_features=128, stop_words='english', sublinear_tf=True)}



Convert PGFrames to JSON
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    json_repr = graph_frame.to_json()

.. code:: ipython3

    json_repr["nodes"][:2]




.. parsed-literal::

    [{'@id': 0,
      '@type': 'Apple',
      'weight': 36.53863443435658,
      'color': 'green',
      'desc': 'Trinitarian undyeable fearedness quinquelobated thermanalgesia unanimous branchful Septentrion deerherd mispleading timbern mechanal papaphobist rowanberry admeasurement disilicide yade undertake innoxiously epiphanous'},
     {'@id': 1,
      '@type': 'Orange',
      'weight': 37.24906812781439,
      'color': 'blue',
      'desc': 'orderer interpellator acouometer though unpoisonable delegation Yellowknife professorial forenotice computational subinternal weepable cliental microtelephone chandleress feroher falltime consociation theoleptic eustomatous'}]



.. code:: ipython3

    json_repr["edges"][:2]




.. parsed-literal::

    [{'@source_id': 0,
      '@target_id': 25,
      '@type': 'isFriend',
      'n_years': 0,
      'shapes': 'dotted',
      'desc': 'nonsetter noncontent xenelasia ozokerite speiss smithing unillumination stenographer unappeasedly bookling buttgenbachite saxhorn tideless pterygote pix topply spraint wherethrough largen seminebulous'},
     {'@source_id': 0,
      '@target_id': 33,
      '@type': 'isFriend',
      'n_years': 15,
      'shapes': 'dashed',
      'desc': 'traily scagliolist maintenance semipectoral cycloolefin pyovesiculosis reptatorial upsilon rotatodentate determiner marbler benzonitrol sandust cystolithectomy volatilization spiritistic micropterygid unegoistical Rosicrucianism meteorography'}]



Create a new ``PandasPGFrame`` from the generated representation.

.. code:: ipython3

    new_frame = PandasPGFrame.from_json(json_repr)

.. code:: ipython3

    new_frame.nodes(raw_frame=True).sample(5)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>@type</th>
          <th>weight</th>
          <th>color</th>
          <th>desc</th>
        </tr>
        <tr>
          <th>@id</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>40</th>
          <td>Orange</td>
          <td>36.165271</td>
          <td>green</td>
          <td>Mareotic dracontian tartrazine cholelithotomy ...</td>
        </tr>
        <tr>
          <th>38</th>
          <td>Apple</td>
          <td>40.665344</td>
          <td>red</td>
          <td>ballet ensuer congressionalist unicellular Het...</td>
        </tr>
        <tr>
          <th>28</th>
          <td>Carrot</td>
          <td>35.038295</td>
          <td>green</td>
          <td>salicorn outgrowing compensatory vorticism bah...</td>
        </tr>
        <tr>
          <th>13</th>
          <td>Orange</td>
          <td>40.386831</td>
          <td>blue</td>
          <td>cutterhead amanuenses Kashubian Alchornea skin...</td>
        </tr>
        <tr>
          <th>55</th>
          <td>Orange</td>
          <td>34.850857</td>
          <td>green</td>
          <td>overdrowsed uncommuted recital joyful oxidizab...</td>
        </tr>
      </tbody>
    </table>
    </div>


