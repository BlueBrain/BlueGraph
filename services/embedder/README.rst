BlueBrainEmbedder
-----------------

Mini-service for retrieving embedding vectors and similarity computation. An example can be found in the `Embedding service API <https://github.com/BlueBrain/BlueGraph/blob/master/services/embedder/examples/notebooks/Embedding%20service%20API.ipynb>`_ notebook. 


Running the service
===================

On its starting the service requires Nexus authentication, in order to fetch available models. Therefore, before launching the service, get your token from `Nexus staging <https://staging.nexus.ocp.bbp.epfl.ch/>`_ and export it to an environment variable as follows:

::

  export NEXUS_TOKEN=<your_token>

Run from the source
^^^^^^^^^^^^^^^^^^^

Simply execute

::

	flask run


Run using Docker
^^^^^^^^^^^^^^^^

First, build the docker image:

::

	docker build . -f service/embedder/Dockerfile

Then, run the container by passing your nexus token env variable using the :code:`--env` parameter.

::

	docker run --env NEXUS_TOKEN=$NEXUS_TOKEN -p 5000:5000 <image>
