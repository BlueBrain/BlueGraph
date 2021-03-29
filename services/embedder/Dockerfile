# BlueGraph: unifying Python framework for graph analytics and co-occurrence analysis. 

# Copyright 2020-2021 Blue Brain Project / EPFL

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# 1. Build an intermediate image which fetches BlueGraph from its private repo
FROM continuumio/miniconda3 as intermediate

# Here we need to install BlueGraph from the private repo, so we
# need to safely copy our private key to the image

# Copy

COPY ./services/embedder/environment.yml .


# Install the package as normal:
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "embedder", "/bin/bash", "-c"]
RUN conda install -c conda-forge conda-pack && conda install -c pytorch faiss-gpu


# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n embedder -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack

# ----------------------------------------------------------------
# 2. Build the final image which installs all the requirements and
# configures / exposes the Flask app

# What is the best base image? 
FROM python:3.6
# FROM python:3.6.5-alpine

# copy the python packages from the intermediate image
COPY --from=intermediate /venv /venv

# Sets the working directory for following COPY and CMD instructions
# Notice we haven’t created a directory by this name - this instruction 
# creates a directory with this name if it doesn’t exist
WORKDIR /app

COPY ./services/embedder /app
ADD ./bluegraph ./bluegraph
COPY ./setup.py .
COPY ./requirements.txt .
# The EXPOSE instruction indicates the ports on which a container 
# will listen for connections
# Since Flask apps listen to port 5000  by default, we expose it
EXPOSE 5000

SHELL ["/bin/bash", "-c"]
RUN source /venv/bin/activate && pip install -r requirements.txt && pip install --no-cache-dir .[stellargraph]
ENTRYPOINT source /venv/bin/activate && flask run --host=0.0.0.0