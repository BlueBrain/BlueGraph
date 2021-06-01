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

# ----------------------------------------------------------------------
# 1. Build an intermediate image which installs dependencies using conda
FROM continuumio/miniconda3 as intermediate

COPY ./services/embedder/environment.yml .
COPY ./requirements.txt .

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "embedder", "/bin/bash", "-c"]
RUN conda-pack -n embedder -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar

RUN /venv/bin/conda-unpack

# ----------------------------------------------------------------------
# 2. Build the final image which installs all the remaining requirements and
# configures / exposes the Flask app

# What is the best base image? 
FROM python:3.7

WORKDIR /app

COPY --from=intermediate /venv /venv
ADD . /app
# RUN chgrp root /app
# RUN chown -R embedder_user:root /app && usermod -d /app embedder_user
RUN chmod -R 777 services/embedder/downloads/

# The EXPOSE instruction indicates the ports on which a container 
# will listen for connections
# Since Flask apps listen to port 5000  by default, we expose it
EXPOSE 5000

SHELL ["/bin/bash", "-c"]
RUN source /venv/bin/activate && \
 	pip install --no-cache-dir .[stellargraph]
ENTRYPOINT source /venv/bin/activate && cd services/embedder && flask run --host=0.0.0.0