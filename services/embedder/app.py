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

"""Main embedding service app."""
import json
import os
import shutil
import re
import time

from flask import Flask, request

from kgforge.core import KnowledgeGraphForge

from bluegraph import PandasPGFrame
from bluegraph.downstream import EmbeddingPipeline
from bluegraph.core import GraphElementEmbedder


def _retrieve_token(request):
    """Retrieve NEXUS token from the request header."""
    auth_string = request.headers.get('Authorization')
    try:
        match = re.match("Bearer (.+)", auth_string)
    except TypeError:
        match = None
    if match:
        return match.groups()[0]


def digest_model_data(model_resource):
    """Digest model meta-data."""
    model_data = {
        "id": model_resource.id,
        "name": model_resource.name,
        "description": model_resource.description,
        "filename": model_resource.distribution.name,
        "created": model_resource._store_metadata._createdAt,
        "modified": model_resource._store_metadata._updatedAt
    }
    return model_data


def _retrieve_models(local=True):
    """Retrieve all models from the catalog."""
    # Check if the download folder exists
    def _get_meta_data(model_name, file):
        return {
            "data": {
                "id": model_name,
                "name": model_name,
                "description": model_name,
                "filename": os.path.join(
                    app.config["DOWNLOAD_DIR"], file),
                "created": time.ctime(os.path.getctime(
                    os.path.join(
                        app.config["DOWNLOAD_DIR"],
                        file))),
                "modified": time.ctime(os.path.getmtime(
                    os.path.join(
                        app.config["DOWNLOAD_DIR"],
                        file)))
            }
        }

    if not os.path.exists(app.config["DOWNLOAD_DIR"]):
        os.makedirs(app.config["DOWNLOAD_DIR"])

    if not local:
        # Fetch from a Nexus-hosted catalog
        resources = app.forge.search({"type": "EmbeddingModel"})
        for resource in resources:
            app.models[resource.name] = {
                "data": digest_model_data(resource),
            }
            app.forge.download(
                resource, "distribution.contentUrl",
                app.config["DOWNLOAD_DIR"])

            pipeline_path = os.path.join(
                app.config["DOWNLOAD_DIR"],
                resource.distribution.name)
            app.models[resource.name]["object"] = EmbeddingPipeline.load(
                pipeline_path,
                embedder_interface=GraphElementEmbedder,
                embedder_ext="zip")

        # Clear the downloads dir
        for f in os.listdir(app.config["DOWNLOAD_DIR"]):
            try:
                os.remove(os.path.join(app.config["DOWNLOAD_DIR"], f))
            except Exception:
                shutil.rmtree(os.path.join(app.config["DOWNLOAD_DIR"], f))
    else:
        # Fetch from a local dir
        for (_, dirs, files) in os.walk(app.config["DOWNLOAD_DIR"]):
            for path in dirs + files:
                if path[0] != ".":
                    match = re.match(r"(.*)\.zip", path)
                    if match:
                        model_name = match.groups()[0]
                    else:
                        model_name = path
                    app.models[model_name] = _get_meta_data(model_name, path)
                    pipeline_path = os.path.join(
                        app.config["DOWNLOAD_DIR"], path)
                    app.models[model_name]["object"] = EmbeddingPipeline.load(
                        pipeline_path,
                        embedder_interface=GraphElementEmbedder,
                        embedder_ext="zip")
            break


app = Flask(__name__)

app.config.from_pyfile('configs/app_config.py')


if app.config["LOCAL"] is False:
    TOKEN = os.environ["NEXUS_TOKEN"]
    app.forge = KnowledgeGraphForge(
        app.config["FORGE_CONFIG"],
        token=TOKEN)
else:
    app.forge = None

app.models = {}
_retrieve_models(app.config["LOCAL"])

# --------------- Handlers ----------------


def _respond_success():
    return (
        json.dumps({"success": True}), 200,
        {'ContentType': 'application/json'}
    )


def _respond_not_found(message=None):
    if message is None:
        message = "Model is not found in the catalog"
    return (
        json.dumps({
            'success': False,
            'message': message
        }), 404,
        {'ContentType': 'application/json'}
    )


def _respond_not_allowed(message=None):
    if message is None:
        message = "Request method is not allowed"
    return (
        json.dumps({
            'success': False,
            'message': message
        }), 405,
        {'ContentType': 'application/json'}
    )


def _preprocess_data(data, data_type, auth=None):
    """Preprocess input data according to the specified type.

    Possoble data types are:

    - "raw" use data as is provided in the request
    - "json_pgframe" create a PandasPGFrame from the provided JSON repr
    - "nexus_dataset" download a JSON dataset from Nexus and
      create a PandasPGFrame from this representation
    # - collection of Nexus resources to build a PG from
    # - (then i guess we need a bucket/org/project/token)
    """
    if data_type == "raw":
        # Use passed data as is
        return data
    elif data_type == "json_pgframe":
        return PandasPGFrame.from_json(data)
    elif data_type == "nexus_dataset":
        if auth is None:
            raise ValueError(
                "To use Nexus-hosted property graph as the dataset "
                "authentication token should be provided in the "
                "request header")
        forge = KnowledgeGraphForge(
            app.config["FORGE_CONFIG"], endpoint=data["endpoint"],
            bucket=data["bucket"], token=auth)
        resource = forge.retrieve(data["resource_id"])
        forge.download(
            resource, "distribution.contentUrl",
            app.config["DOWNLOAD_DIR"])
        downloaded_file = os.path.join(
            app.config["DOWNLOAD_DIR"], resource.distribution.name)
        graph = PandasPGFrame.load_json(downloaded_file)
        os.remove(downloaded_file)
        return graph
    else:
        raise ValueError("Unknown data type")


@app.route("/models/<model_name>", methods=["GET"])  # , "GET", "DELETE"])
def handle_model_request(model_name):
    """Handle request of model data."""
    if model_name in app.models:
        return (
            json.dumps(app.models[model_name]["data"]), 200,
            {'ContentType': 'application/json'}
        )
    else:
        return _respond_not_found()


@app.route("/models/", methods=["GET"])  # , "DELETE"])
def handle_models_request():
    """Handle request of all existing models."""
    # TODO: add sort and filter by creation/modification date
    return (
        json.dumps({"models": {
            k: d["data"] for k, d in app.models.items()
        }}), 200,
        {'ContentType': 'application/json'}
    )


@app.route("/models/<model_name>/embedding/", methods=["GET", "POST"])
def handle_embeddings_request(model_name):
    """Handle request of embedding vectors for provided resources."""
    if model_name in app.models:
        pipeline = app.models[model_name]["object"]
        if request.method == "GET":
            params = request.args.to_dict(flat=False)
            indices = params["resource_ids"]
            embeddings = pipeline.retrieve_embeddings(indices)
            return (
                json.dumps({
                    "vectors": dict(zip(indices, embeddings))
                }), 200,
                {'ContentType': 'application/json'}
            )
        else:
            if pipeline.is_inductive():
                auth_token = _retrieve_token(request)
                content = request.get_json()
                data = content["data"]
                data_type = (
                    content["data_type"]
                    if "data_type" in content else "raw"
                )
                preprocessor_kwargs = (
                    content["preprocessor_kwargs"]
                    if "preprocessor_kwargs" in content else None
                )
                embedder_kwargs = (
                    content["embedder_kwargs"]
                    if "embedder_kwargs" in content else None
                )
                data = _preprocess_data(data, data_type, auth_token)
                vectors = pipeline.run_prediction(
                    data, preprocessor_kwargs, embedder_kwargs)

                if not isinstance(vectors, list):
                    vectors = vectors.tolist()

                return (
                    json.dumps({"vectors": vectors}), 200,
                    {'ContentType': 'application/json'}
                )
            else:
                _respond_not_allowed(
                    "Model is transductive, prediction of "
                    "embedding for unseen data is not supported")

    else:
        return _respond_not_found()


@app.route("/models/<model_name>/neighbors/", methods=["GET", "POST"])
def handle_similar_points_request(model_name):
    """Handle request of similar points to provided resources."""
    if model_name not in app.models:
        return _respond_not_found()

    pipeline = app.models[model_name]["object"]
    params = request.args.to_dict(flat=False)
    k = int(params["k"][0])
    values = (
        params["values"][0] == "True" if "values" in params else False
    )

    if request.method == 'GET':
        indices = params["resource_ids"]
        similar_points, dist = pipeline.get_similar_points(
            existing_indices=indices, k=k)
        if values:
            result = {
                indices[i]: {
                    p: float(dist[i][j]) for j, p in enumerate(points)
                } if points is not None else None
                for i, points in enumerate(similar_points)
            }
        else:
            result = {
                indices[i]: list(points) if points is not None else None
                for i, points in enumerate(similar_points)
            }
    else:
        content = request.get_json()
        vectors = content["vectors"]
        similar_points, dist = pipeline.get_similar_points(
            vectors=vectors, k=k)
        if values:
            result = [
                {point: dist[i].tolist()[j] for j, point in enumerate(el)}
                for i, el in enumerate(similar_points)
            ]
        else:
            result = [
                el.tolist() for el in similar_points
            ]
    return (
        json.dumps({"neighbors": result}), 200,
        {'ContentType': 'application/json'}
    )


@app.route("/models/<model_name>/<component_name>/")
def handle_info_request(model_name, component_name):
    """Handle request of details on different model components."""
    if model_name in app.models:
        pipeline = app.models[model_name]["object"]
        info = None
        if component_name == "preprocessor":
            if pipeline.preprocessor is not None:
                info = pipeline.preprocessor.info()
                info["interface"] = pipeline.preprocessor.__class__.__name__
            else:
                return _respond_not_found(
                    "Model does not contain a preprocessor")
        elif component_name == "embedder":
            if pipeline.embedder is not None:
                info = pipeline.embedder.info()
            else:
                return _respond_not_found(
                    "Model does not contain an embedder")
        elif component_name == "similarity-processor":
            info = pipeline.similarity_processor.info()
            info["interface"] = pipeline.similarity_processor.__class__.__name__

        # Convert all the values to str
        for k in info.keys():
            info[k] = str(info[k])

        return (
            json.dumps(info), 200,
            {'ContentType': 'application/json'}
        )
    else:
        return _respond_not_found()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
