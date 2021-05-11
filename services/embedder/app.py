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

# import jwt

from flask import Flask, request

from kgforge.core import KnowledgeGraphForge
# from kgforge.specializations.resources import Dataset

from bluegraph.downstream import EmbeddingPipeline
from bluegraph.core import GraphElementEmbedder


# def _retrieve_token(request):
#     """Retrieve token from the request header."""
#     auth_string = request.headers.get('Authorization')
#     try:
#         match = re.match("Bearer (.+)", auth_string)
#     except TypeError:
#         raise ValueError("Invalid token or empty")
#     if not match:
#         return (
#             json.dumps({'success': False}),
#             403,
#             {'ContentType': 'application/json'}
#         )

#     return match.groups()[0]


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
            except:
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


try:
    TOKEN = os.environ["NEXUS_TOKEN"]
    app.forge = KnowledgeGraphForge(
        app.config["FORGE_CONFIG"],
        token=TOKEN)
except KeyError:
    app.forge = None

app.models = {}
_retrieve_models()


# def retrieve_model_resource(forge, model_name, download=False):
#     """Retrieve model resource by its name."""
#     query = f"""
#         SELECT ?id
#         WHERE {{
#             ?id a EmbeddingModel;
#                 name "{model_name}";
#                 <https://bluebrain.github.io/nexus/vocabulary/deprecated> false.
#         }}
#     """
#     resources = forge.sparql(query, limit=1)
#     if resources and len(resources) > 0:
#         resource = forge.retrieve(resources[0].id)
#         if download:
#             forge.download(resource, "contentUrl", app.config["DOWNLOAD_DIR"])


# def retrieve_all_model_resources(forge):
#     """Retrieve all models from the catalog."""
#     query = """
#         SELECT ?id
#         WHERE {
#             ?id a EmbeddingModel;
#                 <https://bluebrain.github.io/nexus/vocabulary/deprecated> false.
#         }
#     """
#     resources = forge.sparql(query, limit=1000)
#     return [
#         forge.retrieve(r.id) for r in resources
#     ]


# def get_existing_models(forge):
#     """Get all the existing models."""
#     model_resources = retrieve_all_model_resources(forge)
#     models = [
#         digest_model_data(resource)
#         for resource in model_resources
#     ]
#     return models


# def deprecate_resource(forge, resource):
#     """Deprecate the resource together with its distribution."""
#     base = resource.id.rsplit('/', 1)[0]
#     file_id = resource.distribution.contentUrl.rsplit('/', 1)[1]
#     file = forge.retrieve(f"{base}/{file_id}")

#     forge.deprecate(resource)
#     forge.deprecate(file)


# def clear_catalogue(forge):
#     """Remove all the existing models."""
#     model_resources = retrieve_all_model_resources(forge)
#     for resource in model_resources:
#         deprecate_resource(forge, resource)


# def post_model(forge, agent, name, description, distribution):
#     # Try retreiving model resource
#     model_resource = retrieve_model_resource(forge, name)
#     if model_resource:
#         # Update an existing model
#         if description:
#             model_resource.description = description
#         if distribution:
#             model_resource.distribution = forge.attach(
#                 distribution, content_type="application/octet-stream")
#         forge.update(model_resource)
#     else:
#         # Create a new model resource
#         model_resource = Dataset(
#             forge,
#             name=name,
#             description=description)
#         model_resource.type = ["Dataset", "EmbeddingModel"]
#         # Add distrubution
#         model_resource.add_distribution(
#             distribution, content_type="application/octet-stream")
#         # Add contribution
#         model_resource.add_contribution(agent, versioned=False)
#         model_resource.contribution.hadRole = "Engineer"

#         forge.register(model_resource)
#     return (
#         json.dumps({
#             "success": True
#         }), 200,
#         {'ContentType': 'application/json'}
#     )


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


def _preprocess_data(data, data_type):
    if data_type == "raw":
        # Use passed data as is
        return data
    else:
        # Here possoble data types are
        # - "raw_labeled" of a form ["label", "raw_data"]
        # - resource PG id to download from Nexus
        # - collection of Nexus resources to build a PG from
        # - (then i guess we need a bucket/org/project/token)
        pass


@app.route("/model/<model_name>", methods=["GET"])  # , "GET", "DELETE"])
def handle_model_request(model_name):
    """Handle request of model data."""
    # token = _retrieve_token(request)
    # forge = KnowledgeGraphForge(
    #     app.config["FORGE_CONFIG"], token=token)

    if model_name in app.models:
        return (
            json.dumps(app.models[model_name]["data"]), 200,
            {'ContentType': 'application/json'}
        )
    else:
        return _respond_not_found()
    # elif request.method == "POST":
    #     # Create an agent resource
    #     agent_data = jwt.decode(token, verify=False)
    #     agent = forge.reshape(
    #         forge.from_json(agent_data), keep=[
    #             "name", "email", "sub", "preferred_username"])
    #     agent.id = agent.sub
    #     agent.type = "Person"

    #     model_desc = request.args.get("description")
    #     # Handle uploaded distribution if available
    #     dist = None
    #     # check if the post request has the file part
    #     if 'distribution' in request.files:
    #         file = request.files['distribution']
    #         # If the user does not select a file, the browser submits an
    #         # empty file without a filename.
    #         if file.filename != '' and file:
    #             path_to_file = os.path.join(
    #                 app.config['DOWNLOAD_DIR'], file.filename)
    #             file.save(path_to_file)
    #             dist = path_to_file

    #     return post_model(
    #         forge, agent, model_name, model_desc, dist)

    # elif request.method == "DELETE":
    #     model_resource = retrieve_model_resource(forge, model_name)
    #     if model_resource:
    #         deprecate_resource(forge, model_resource)
    #         return _respond_success()
    #     else:
    #         return _respond_not_found()


@app.route("/models/", methods=["GET"])  # , "DELETE"])
def handle_models_request():
    """Handle request of all existing models."""
    # TODO: add sort and filter by creation/modification date
    # if request.method == "GET":
    return (
        json.dumps({"models": {
            k: d["data"] for k, d in app.models.items()
        }}), 200,
        {'ContentType': 'application/json'}
    )
    # elif request.method == "DELETE":
    #     clear_catalogue(forge)
    #     return _respond_success()


@app.route("/model/<model_name>/embeddings/")
def handle_embeddings_request(model_name):
    """Handle request of embedding vectors for provided resources."""
    # token = _retrieve_token(request)
    # forge = KnowledgeGraphForge(
    #     app.config["FORGE_CONFIG"], token=token)

    params = request.args.to_dict(flat=False)
    indices = params["resource_ids"]
    # model_resource = retrieve_model_resource(
    #     forge, model_name, download=True)
    if model_name in app.models:
        embeddings = app.models[
            model_name]["object"].retrieve_embeddings(indices)
        return (
            json.dumps({"embeddings": dict(zip(indices, embeddings))}), 200,
            {'ContentType': 'application/json'}
        )
    else:
        return _respond_not_found()


@app.route("/model/<model_name>/similar-points/", methods=["GET", "POST"])
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
        json.dumps({"similar_points": result}), 200,
        {'ContentType': 'application/json'}
    )


@app.route("/model/<model_name>/predict-embedding/")
def handle_predict_embedding(model_name):
    """Perform prediction of the embedding."""
    if model_name in app.models:
        pipeline = app.models[model_name]["object"]
        if pipeline.is_inductive():
            params = request.args.to_dict(flat=False)
            data = params["data"]
            data_type = (
                params["data_type"]
                if "data_type" in params else "raw"
            )
            preprocessor_kwargs = (
                params["preprocessor_kwargs"]
                if "preprocessor_kwargs" in params else None
            )
            embedder_kwargs = (
                params["embedder_kwargs"]
                if "embedder_kwargs" in params else None
            )
            _preprocess_data(data, data_type)
            vectors = pipeline.run_prediction(
                data, preprocessor_kwargs, embedder_kwargs)
            return (
                json.dumps({"vectors": vectors.tolist()}), 200,
                {'ContentType': 'application/json'}
            )
        else:
            _respond_not_allowed(
                "Model is transductive, prediction of "
                "embedding for unseen data is not supported")


@app.route("/model/<model_name>/details/<component_name>/")
def handle_info_request(model_name, component_name):
    """Handle request of details on different model components."""
    # token = _retrieve_token(request)
    # forge = KnowledgeGraphForge(
    #     app.config["FORGE_CONFIG"], token=token)

    # model_resource = retrieve_model_resource(
    #     forge, model_name, download=True)
    if model_name in app.models:
        # pipeline_path = os.path.join(
        #     app.config["DOWNLOAD_DIR"],
        #     model_resource.distribution.name)
        # pipeline = EmbeddingPipeline.load(
        #     pipeline_path,
        #     embedder_interface=GraphElementEmbedder,
        #     embedder_ext="zip")
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
