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


def _retrieve_models():
    """Retrieve all models from the catalog."""
    resources = app.forge.search({"type": "EmbeddingModel"})
    # Check if the download folder exists
    if not os.path.exists(app.config["DOWNLOAD_DIR"]):
        os.makedirs(app.config["DOWNLOAD_DIR"])

    for resource in resources:
        app.models[resource.name] = {
            "data": digest_model_data(resource),
        }
        app.forge.download(
            resource, "distribution.contentUrl", app.config["DOWNLOAD_DIR"])

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


app = Flask(__name__)

app.config.from_pyfile('configs/app_config.py')

app.forge = KnowledgeGraphForge(
    app.config["FORGE_CONFIG"],
    token=os.environ["NEXUS_TOKEN"])

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


@app.route("/model/<model_name>", methods=["GET"])  # , "GET", "DELETE"])
def handle_model_request(model_name):
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


@app.route("/model/<model_name>/similar-points/")
def handle_similar_points_request(model_name):
    # token = _retrieve_token(request)
    # forge = KnowledgeGraphForge(
    #     app.config["FORGE_CONFIG"], token=token)

    params = request.args.to_dict(flat=False)
    indices = params["resource_ids"]
    k = int(params["k"][0])
    values = (
        params["values"][0] == "True" if "values" in params else False
    )
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
        similar_points, dist = app.models[
            model_name]["object"].get_similar_points(
                indices, k)
        if values:
            result = {
                indices[i]: {
                    p: float(dist[i][j]) for j, p in enumerate(points)
                }
                for i, points in enumerate(similar_points)
            }
        else:
            result = {
                indices[i]: list(points)
                for i, points in enumerate(similar_points)
            }
        return (
            json.dumps({"similar_points": result}), 200,
            {'ContentType': 'application/json'}
        )
    else:
        return _respond_not_found()


@app.route("/model/<model_name>/details/<component_name>/")
def handle_preprocessor_info_request(model_name, component_name):
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
            info = pipeline.preprocessor.info()
            info["interface"] = pipeline.preprocessor.__class__.__name__
        elif component_name == "embedder":
            info = pipeline.embedder.info()
        elif component_name == "similarity-processor":
            info = pipeline.similarity_processor.info()
            info["interface"] = pipeline.similarity_processor.__class__.__name__
        return (
            json.dumps(info), 200,
            {'ContentType': 'application/json'}
        )
    else:
        return _respond_not_found()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
