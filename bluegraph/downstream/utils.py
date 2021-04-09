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
"""Collection of utils for benchmarking embedding models."""
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (recall_score,
                             precision_score, roc_auc_score,
                             f1_score, confusion_matrix)


def get_confusion_matrix(true_labels, predicted_label):
    return confusion_matrix(
        true_labels, predicted_label, normalize='true')


def get_classification_scores(true_labels, predicted_labels, average="micro",
                              multiclass=False):
    scores = {
        "accuracy": sum(
            true_labels == predicted_labels) / len(predicted_labels),
        "precision": precision_score(
            true_labels, predicted_labels, average=average),
        "recall": recall_score(
            true_labels, predicted_labels, average=average),
        "f1_score": f1_score(
            true_labels, predicted_labels, average=average)
    }
    if multiclass:
        binarizer = MultiLabelBinarizer()
        true_labels = binarizer.fit_transform(true_labels)
        predicted_labels = binarizer.transform(predicted_labels)
    scores["roc_auc_score"] = roc_auc_score(
        true_labels, predicted_labels, average=average,
        multi_class="ovr")

    return scores


def transform_to_2d(node_embeddings):
    """Transform embeddings to the 2D space using TSNE."""
    transformer = TSNE(n_components=2)
    node_embeddings_2d = transformer.fit_transform(node_embeddings)
    return node_embeddings_2d


def cluster_nodes(node_embeddings, k=4):
    """Cluser nodes in the embedding space."""
    kmeans = KMeans(n_clusters=k, random_state=0).fit(node_embeddings)
    return kmeans.labels_


def plot_2d(pgframe, vector_prop=None, vectors=None, label_prop=None, labels=None,
            title=None, silent=False):
    """Plot a 2D representation of nodes."""
    if vectors is None:
        if vector_prop is None:
            raise ValueError(
                "Vectors to plot are not specified: neither 'vector_prop' "
                "nor 'vectors' is provided")
        vectors = pgframe.get_node_property_values(vector_prop).to_list()

    unlabeled = False
    if labels is None:
        if label_prop is not None:
            labels = pgframe.get_node_property_values(label_prop).to_list()
        else:
            labels = [0] * pgframe.number_of_nodes()
            unlabeled = True

    # Generate color map
    unique_labels = set(labels)
    cm = plt.get_cmap('gist_rainbow')
    generated_colors = np.array([
        cm(1. * i / len(unique_labels))
        for i in range(len(unique_labels))
    ])
    np.random.shuffle(generated_colors)

    alpha = 1
    fig, ax = plt.subplots(figsize=(7, 7))

    # create a scatter per node label
    for i, l in enumerate(unique_labels):
        indices = np.where(np.array(labels) == l)
        ax.scatter(
            vectors[indices, 0],
            vectors[indices, 1],
            c=[generated_colors[i]] * indices[0].shape[0],
            cmap="jet",
            s=50,
            alpha=alpha,
            label=l if not unlabeled else None
        )
    if not unlabeled:
        ax.legend()

    title = (
        title
        if title is not None
        else "2D visualization of the input node representation"
    )
    ax.set_title(title)
    if not silent:
        plt.show()
