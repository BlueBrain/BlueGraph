import numpy as np

from scipy.spatial import cKDTree


def get_similar_nodes(embeddings, node_id, number=10, node_subset=None):
    """Get N most similar entities."""
    embeddings = embeddings
    if node_subset is not None:
        # filter embeddings
        embeddings = embeddings.loc[node_subset]
    if embeddings.shape[0] < number:
        number = embeddings.shape[0]
    search_vec = embeddings.loc[node_id]["embedding"]
    matrix = np.matrix(embeddings["embedding"].to_list())
    closest_indices = cKDTree(matrix).query(search_vec, k=number)[1]
    return embeddings.index[closest_indices].to_list()
