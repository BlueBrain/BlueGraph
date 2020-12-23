"""Utils for generation of co-occurrence networks."""
import math

import pickle

import pandas as pd
import multiprocessing as mp
import networkx as nx
import queue

SENTINEL = "END"


def filter_unfrequent(data, factor, n, keep=None):
    """Filter unfrequent entities."""
    largest_indices = data[factor].apply(
        lambda x: len(x)).nlargest(n).index
    filtered_data = data.loc[largest_indices]
    if keep is not None:
        for n in keep:
            index = [i.lower() for i in filtered_data.index]
            if n.lower() not in index and n in data.index:
                filtered_data.loc[n] = data.loc[n]
    return filtered_data


def cofrequence(occurrence_data, factor, s, t):
    """Compute co-mention frequence."""
    intersection = occurrence_data.loc[s][factor].intersection(
        occurrence_data.loc[t][factor])
    return len(intersection)


def mutual_information(occurrence_data, factor, total_instances,
                       s, t, mitype=None):
    """Compute mutual information on a pair of terms.

    occurrence_data : pandas.DataFrame
        Dataframe containing term occurrence data, IDs of terms are given by
        the index column of the data frame.
    factor : str
        Name of the column containing the term occurrence data (for example,
        a set of unique scientific articles per each term in the index).
    total_instances : int
        Total number of all unique instances of the occurrence factor (for
        example, the total number of all scientific articles in the dataset).
    s : str
        Source term
    t : str
        Traget term
    mitype : str, optional
        Mutual information score type. Possible types 'expected', 'normalized',
        'pmi2', 'pmi3', by default, no normalization is applied (i.e. positive
        pointwise mutual information is computed).
    """
    co_freq = cofrequence(occurrence_data, factor, s, t)
    s_freq = len(occurrence_data.loc[s][factor])
    t_freq = len(occurrence_data.loc[t][factor])
    if co_freq > 0:
        if mitype is not None:
            if mitype == "expected":
                mi = math.log2(
                    (total_instances * co_freq) / (s_freq * t_freq)
                ) * (co_freq / total_instances)
            elif mitype == "normalized":
                alpha = - math.log2(co_freq / total_instances)
                mi = math.log2(
                    (total_instances * co_freq) / (s_freq * t_freq)) / alpha
            elif mitype == "pmi2":
                mi = math.log2((co_freq ** 2) / (s_freq * t_freq))
            elif mitype == "pmi3":
                mi = math.log2(
                    (co_freq ** 3) / (s_freq * t_freq * total_instances))
        else:
            mi = math.log2((total_instances * co_freq) / (s_freq * t_freq))
    else:
        mi = 0
    return mi


def scan_targets(occurrence_data, factor_column, factor_count,
                 indices_to_nodes, source_index, generated_edges,
                 limit=None):
    """Scan possible co-occurrence targets for the input source term."""
    edge_list = []
    for target_index in range(source_index + 1, len(indices_to_nodes)):
        s = indices_to_nodes[source_index]
        t = indices_to_nodes[target_index]
        frequency = cofrequence(
            occurrence_data, factor_column, s, t)

        if frequency > 0:
            ppmi = mutual_information(
                occurrence_data, factor_column,
                factor_count, s, t)
            npmi = mutual_information(
                occurrence_data, factor_column,
                factor_count, s, t, mitype="normalized")
            edge_list.append({
                "source": s,
                "target": t,
                "frequency": frequency,
                "ppmi": ppmi if ppmi > 0 else 0,
                "npmi": npmi if npmi > 0 else 0,
            })

        if limit:
            if len(generated_edges) + len(edge_list) == limit:
                print("Reached the edge limit ({})".format(limit))
                return edge_list

    return edge_list


def scanning_loop(data, factor_column, factor_count,
                  indices_to_nodes, task_queue, generated_edges,
                  limit=None):
    """Main scanning loop of the edge scanner."""
    first_scan = True

    while True:
        try:
            source_index = task_queue.get(timeout=0.1)
            if source_index == SENTINEL:
                break
        except queue.Empty:
            pass
        else:
            if first_scan:
                first_scan = False
            edge_list = scan_targets(
                data, factor_column, factor_count,
                indices_to_nodes, source_index, generated_edges,
                limit=limit)
            generated_edges += edge_list


def schedule_scanning(task_queue, indices, n_workers):
    """Schedule scanning work."""
    for i in indices:
        task_queue.put(i)
    for _ in range(n_workers):
        task_queue.put(SENTINEL)
    task_queue.close()
    task_queue.join_thread()


def generate_cooccurrence_network(occurrence_data,
                                  factor_column,
                                  factor_count,
                                  n_most_frequent=None,
                                  limit=None,
                                  parallelize=False,
                                  cores=None,
                                  dump_path=None,
                                  keep=None):
    """Generate a co-occurrence network.

    Every unique pair of terms is examined for co-occurrence: if they co-occur
    at least once, an edge between these terms is added to the co-occurrence
    network. Every added edge is equipped with five weights:

    - raw co-occurrence frequency
    - positive pointwise mutual information (PPMI)
    - normalized pointwise mutual information (NPMI)
    - distance based on PPMI (1 / PPMI)
    - distance based on NPMI (1 / NPMI)

    Parameters
    ----------
    occurrence_data : pd.DataFrame
        Dataframe containing term occurrence data, IDs of terms are given by
        the index column of the data frame.
    factor_column : str
        Name of the column containing the term occurrence data (for example,
        a set of unique scientific articles per each term in the index).
    facour_count : int
        Total number of all unique instances of the occurrence factor (for
        example, the total number of all scientific articles in the dataset).
    n_most_frequent : int, optional
        Number of most frequent entitites to include in the co-occurrence
        network. By default is not set, therefore, all the terms from the
        occurrence table are included.
    limit : int, optional
        Limit on the number of edges to generate, by default no limit is set.
    paralellize : bool, optional
        Flag indicating if the graph generation process should be parallelized,
        by default the generation process is sequential.
    cores : int, optional
        Number of cores to use during the parallel network generation.
    dump_path : str, optional
        Path for dumping generated network (only the edge list and edge
        attributes are saved).
    keep : iterable, optional
        Collection of terms to keep even if they are not included in N most
        frequent terms.
    """
    if n_most_frequent is not None:
        print("Fitering data.....")
        occurrence_data = filter_unfrequent(
            occurrence_data, factor_column, n_most_frequent, keep=keep)
        print("Selected {} most frequent terms".format(occurrence_data.shape[0]))
    else:
        occurrence_data = occurrence_data

    nodes = sorted(occurrence_data.index)
    indices_to_nodes = {i: n for i, n in enumerate(nodes)}

    all_edges = []

    total_pairs = (len(nodes) * (len(nodes) - 1)) / 2
    print("Examining {} pairs of terms for co-occurrence...".format(
        int(total_pairs)))

    if parallelize:
        # Create worker processes
        if cores is None:
            cores = 4
        processes = []

        task_queue = mp.Queue()

        # Shared edge list
        manager = mp.Manager()
        generated_edges = manager.list()

        # Each worker executes a scanning loop until the SENTIEL token
        # is encountered
        for i in range(cores):
            process = mp.Process(
                target=scanning_loop,
                args=(
                    occurrence_data,
                    factor_column,
                    factor_count,
                    indices_to_nodes,
                    task_queue,
                    generated_edges),
            )
            process.start()
            processes.append(process)

        # Initialize scheduler process
        tasker_process = mp.Process(
            target=schedule_scanning,
            args=(task_queue, range(len(nodes)), cores))
        tasker_process.start()

        tasker_process.join()

        for worker_process in processes:
            worker_process.join()
        all_edges = list(generated_edges)
    else:
        for source_index in range(len(nodes)):
            edges = scan_targets(
                occurrence_data, factor_column, factor_count,
                indices_to_nodes, source_index, all_edges,
                limit=limit)
            all_edges += edges
            if len(all_edges) == limit:
                break

    print("Generated {} edges                    ".format(
        len(all_edges)))

    edge_list = pd.DataFrame(all_edges)
    edge_list["distance_ppmi"] = 1 / edge_list["ppmi"]
    edge_list["distance_npmi"] = 1 / edge_list["npmi"]

    print("Created a co-occurrence graph:")
    print("\tnumber of nodes: ", len(nodes))
    print("\tnumber of edges: ", edge_list.shape[0])
    print("Saving the edges...")
    if dump_path:
        with open(dump_path, "wb") as f:
            pickle.dump(edge_list, f)

    print("Creating a graph object...")
    graph = nx.from_pandas_edgelist(
        edge_list, edge_attr=[
            "frequency",
            "ppmi",
            "npmi",
            "distance_ppmi",
            "distance_npmi"
        ])

    return graph
