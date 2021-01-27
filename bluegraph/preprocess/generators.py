import math
import pandas as pd

import multiprocessing as mp
import queue


SENTINEL = "END"


def aggregate_index(x):
    return set(x.index)


def mutual_information(co_freq, s_freq, t_freq, total_instances, mitype=None):
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
                raise ValueError(
                    "Provided Mutual information score type (mitype) is not "
                    "supported. Provide one value from the following list "
                    "['expected', 'normalized','pmi2', 'pmi3'] ")
        else:
            mi = math.log2((total_instances * co_freq) / (s_freq * t_freq))
    else:
        mi = 0
    return mi if mi > 0 else 0


def compute_frequency(pfgrame, s, t, common_factors,
                      total_factor_instances,
                      factor_aggregator, reverse_edges):
    return len(common_factors)


def _get_s_t_frequencies(pgframe, s, t, factor_aggregator,
                         reverse_edges):
    if not reverse_edges:
        s_targets = pgframe._edges.xs(s, level=0, axis=0)
        t_targets = pgframe._edges.xs(t, level=0, axis=0)
    else:
        s_targets = pgframe._edges.xs(s, level=1, axis=0)
        t_targets = pgframe._edges.xs(t, level=1, axis=0)

    s_freq = len(factor_aggregator(s_targets))
    t_freq = len(factor_aggregator(t_targets))
    return s_freq, t_freq


def compute_ppmi(pgframe, s, t, common_factors,
                 total_factor_instances,
                 factor_aggregator, reverse_edges):
    co_freq = len(common_factors)
    s_freq, t_freq = _get_s_t_frequencies(
        pgframe, s, t, factor_aggregator, reverse_edges)

    return mutual_information(
        co_freq, s_freq, t_freq, total_factor_instances)


def compute_npmi(pgframe, s, t, common_factors,
                 total_factor_instances,
                 factor_aggregator, reverse_edges):
    co_freq = len(common_factors)
    s_freq, t_freq = _get_s_t_frequencies(
        pgframe, s, t, factor_aggregator, reverse_edges)
    return mutual_information(
        co_freq, s_freq, t_freq, total_factor_instances,
        mitype="normalized")


COOCCURRENCE_STATISTICS = {
    "frequency": compute_frequency,
    "ppmi": compute_ppmi,
    "npmi": compute_npmi
}


def scan_targets(pfgrame, indices_to_nodes, source_index,
                 factor_aggregator, compute_statistics, total_factor_instances,
                 generated_edges, reverse_edges=False, limit=None):
    """Scan possible co-occurrence targets for the input source term."""
    edge_list = []
    for target_index in range(source_index + 1, len(indices_to_nodes)):
        s = indices_to_nodes[source_index]
        t = indices_to_nodes[target_index]

        if not reverse_edges:
            s_targets = pfgrame._edges.xs(s, level=0, axis=0)
            t_targets = pfgrame._edges.xs(t, level=0, axis=0)
        else:
            s_targets = pfgrame._edges.xs(s, level=1, axis=0)
            t_targets = pfgrame._edges.xs(t, level=1, axis=0)

        if factor_aggregator is None:
            factor_aggregator = aggregate_index

        s_factors = factor_aggregator(s_targets)
        t_factors = factor_aggregator(t_targets)

        common_factors = s_factors.intersection(t_factors)

        if len(common_factors) > 0:
            edge = {
                "@source_id": s,
                "@target_id": t,
                "common_factors": common_factors
            }

            for stat in compute_statistics:
                edge[stat] = COOCCURRENCE_STATISTICS[stat](
                    pfgrame, s, t, common_factors,
                    total_factor_instances,
                    factor_aggregator, reverse_edges)

            edge_list.append(edge)

        if limit:
            if len(generated_edges) + len(edge_list) == limit:
                print("Reached the edge limit ({})".format(limit))
                return edge_list

    return edge_list


def scanning_loop(pgframe, indices_to_nodes, factor_aggregator,
                  compute_statistics, total_factor_instances,
                  all_edges, reverse_edges, task_queue,
                  generated_edges, limit=None):
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
                pgframe, indices_to_nodes, source_index, factor_aggregator,
                compute_statistics, total_factor_instances,
                all_edges, reverse_edges, limit=limit)
            generated_edges += edge_list


def schedule_scanning(task_queue, indices, n_workers):
    """Schedule scanning work."""
    for i in indices:
        task_queue.put(i)
    for _ in range(n_workers):
        task_queue.put(SENTINEL)
    task_queue.close()
    task_queue.join_thread()


def generate_cooccurrence_edges(pgframe, edge_type=None,
                                factor_aggregator=None,
                                reverse_edges=False,
                                compute_statistics=None,
                                parallelize=False,
                                cores=4, limit=None):

    if compute_statistics is None:
        compute_statistics = []
        total_factor_instances = None
    else:
        if factor_aggregator is None:
            if not reverse_edges:
                targets = pgframe._edges[
                            pgframe._edges["@type"] == edge_type
                        ].index.get_level_values(1).unique()
            else:
                targets = pgframe._edges[
                        pgframe._edges["@type"] == edge_type
                    ].index.get_level_values(0).unique()
            total_factor_instances = len(targets)
        else:
            total_factor_instances = len(factor_aggregator(pgframe._edges))

    if not reverse_edges:
        sources = pgframe._edges[
            pgframe._edges[
                "@type"] == edge_type].index.get_level_values(0).unique()
    else:
        sources = pgframe._edges[
            pgframe._edges[
                "@type"] == edge_type].index.get_level_values(1).unique()

    indices_to_nodes = {i: n for i, n in enumerate(sources)}

    all_edges = []

    total_pairs = (len(sources) * (len(sources) - 1)) / 2
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
                    pgframe, indices_to_nodes,
                    factor_aggregator,
                    compute_statistics,
                    total_factor_instances,
                    all_edges,
                    reverse_edges, task_queue,
                    generated_edges, limit),
            )
            process.start()
            processes.append(process)

        # Initialize scheduler process
        tasker_process = mp.Process(
            target=schedule_scanning,
            args=(task_queue, range(len(sources)), cores))
        tasker_process.start()

        tasker_process.join()

        for worker_process in processes:
            worker_process.join()
        all_edges = list(generated_edges)
    else:
        for source_index in range(len(sources)):
            edges = scan_targets(
                pgframe, indices_to_nodes, source_index,
                factor_aggregator,
                compute_statistics,
                total_factor_instances,
                all_edges,
                reverse_edges, limit=limit)
            all_edges += edges
            if len(all_edges) == limit:
                break

    edge_frame = pd.DataFrame(all_edges)
    edge_frame = edge_frame.set_index(
        ["@source_id", "@target_id"])
    return edge_frame
