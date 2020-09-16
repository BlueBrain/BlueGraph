"""Utils for generation of co-mention networks."""
import functools
import math
import os

import pickle

import pandas as pd
import multiprocessing as mp
import networkx as nx


# class ComentionScanner(object):
#     """Scanner of co-mentions."""

#     def __init__(self, data, factor_column, factor_count, finish_event):
#         self.name = mp.current_process().name
#         self.data = data
#         self.factor_column = self.factor_column
#         self.factor_count = self.factor_count
#         self.finish_event = finish_event

#         self.scanning_loop()

#     def scanning_loop(self):
#         """Main scanning loop of the scanner."""
#         print(f"[{self.name}] Doing the scanning")

#         while not self.finish_event.is_set():
#             print(f"[{self.name}] Still scanning")

#             # TODO: do some actual work

#         print(f"[{self.name}] Finished work")


# def parallel_network_generation(occurrence_data,
#                                 factor_column=None,
#                                 factor_count=None,
#                                 n_most_frequent=None,
#                                 limit=None,
#                                 parallelize=False, dump_edges=False,
#                                 dump_path=None,
#                                 keep=None):
#     # Configuration
#     n_workers = 4
#     processes = []
#     name = mp.current_process().name

#     task_queue = mp.Queue()

#     finish_event = mp.Event()

#     print(f"[{name}] Creating the worker processes")
#     for i in range(n_workers):
#         process = mp.Process(
#             name=f"Worker-{i}",
#             target=ComentionScanner,
#             args=(
#                 occurrence_data,
#                 factor_column,
#                 factor_count,
#                 finish_event,),
#         )
#         process.start()
#         processes.append(process)

#     # TODO: feed the text mining tasks to worker processes
#     print(f"[{name}] Feeding the tasks to the workers")

#     for process in processes:
#         assert process.is_alive()

#     # Wait for the processes to finish
#     print(f"[{name}] Waiting for the workers to finish")
#     for process in processes:
#         process.join()

#     print(f"[{name}] Finished mining")


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
    """Compute mutual information on a pair of terms."""
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


def generate_comention_network(occurrence_data, factor_count,
                               factor_column=None,
                               n_most_frequent=None,
                               limit=None,
                               parallelize=False,
                               dump_path=None,
                               keep=None):
    """Generate a term co-occurrence network."""
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

    def scan_targets(source_index, generated_edges,
                     print_progress=False, processed_pairs=0):

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
                generated_edges += 1
            if limit and generated_edges == limit:
                print("Reached the edge limit ({})".format(limit))
                return edge_list, generated_edges, processed_pairs
            processed_pairs += 1
            if print_progress:
                percent = int((processed_pairs / total_pairs) * 100)
                print(
                    f"Processed {processed_pairs} ({percent}%) pairs     \x1b[1K\r",
                    end="")

        return edge_list, generated_edges, processed_pairs

    if parallelize:
        # pool = Pool()  # Create a multiprocessing Pool
        # edges = pool.map(
        #     functools.partial(
        #         scan_targets,
        #         indices_to_nodes,
        #         factor_column, limit),
        #     range(len(nodes)))

        # all_edges = sum(edges, [])
        raise NotImplementedError()
    else:
        processed_pairs = 0
        generated_edges = 0
        for index in range(len(nodes)):
            edges, generated_edges, processed_pairs = scan_targets(
                index, generated_edges, print_progress=True,
                processed_pairs=processed_pairs)
            all_edges += edges
            if generated_edges == limit:
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
