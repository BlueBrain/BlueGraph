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
import math
import pandas as pd

import multiprocessing as mp
import queue

from bluegraph.core.utils import _aggregate_values, safe_intersection

SENTINEL = "END"


def mutual_information(co_freq, s_freq, t_freq, total_instances, mitype=None):
    """Compute mutual information on a pair of terms.

    Parameters
    ----------
    co_freq : int
        Co-occurrence frequency of s & t
    s_freq : int
        Occurrence frequency of s
    t_freq : int
        Occurrence frequency of t
    total_instances : int
        Total number of all unique instances of the occurrence factor (for
        example, the total number of all scientific articles in the dataset).
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
                mi = (
                    (math.log2(
                        (total_instances * co_freq) / (s_freq * t_freq)) / alpha)
                    if alpha != 0 else 0
                )
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


def _compute_frequency(pgframe, s, t, node_property, common_factors,
                       total_factor_instances,
                       factor_aggregator, reverse_edges):
    return len(common_factors)


def _get_s_t_frequencies(pgframe, s, t, node_property, factor_aggregator,
                         reverse_edges):
    if node_property is not None:
        if factor_aggregator is None:
            s_factors = pgframe._nodes.loc[s][node_property]
            t_factors = pgframe._nodes.loc[t][node_property]
        else:
            s_factors = factor_aggregator(
                pgframe._nodes.loc[s])
            t_factors = factor_aggregator(
                pgframe._nodes.loc[t])
        s_freq = len(s_factors)
        t_freq = len(t_factors)
    else:
        if not reverse_edges:
            s_targets = pgframe._edges.xs(s, level=0, axis=0)
            t_targets = pgframe._edges.xs(t, level=0, axis=0)
        else:
            s_targets = pgframe._edges.xs(s, level=1, axis=0)
            t_targets = pgframe._edges.xs(t, level=1, axis=0)

        s_freq = len(factor_aggregator(s_targets))
        t_freq = len(factor_aggregator(t_targets))
    return s_freq, t_freq


def _compute_ppmi(pgframe, s, t, node_property, common_factors,
                  total_factor_instances,
                  factor_aggregator, reverse_edges):
    co_freq = len(common_factors)
    s_freq, t_freq = _get_s_t_frequencies(
        pgframe, s, t, node_property, factor_aggregator, reverse_edges)

    return mutual_information(
        co_freq, s_freq, t_freq, total_factor_instances)


def _compute_npmi(pgframe, s, t, node_property, common_factors,
                  total_factor_instances,
                  factor_aggregator, reverse_edges):
    co_freq = len(common_factors)
    s_freq, t_freq = _get_s_t_frequencies(
        pgframe, s, t, node_property, factor_aggregator, reverse_edges)
    return mutual_information(
        co_freq, s_freq, t_freq, total_factor_instances,
        mitype="normalized")


COOCCURRENCE_STATISTICS = {
    "frequency": _compute_frequency,
    "ppmi": _compute_ppmi,
    "npmi": _compute_npmi
}


def schedule_scanning(task_queue, indices, n_workers):
    """Schedule scanning work."""
    for i in indices:
        task_queue.put(i)
    for _ in range(n_workers):
        task_queue.put(SENTINEL)
    task_queue.close()
    task_queue.join_thread()


def aggregate_index(x):
    return set(x.index)


class CooccurrenceGenerator(object):
    """Generator of co-occurrence edges from PGFrames.

    This interface allows to inspect nodes of the wrapped graph
    for their co-occurrence. The co-occurrence can be based on node
    properties: two nodes co-occur when they share some property values.
    For instance, two terms have common values in sets
    of papers in which they occur, i.e. two terms co-occur in the same papers.
    The co-occurrence can be also based on edge types: two nodes co-occur when
    they both have an edge of the same type pointing to the same target node.
    For example, two nodes representing terms have an edge of the type
    'occursIn' pointing to the same node representing a paper. The class
    generate edges between co-occurring nodes according to the input criteria
    and computes a set of statistics (frequency, PPMI, NPMI) quantifying
    their co-occurrence relationships.

    """

    def __init__(self, pgframe):
        self.pgframe = pgframe

    def _get_node_factors(self, s, t, node_property, factor_aggregator):
        if factor_aggregator is None:
            s_factors = self.pgframe._nodes.loc[s][node_property]
            t_factors = self.pgframe._nodes.loc[t][node_property]
        else:
            s_factors = factor_aggregator(self.pgframe._nodes.loc[s])
            t_factors = factor_aggregator(self.pgframe._nodes.loc[t])
        return s_factors, t_factors

    def _get_edge_factors(self, s, t, factor_aggregator, reverse_edges):
        if not reverse_edges:
            s_targets = self.pgframe._edges.xs(s, level=0, axis=0)
            t_targets = self.pgframe._edges.xs(t, level=0, axis=0)
        else:
            s_targets = self.pgframe._edges.xs(s, level=1, axis=0)
            t_targets = self.pgframe._edges.xs(t, level=1, axis=0)

        s_factors = factor_aggregator(s_targets)
        t_factors = factor_aggregator(t_targets)
        return s_factors, t_factors

    def _scan_targets(self, indices_to_nodes, node_property, source_index,
                      factor_aggregator, compute_statistics,
                      total_factor_instances,
                      generated_edges, reverse_edges=False, limit=None,
                      verbose=False):
        """Scan possible co-occurrence targets for the input source term."""
        edge_list = []
        for target_index in range(source_index + 1, len(indices_to_nodes)):
            s = indices_to_nodes[source_index]
            t = indices_to_nodes[target_index]

            if node_property is not None:
                s_factors, t_factors = self._get_node_factors(
                    s, t, node_property, factor_aggregator)
            else:
                if factor_aggregator is None:
                    factor_aggregator = aggregate_index
                s_factors, t_factors = self._get_edge_factors(
                    s, t, factor_aggregator, reverse_edges)

            common_factors = safe_intersection(
                s_factors, t_factors)

            if len(common_factors) > 0:
                edge = {
                    "@source_id": s,
                    "@target_id": t,
                    "common_factors": common_factors
                }

                for stat in compute_statistics:
                    edge[stat] = COOCCURRENCE_STATISTICS[stat](
                        self.pgframe, s, t,
                        node_property,
                        common_factors,
                        total_factor_instances,
                        factor_aggregator,
                        reverse_edges)

                edge_list.append(edge)

            if limit:
                if len(generated_edges) + len(edge_list) == limit:
                    if verbose:
                        print("Reached the edge limit ({})".format(limit))
                    return edge_list

        return edge_list

    def _scanning_loop(self, indices_to_nodes, node_property,
                       factor_aggregator, compute_statistics,
                       total_factor_instances,
                       all_edges, reverse_edges, task_queue,
                       generated_edges, limit=None, verbose=False):
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
                edge_list = self._scan_targets(
                    indices_to_nodes, node_property, source_index,
                    factor_aggregator, compute_statistics,
                    total_factor_instances,
                    all_edges, reverse_edges, limit=limit,
                    verbose=verbose)
                generated_edges += edge_list

    def _generate_cooccurrence(self, sources,
                               node_property,
                               factor_aggregator,
                               total_factor_instances,
                               reverse_edges,
                               compute_statistics,
                               parallelize, cores, limit, verbose=False):
        indices_to_nodes = {i: n for i, n in enumerate(sources)}

        all_edges = []

        total_pairs = (len(sources) * (len(sources) - 1)) / 2
        if verbose:
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
                    target=self._scanning_loop,
                    args=(
                        indices_to_nodes,
                        node_property,
                        factor_aggregator,
                        compute_statistics,
                        total_factor_instances,
                        all_edges,
                        reverse_edges, task_queue,
                        generated_edges, limit, verbose),
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
                edges = self._scan_targets(
                    indices_to_nodes,
                    node_property, source_index,
                    factor_aggregator,
                    compute_statistics,
                    total_factor_instances,
                    all_edges, reverse_edges, limit=limit)
                all_edges += edges
                if len(all_edges) == limit:
                    break
        if len(all_edges) > 0:
            edge_frame = pd.DataFrame(all_edges)
            edge_frame = edge_frame.set_index(
                ["@source_id", "@target_id"])
        else:
            edge_frame = pd.DataFrame(
                columns=[
                    "@source_id", "@target_id", "common_factors"
                ] + compute_statistics)
        return edge_frame

    def generate_from_nodes(self, node_property,
                            node_type=None,
                            factor_aggregator=None,
                            total_factor_instances=None,
                            compute_statistics=None,
                            parallelize=False,
                            cores=4, limit=None, verbose=False):
        if compute_statistics is None:
            compute_statistics = []
            total_factor_instances = None
        else:
            # Compute total instances of the occurrence factor
            # (needed for statistics)
            if total_factor_instances is None:
                if factor_aggregator is None:
                    total_factor_instances = len(_aggregate_values(
                        self.pgframe.nodes(
                            raw_frame=True, typed_by=node_type)))
                else:
                    if verbose:
                        print("Computing total factor instances...")
                    total_factor_instances = len(
                        factor_aggregator(
                            self.pgframe.nodes(
                                raw_frame=True, typed_by=node_type)))

        sources = self.pgframe.nodes(typed_by=node_type)

        return self._generate_cooccurrence(
            sources, node_property, factor_aggregator,
            total_factor_instances, None, compute_statistics,
            parallelize, cores, limit, verbose)

    def generate_from_edges(self, edge_type,
                            factor_aggregator=None,
                            total_factor_instances=None,
                            reverse_edges=False,
                            compute_statistics=None,
                            parallelize=False,
                            cores=4, limit=None, verbose=False):

        if compute_statistics is None:
            compute_statistics = []
            total_factor_instances = None
        else:
            # Compute total instances of the occurrence factor
            # (needed for statistics)
            if total_factor_instances is None:
                if factor_aggregator is None:
                    if not reverse_edges:
                        targets = self.pgframe._edges[
                            self.pgframe._edges["@type"] == edge_type
                        ].index.get_level_values(1).unique()
                    else:
                        targets = self.pgframe._edges[
                            self.pgframe._edges["@type"] == edge_type
                        ].index.get_level_values(0).unique()
                    if total_factor_instances is None:
                        total_factor_instances = len(targets)
                else:
                    if verbose:
                        print("Computing total factor instances...")
                    total_factor_instances = len(
                        factor_aggregator(self.pgframe._edges))

        if not reverse_edges:
            sources = self.pgframe._edges[
                self.pgframe._edges[
                    "@type"] == edge_type].index.get_level_values(
                        0).unique()
        else:
            sources = self.pgframe._edges[
                self.pgframe._edges[
                    "@type"] == edge_type].index.get_level_values(
                        1).unique()

        return self._generate_cooccurrence(
            sources, None, factor_aggregator,
            total_factor_instances, reverse_edges, compute_statistics,
            parallelize, cores, limit, verbose)
