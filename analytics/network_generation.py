

def filter_unfrequent(data, factor, n, keep=None):
    filtered_data = data.nlargest(n, columns=["paper_frequency"])
    if keep is not None:
        for n in keep:
            index = [i.lower() for i in filtered_data.index]
            if n.lower() not in index and n in data.index:
                filtered_data.loc[n] = data.loc[n]
    return filtered_data


def cofrequence(occurrence_data, factor, s, t):
    intersection = occurrence_data.loc[s][factor].intersection(
        occurrence_data.loc[t][factor])
    return len(intersection)


def mutual_information(occurrence_data, factor, total_instances, s, t, mitype=None):
    co_freq = cofrequence(occurrence_data, factor, s, t)
    s_freq = occurrence_data.loc[s][factor + "_frequency"]
    t_freq = occurrence_data.loc[t][factor + "_frequency"]
    if co_freq > 0:
        if mitype is not None:
            if mitype == "expected":
                mi = math.log2((total_instances * co_freq) / (s_freq * t_freq)) * (co_freq / total_instances)
            elif mitype == "normalized":
                alpha = - math.log2(co_freq / total_instances)
                mi = math.log2((total_instances * co_freq) / (s_freq * t_freq)) / alpha
            elif mitype == "pmi2":
                mi = math.log2((co_freq ** 2) / (s_freq * t_freq))
            elif mitype == "pmi3":
                mi = math.log2((co_freq ** 3) / (s_freq * t_freq * total_instances))
        else:
            mi = math.log2((total_instances * co_freq) / (s_freq * t_freq))
    else:
        mi = 0
    return mi


def scan_targets(nodes, start_i, start_j, indices_to_nodes,
                 factor, limit, source_index):
    edge_list = []
    for target_index in range(source_index + 1, len(nodes)):
        if start_j is not None and source_index == start_i and\
           target_index <= start_j:
            pass
        else:
            s = indices_to_nodes[source_index]
            t = indices_to_nodes[target_index]

            frequency = cofrequence(occurrence_data, factor, s, t)

            if frequency > 0:
                ppmi = mutual_information(occurrence_data, factor, COUNTS[factor], s, t)
                npmi = mutual_information(occurrence_data, factor, COUNTS[factor], s, t, mitype="normalized")
#                 ppmi2 = mutual_information(occurrence_data, factor, COUNTS[factor], s, t, mitype="pmi2")
#                 ppmi3 = mutual_information(occurrence_data, factor, COUNTS[factor], s, t, mitype="pmi3")
                edge_list.append({
                    "source": s,
                    "target": t,
                    "frequency": frequency,
                    "ppmi": ppmi if ppmi > 0 else 0,
                    "npmi": npmi if npmi > 0 else 0,
#                     "ppmi2": ppmi2 if ppmi2 > 0 else 0,
#                     "ppmi3": ppmi3 if ppmi3 > 0 else 0,
                })
            if limit and len(edge_list) == limit:
                return edge_list

    return edge_list


def generate_comention_network(occurence_data, factor_column=None, limit=None,
                               parallelize=False):
    """Generate a term co-occurrence network."""
    nodes = sorted(occurence_data.index)
    indices_to_nodes = {i: n for i, n in enumerate(nodes)}

    all_edges = []

    if parallelize:
        pool = Pool()  # Create a multiprocessing Pool
        edges = pool.map(
            functools.partial(
                scan_targets,
                nodes, start_i, start_j,
                indices_to_nodes,
                factor, limit),
            range(start_i, len(nodes)))

        all_edges = sum(edges, [])
    else:
        for index in range(start_i, len(nodes)):
            all_edges += scan_targets(
                nodes, start_i, start_j,
                indices_to_nodes,
                factor, limit,
                index)
    print("Generated {} edges".format(len(all_edges)))

    return all_edges
