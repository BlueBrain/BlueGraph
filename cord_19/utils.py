"""Module containing utils for COVID-19 network generation and analysis."""
import ast
import operator
import pickle

import pandas as pd
import networkx as nx

from collections import Counter

from networkx.readwrite.json_graph.cytoscape import cytoscape_data

from kganalytics.network_generation import generate_cooccurrence_network
from kganalytics.metrics import compute_all_metrics

from kganalytics.export import (save_nodes,
                                save_to_gephi,
                                save_network)

from kganalytics.paths import minimum_spanning_tree


NON_ASCII_REPLACE = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "κ": "kappa",
    "’": "'",
    "–": "-",
    "‐": "-",
    "é": "e",
    "ó": "o"
}


def clean_up_entity(s):
    """Clean-up entity by removing common errors from NER."""
    s = str(s).lower()
    result = s.strip().strip("\"").strip("\'")\
        .strip("&").strip("#").replace(".", "").replace("- ", "-")

    # cleaning-up non-ascii symbols
    for s in result:
        try:
            s.encode('ascii')
        except UnicodeEncodeError:
            if s in NON_ASCII_REPLACE:
                s_ascii = NON_ASCII_REPLACE[s]
            else:
                s_ascii = ""
            result = result.replace(s, s_ascii)

    return result


def has_min_length(entities, length):
    """Check if a term has the min required length."""
    return entities.apply(lambda x: len(x) > length - 1)


def is_experiment_related(title):
    """Check if the title is experiment related."""
    name = title.split(":")[1].lower()

    long_keywords = [
        "method",
        "material",
        "experimental"
        # "materials and method",
        # "methods and material",
        # "experimental framework",
        # "experimental method",
        # "experimentation methodology",
        # "experimental design",
        # "experimental setup",
        # "experimental set-up",
        # "methods and experiment",
        # "experimental material",
        # "subjects and method",
        # "methods & materials",
        # "material and metod",
    ]

#     if section_name in short_keywords:
#         return True
    for k in long_keywords:
        if k in name:
            return True

    return False


def mentions_to_occurrence(raw_data,
                           term_column="entity",
                           factor_columns=None,
                           term_cleanup=None,
                           term_filter=None,
                           mention_filter=None,
                           aggregation_function=None,
                           dump_prefix=None):
    """Convert a raw mentions data into occurrence data.

    This function converts entity mentions into a dataframe
    indexed by unique entities. Each row contains aggregated data
    for different occurrence factors (sets of factor instances where
    the given term occurrs, e.g. sets of papers/section/paragraphs where the
    give term was mentioned).

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Dataframe containing occurrence data with the following columns:
        one column for terms, one or more columns for occurrence factors (
        e.g. paper/section or paragraph of a term occurrence).
    term_column : str
        Name of the column with terms
    factor_columns : collection of str
        Set of column names containing occurrence factors (e.g.
        "paper"/"section"/"paragraph").
    term_cleanup : func, optional
        A clean-up function to be applied to every term
    term_filter : func, optional
        A filter function to apply to terms (e.g. include
        terms only with 2 or more symbols)
    mention_filter : func, optional
        A filter function to apply to occurrence factors (e.g. filter out
        all the occurrences in sections called "Methods")
    aggregation_function : func, optional
        Function to be applied to aggregated occurrence factors. By default,
        the constructor of `set`.
    dump_prefix : str, optional
        Prefix to use for dumping the resulting occurrence dataset.

    Returns
    -------
    occurence_data : pd.DataFrame
        Dataframe indexed by distinct terms containing aggregated occurrences
        of terms as columns (e.g. for each terms, sets of papers/sections/
        paragraphs) where the term occurs.
    factor_counts : dict
        Dictionary whose keys are factor column names (
        e.g. "paper"/"section"/"paragraph") and whose values are counts of
        unique factor instances (e.g. total number of papers/sections/
        paragraphs in the dataset)
    """
    print("Cleaning up the entities...")

    if factor_columns is None:
        factor_columns = []

    if term_cleanup is not None:
        raw_data[term_column] = raw_data[term_column].apply(term_cleanup)

    if term_filter is not None:
        raw_data = raw_data[term_filter(raw_data[term_column])]

    if mention_filter is not None:
        raw_data = raw_data[mention_filter(raw_data)]

    factor_counts = {}
    for factor_column in factor_columns:
        factor_counts[factor_column] = len(raw_data[factor_column].unique())

    print("Aggregating occurrences of entities....")
    if aggregation_function is None:
        aggregation_function = set

    occurence_data = raw_data.groupby(term_column).aggregate(
        lambda x: aggregation_function(x))

    if dump_prefix is not None:
        print("Saving the occurrence data....")
        with open("{}occurrence_data.pkl".format(dump_prefix), "wb") as f:
            pickle.dump(occurence_data, f)

        with open("{}counts.pkl".format(dump_prefix), "wb") as f:
            pickle.dump(factor_counts, f)

    return occurence_data, factor_counts


def aggregate_cord_entities(x, factors):
    """Aggregate a collection of entity mentions.

    Entity types are aggregated as lists (to preserve the multiplicity,
    e.g. how many times a given entity was recognized as a particular type).
    The rest of the input occurrence factors are aggregated as sets
    (e.g. sets of unique papers/sections/paragraphs).
    """
    result = {
        "entity_type": list(x.entity_type)
    }
    for f in factors:
        result[f] = set(x[f])

    return result


def prepare_occurrence_data(mentions_df=None,
                            mentions_path=None,
                            occurrence_data_path=None,
                            factors=None,
                            factor_counts_path=None):
    """Prepare mentions data for the co-occurrence analysis.

    mentions_df : pd.DataFrame, optional
    mentions_path : str, optional
    occurrence_data_path : str, optional
    factors : list of str, optional
    factor_counts_path : str, optional

    """
    if factors is None:
        # use all factors
        factors = ["paper", "section", "paragraph"]

    mentions = None
    if mentions_path:
        # Read raw mentions and transform them to the occurrence data
        if "pkl" in mentions_path:
            with open(mentions_path, "rb") as f:
                mentions = pickle.load(f)
        else:
            mentions = pd.read_csv(f)
    elif mentions_df is not None:
        mentions = mentions_df

    if mentions is not None:
        mentions = mentions[["entity", "entity_type", "occurrence"]]
        mentions["paper"] = mentions["occurrence"].apply(
            lambda x: x.split(":")[0])
        mentions["section"] = mentions["occurrence"].apply(
            lambda x: ":".join([x.split(":")[0], x.split(":")[1]]))
        mentions = mentions.rename(columns={"occurrence": "paragraph"})
        occurrence_data, counts = mentions_to_occurrence(
            mentions,
            term_column="entity",
            factor_columns=factors,
            term_cleanup=clean_up_entity,
            term_filter=lambda x: has_min_length(x, 3),
            aggregation_function=lambda x: aggregate_cord_entities(x, factors),
            mention_filter=lambda data: ~data["section"].apply(
                is_experiment_related))

        # Filter entities that occur only once (only in one paragraph,
        # usually represent noisy terms)
        occurrence_data = occurrence_data[occurrence_data["paragraph"].apply(
            lambda x: len(x) > 1)]
        if occurrence_data_path:
            print("Saving pre-calculated occurrence data....")
            with open(occurrence_data_path, "wb") as f:
                pickle.dump(occurrence_data, f)
        if factor_counts_path:
            with open(factor_counts_path, "wb") as f:
                pickle.dump(counts, f)

    elif occurrence_data_path is not None:
        with open(occurrence_data_path, "rb") as f:
            occurrence_data = pickle.load(f)
    else:
        occurrence_data = None
        counts = None

    if counts is None and factor_counts_path:
        with open(factor_counts_path, "rb") as f:
            counts = pickle.load(f)
    return occurrence_data, counts


def generate_curation_table(data):
    data, counts = prepare_occurrence_data(data)
    data = data.reset_index()
    data["paper_frequency"] = data[
        "paper"].transform(lambda x:  len([str(p).split(":")[0] for p in x]))
    data["paper"] = data[
        "paper"].transform(lambda x:  list(x))
    data["paragraph"] = data[
        "paragraph"].transform(lambda x:  list(x))
    data["section"] = data[
        "section"].transform(lambda x:  list(x))
    data["raw_frequency"] = data["entity_type"].apply(len)
    data["entity_type"] = data[
        "entity_type"].transform(
            lambda x:  ", ".join(list(set(x))))
    return data, counts


def merge_with_ontology_linking(occurence_data,
                                factor_columns,
                                linking_df=None,
                                linking_path=None,
                                linked_occurrence_data_path=None):
    """Merge occurrence data with ontology linking data."""
    if factor_columns is None:
        factor_columns = ["paper", "section", "paragraph"]

    # Ontology linking
    linking = None
    if linking_path:
        # Open ontology linking files
        print("Loading the ontology linking...")
        if "pkl" in linking_path:
            with open(linking_path, "rb") as f:
                linking = pickle.load(f)
        else:
            linking = pd.read_csv(linking_path)
    elif linking_df is not None:
        linking = linking_df

    if linking is not None:
        linking = linking.rename(columns={"mention": "entity"})
        linking["concept"] = linking["concept"].apply(lambda x: x.lower())
        linking["entity"] = linking["entity"].apply(lambda x: x.lower())
        # The provided occcurence_data is expected to be lower cased and
        # the merge is performed on the 'entity' column and not the column one.

        print("Merging the occurrence data with the ontology linking...")

        # Merge occurrence data with the linking data
        occurence_data = occurence_data.reset_index()
        occurence_data["raw_types"] = occurence_data.apply(
            lambda x: [x.entity_type] * x.raw_frequency, axis=1)
        merged_data = occurence_data.merge(
            linking, on="entity", how="left")
        merged_data.loc[
            merged_data["concept"].isna(), "concept"] = merged_data[
                merged_data["concept"].isna()]["entity"]
        for col in factor_columns:
            merged_data[col] = merged_data[col].apply(lambda x: list(x))

        def aggregate_linking_data(x, factors):
            if x.name == "entity":
                return list(x)
            elif x.name in factors:
                return set(sum(x, []))
            elif x.name in ["uid", "definition", "taxonomy", "semantic_type"]:
                return list(x)[0]
            return sum(x, [])

        occurrence_data_linked = merged_data.groupby("concept").aggregate(
            lambda x: aggregate_linking_data(x, factor_columns))

        occurrence_data_linked = occurrence_data_linked.reset_index()
        occurrence_data_linked = occurrence_data_linked.rename(
            columns={
                "concept": "entity",
                "entity": "aggregated_entities"
            })
        occurrence_data_linked = occurrence_data_linked.set_index("entity")

        if linked_occurrence_data_path:
            with open(linked_occurrence_data_path, "wb") as f:
                print("Saving pre-calculated linked occurrence data....")
                pickle.dump(occurrence_data_linked, f)
        return occurrence_data_linked
    elif linked_occurrence_data_path:
        print("Loading linked occurrence data...")
        with open(linked_occurrence_data_path, "rb") as f:
            occurrence_data_linked = pickle.load(f)
    else:
        raise ValueError(
            "Neither linking data nor pre-computed linked occurrence "
            "data has been specified"
        )
    return occurrence_data_linked


def generate_comention_analysis(occurrence_data, counts, type_data=None,
                                min_occurrences=1, n_most_frequent=None,
                                keep=None, factors=None, cores=None,
                                graph_dump_prefix=None,
                                communities=True, remove_zero_mi=False):
    # Filter entities that occur only once (only in one paragraph, usually
    # represent noisy terms)
    occurrence_data = occurrence_data[occurrence_data["paragraph"].apply(
        lambda x: len(x) >= min_occurrences)]
    occurrence_data["paragraph_frequency"] = occurrence_data["paragraph"].apply(
        lambda x: len(x))
    occurrence_data["section_frequency"] = occurrence_data["section"].apply(
        lambda x: len(x))
    occurrence_data["paper_frequency"] = occurrence_data["paper"].apply(
        lambda x: len(x))

    graphs = {}
    trees = {}
    for f in factors:
        print("-------------------------------")
        print("Factor: {}".format(f))
        print("-------------------------------")

        if cores is None:
            cores = 8
        graph = generate_cooccurrence_network(
            occurrence_data,
            f,
            counts[f],
            n_most_frequent=n_most_frequent,
            dump_path=(
                "{}_{}_edge_list.pkl".format(graph_dump_prefix, f)
                if graph_dump_prefix else None
            ),
            keep=keep,
            parallelize=True,
            cores=cores)

        if graph is not None:
            graphs[f] = graph
        else:
            return None, None
        print()

        # Remove edges with zero mutual information
        if remove_zero_mi:
            edges_to_remove = [
                e
                for e in graphs[f].edges()
                if graphs[f].edges[e]["ppmi"] == 0
            ]
            for s, t in edges_to_remove:
                graphs[f].remove_edge(s, t)

        # Set entity types
        if type_data is not None:
            type_dict = type_data.to_dict()["type"]
            nx.set_node_attributes(
                graphs[f],
                type_dict,
                "entity_type")

        # Set factors as attrs
        nx.set_node_attributes(
            graphs[f],
            occurrence_data["paper"].apply(lambda x: list(x)).to_dict(),
            "paper")

        compute_all_metrics(
            graphs[f],
            degree_weights=["frequency"],
            betweenness_weights=[],
            community_weights=["frequency", "npmi"] if communities else [],
            print_summary=True)

        # Compute a spanning tree
        print("Computing the minimum spanning tree...")
        trees[f] = minimum_spanning_tree(graphs[f], weight="distance_npmi")
        nx.set_node_attributes(
            trees[f],
            type_dict,
            "entity_type")

        if graph_dump_prefix:
            save_network(trees[f], "{}_{}_tree".format(graph_dump_prefix, f))

        if graph_dump_prefix:
            save_nodes(
                graphs[f],
                "{}_{}_node_list.pkl".format(graph_dump_prefix, f))

    return graphs, trees


def generate_full_analysis(mentions_df=None, mentions_path=None,
                           occurrence_data_path=None, factor_counts_path=None,
                           linking_df=None, linking_path=None,
                           linked_occurrence_data_path=None,
                           type_mapping=None, type_data_path=None,
                           n_most_frequent=None, min_occurrences=None,
                           graph_dump_prefix=None, gephi_dump_prefix=None,
                           factors=None, cores=None):

    if factors is None:
        factors = ["paper", "section", "paragraph"]

    # Converting mention data into occurrences
    occurrence_data, counts = prepare_occurrence_data(
        mentions_df=mentions_df,
        mentions_path=mentions_path,
        occurrence_data_path=occurrence_data_path,
        factors=factors,
        factor_counts_path=factor_counts_path)

    # Ontology linking
    occurrence_data_linked = merge_with_ontology_linking(
        occurrence_data,
        factor_columns=factors,
        linking_df=linking_df,
        linking_path=linking_path,
        linked_occurrence_data_path=linked_occurrence_data_path)

    # Peform type mapping according to the provided dictionary
    if type_data_path:
        type_data = pd.read_pickle(type_data_path)
    elif type_mapping is not None:
        type_data = resolve_taxonomy_to_types(
            occurrence_data_linked, type_mapping)
        if type_data_path:
            type_data.to_pickle(type_data_path)

    graphs, trees = generate_comention_analysis(
        occurrence_data_linked, counts,
        min_occurrences=min_occurrences,
        type_data=type_data,
        n_most_frequent=n_most_frequent,
        graph_dump_prefix=graph_dump_prefix,
        factors=factors,
        cores=cores)

    for f in factors:
        if gephi_dump_prefix:
            node_attr_mapping = {
                "degree_frequency": "Degree",
                "community_npmi": "Community",
            }
            if type_data is not None:
                node_attr_mapping["entity_type"] = "Type"

            edge_attr_mapping = {
                "npmi": "Weight"
            }

            save_to_gephi(
                graphs[f],
                "{}_graph_{}".format(gephi_dump_prefix, f),
                node_attr_mapping=node_attr_mapping,
                edge_attr_mapping=edge_attr_mapping)

            save_to_gephi(
                trees[f],
                "{}_tree_{}".format(gephi_dump_prefix, f),
                node_attr_mapping=node_attr_mapping,
                edge_attr_mapping=edge_attr_mapping)

    return graphs, trees


def resolve_taxonomy_to_types(occurrence_data, mapping):

    def assign_raw_type(x):
        counts = {}
        for t in x:
            if t in counts:
                counts[t] += 1
            else:
                counts[t] = 1
        return max(counts.items(), key=operator.itemgetter(1))[0]

    def assign_mapped_type(x):
        taxonomy = (
            ast.literal_eval(x.taxonomy)
            if isinstance(x.taxonomy, str)
            else x.taxonomy
        )
        types = [el for _, el in taxonomy]
        result_type = None

        for target_type, taxonomy_classes in mapping.items():
            include_found = False
            exclude_found = False
            if "include" in taxonomy_classes:
                include_found = False
                for t in taxonomy_classes["include"]:
                    if t in types:
                        include_found = True
                        break
            if "exclude" in taxonomy_classes:
                for t in taxonomy_classes["exclude"]:
                    if t in types:
                        exclude_found = True
                        break
            if include_found and not exclude_found:
                result_type = target_type
                break

        if result_type is None:
            result_type = assign_raw_type(x.raw_types)

        return result_type

    type_data = pd.DataFrame(index=occurrence_data.index, columns=["type"])

    known_taxonomy = occurrence_data[occurrence_data["taxonomy"].notna()]
    type_data.loc[known_taxonomy.index, "type"] = known_taxonomy.apply(
        assign_mapped_type, axis=1)

    unknown_taxonomy = occurrence_data[occurrence_data["taxonomy"].isna()]
    type_data.loc[unknown_taxonomy.index, "type"] = unknown_taxonomy[
        "raw_types"].apply(assign_raw_type)
    return type_data


def link_ontology(linking, type_mapping, curated_table):
    linked_table = merge_with_ontology_linking(
        curated_table,
        factor_columns=["paper", "section", "paragraph"],
        linking_df=linking.rename(columns={"mention": "entity"}))

    linked_table = linked_table.reset_index()
    linked_table["paper_frequency"] = linked_table["paper"].apply(lambda x: len(x))
    linked_table["paper"] = linked_table["paper"].apply(lambda x: list(x))
    linked_table["section"] = linked_table["section"].apply(lambda x: list(x))
    linked_table["paragraph"] = linked_table["paragraph"].apply(lambda x: list(x))

    # Produce a dataframe with entity types according to type_mapping
    types = resolve_taxonomy_to_types(
        linked_table.set_index("entity"),
        type_mapping)
    linked_table = linked_table.merge(types, on="entity", how="left").rename(
        columns={"type": "entity_type"})
    linked_table["entity_type_label"] = linked_table["entity_type"]
    return linked_table


def generate_paper_lookup(graph):
    paper_table = {}
    for n in graph.nodes():
        if "paper" in graph.nodes[n]:
            paper_table[n] = list(graph.nodes[n]["paper"])
    return paper_table


def build_cytoscape_data(graph, positions=None):
    elements = cytoscape_data(graph)

    if positions is not None:
        for el in elements["elements"]['nodes']:
            if el["data"]["id"] in positions:
                el["position"] = positions[el["data"]["id"]]

    elements = elements["elements"]['nodes'] + elements["elements"]['edges']
    for element in elements:
        element["data"]["id"] = (
            str(element["data"]["source"] + '_' + element[
                "data"]["target"]).replace(" ", "_")
            if "source" in element["data"]
            else element["data"]["id"]
        )
        papers = []
        if 'paper' in element["data"]:
            papers = element["data"].pop("paper")
            element["data"]["paper_frequency"] = len(papers)

        if "source" in element["data"]:
            element["data"]["type"] = "edge"
        else:
            element["data"]["type"] = "node"

    return elements


def most_common(x):
    c = Counter(x)
    return c.most_common(1)[0][0]


CORD_ATTRS_RESOLVER = {
    "entity_type": most_common,
    "paper": lambda x: list(set(sum(x, []))),
    "degree_frequency": sum,
    "pagerank_frequency": max,
    "community_frequency": most_common,
    "community_npmi": most_common,
    "frequency": sum,
    "ppmi": max,
    "npmi": max,
    "distance_ppmi": min,
    "distance_npmi": min
}
