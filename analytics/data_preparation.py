"""Module for co-mention network preparation."""
import pickle


def dummy_clean_up(s):
    """Dummy clean-up to remove some errors from NER."""
    s = str(s).lower()
    result = s.strip().strip("\"").strip("\'")\
        .strip("&").strip("#").replace(".", "")
    return result


def is_not_single_letter(entities):
    """Check if a term is not a single letter."""
    return entities.apply(lambda x: len(x) > 1)


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
                           factor_columns=[],
                           term_cleanup=None,
                           term_filter=None,
                           mention_filter=None,
                           filter_methods=False,
                           dump_prefix=None):
    """Convert a raw mentions data into occurrence data."""
    print("Cleaning up the entities...")

    # Entiti
    if term_cleanup is not None:
        raw_data[term_column] = raw_data[term_column].apply(term_cleanup)

    if term_filter is not None:
        raw_data = raw_data[term_filter(raw_data)]

    if mention_filter is not None:
        raw_data = raw_data[mention_filter(raw_data)]

    factor_counts = {}
    for factor_column in factor_columns:
        factor_counts[factor_column] = len(raw_data[factor_column].unique())

    print("Aggregating occurrences of entities....")
    occurence_data = raw_data.groupby(term_column).aggregate(
        lambda x: set(x))

    if dump_prefix is not None:
        print("Saving the occurrence data....")
        with open("{}_occurrence_data.pkl".format(dump_prefix), "wb") as f:
            pickle.dump(occurence_data, f)

        with open("{}_counts.pkl".format(dump_prefix), "wb") as f:
            pickle.dump(factor_counts, f)

    return occurence_data, factor_counts


def merge_with_ontology_linking(occurence_data, linking_data):