import math

import pandas as pd


def normalize_to_set(value):
    if not isinstance(value, set):
        print(value, type(value))
        if math.isnan(value):
            return set()
        elif isinstance(value, str) or not math.isnan(value):
            return {value}
    return value


def _aggregate_values(values):
    value_set = set()
    for el in values:
        if isinstance(el, set):
            value_set.update(el)
        elif isinstance(el, str):
            value_set.add(el)
        elif not math.isnan(el):
            value_set.add(el)
    if len(value_set) == 1:
        return list(value_set)[0]
    elif len(value_set) == 0:
        return math.nan
    return value_set


def safe_intersection(set1, set2):
    set1 = normalize_to_set(set1)
    set2 = normalize_to_set(set2)
    return set1.intersection(set2)


def element_has_type(element_type, query_type):
    if not isinstance(element_type, set):
        element_type = {element_type}
    if not isinstance(query_type, set):
        query_type = {query_type}
    return query_type.issubset(element_type)


def str_to_set(s):
    """Parse string representation of a set."""
    if s[0] == "{":
        s = s[1:-1]
        return set([t.strip()[1:-1] for t in s.split(",")])
    return s


def top_n(data_dict, n, smallest=False):
    """Return top `n` keys of the input dictionary by their value."""
    df = pd.DataFrame(dict(data_dict).items(), columns=["id", "value"])
    if smallest:
        df = df.nsmallest(n, columns=["value"])
    else:
        df = df.nlargest(n, columns=["value"])
    return(list(df["id"]))
