import math

import pandas as pd


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
