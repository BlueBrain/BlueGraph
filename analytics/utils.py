import pandas as pd


def top_n(data_dict, n):
    df = pd.DataFrame(dict(data_dict).items(), columns=["id", "value"])
    df = df.nlargest(n, columns=["value"])
    return(list(df["id"]))
