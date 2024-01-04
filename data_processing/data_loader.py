import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def drop_column(df, column_list):
    return df.drop(columns=column_list)
