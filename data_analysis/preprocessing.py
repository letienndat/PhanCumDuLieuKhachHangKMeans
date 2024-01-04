from sklearn.preprocessing import StandardScaler


def standardize_data(df, columns):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[columns])
    return X
