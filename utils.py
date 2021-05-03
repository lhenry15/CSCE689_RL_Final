import numpy as np
import pandas as pd

def read_csv(file, header=None, sep=',', index_col=None, skiprows=None, usecols=None, encoding='utf-8'):
    """Loads data from a CSV

    Returns:
        DataFrame
    """
    data_df = pd.read_csv(file, header=0, sep=sep, index_col=index_col, skiprows=skiprows, usecols=usecols, encoding=encoding)

    return data_df

def dataframe_to_matrix(df, labelindex=0, startcol=1):
    """ Converts a python dataframe in the expected anomaly dataset format to numpy arrays.

    The expected anomaly dataset format is a CSV with the label ('anomaly'/'nominal')
    as the first column. Other columns are numerical features.

    Note: Both 'labelindex' and 'startcol' are 0-indexed.
        This is different from the 'read_data_as_matrix()' method where
        the 'opts' parameter has same-named attributes but are 1-indexed.

    :param df: Pandas dataframe
    :param labelindex: 0-indexed column number that refers to the class label
    :param startcol: 0-indexed column number that refers to the first column in the dataframe
    :return: (np.ndarray, np.array)
    """
    cols = df.shape[1] - startcol
    x = np.zeros(shape=(df.shape[0], cols))
    for i in range(cols):
        x[:, i] = df.iloc[:, i + startcol]
    labels = np.array([0 if df.iloc[i, labelindex] == "anomaly" else 1 for i in range(df.shape[0])], dtype=int)

    return x, labels


def read_data_as_matrix(datapath):
    """ Reads data from CSV file and returns numpy matrix.

    Important: Assumes that the first column has the label \in {anomaly, nominal}

    :param datapath: the path of the data file
    :return: numpy.ndarray
    """
    with open(datapath, "r") as f:
        data = read_csv(f, sep=',')
    X_train, labels = dataframe_to_matrix(data)
    anomalies = np.where(labels == 0)[0]
    return X_train, labels, anomalies
    
def evaluate(e, agent, num=10):
    """ Evaluate the performance
    """
    results = []
    for _ in range(num):
        s = e.reset()
        done = False
        total_r = 0.0
        while not done:
            a = agent.step(s)
            s, r, done, _ = e.step(a)
            total_r += r
        results.append(total_r)
    return np.mean(results), np.std(results)

def run_iforest(X):
    """ Predict the anomaly score with iForest
    """
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest()
    clf.fit(X)
    scores = clf.decision_function(X)
    feature_importances = [est.feature_importances_ for est in clf.estimators_]

    return scores, feature_importances
