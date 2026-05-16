import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

def predict(x, model):
    X = np.array(x).reshape(1, -1)
    model = pickle.load(open(model, 'rb'))
    y_pred = model.predict(X)
    return y_pred

def train(x, y , path_csv, path_pickle):
    df = pd.read_csv(path_csv)
    df.loc[len(df)] = [x, y]
    df.to_csv(path_csv, index=False)

    X, Y = df["x"].values.reshape(-1, 1), df["y"].values.reashape(-1, 1)
    model = LinearRegression()
    model.fit(X, Y)
    pickle.dump(model, open(path_pickle, 'wb'))