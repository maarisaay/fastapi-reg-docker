import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def predict(x, path2pickle):
    X = np.array([x]).reshape(-1, 1)
    model = pickle.load(open(path2pickle, "rb"))
    return model.predict(X)[0]


def train(x, y, path2csv, path2pickle):
    df = pd.read_csv(path2csv)
    df.loc[len(df)] = [x, y]
    df.to_csv(path2csv, index=False)

    model = LinearRegression()
    model.fit(df[["x"]].values, df["y"].values)

    pickle.dump(model, open(path2pickle, 'wb'))
    
    return model

def get_model_params(path2pickle):
    model = pickle.load(open(path2pickle, "rb"))

    return {
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_)
    }