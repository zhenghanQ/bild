import h5py
import pickle
import numpy as np
import pandas as pd

def projection(data, covars):
    """ data: pd.DataFrame - contains data values
    covars: pd.DataFrame - contains covariates to be removed
    returns: pd.DataFrame - data of data with the linear effect
    of covariates removed
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data input needs to be a dataframe")
    if not isinstance(covars, pd.DataFrame):
        raise ValueError("covar input needs to be a dataframe")

    X, C = data.values, covars.values
    P = np.eye(C.shape[0]) - C.dot(np.linalg.pinv(C.T.dot(C))).dot(C.T)
    return pd.DataFrame(P.dot(X), columns=data.columns, index=data.index)

def save_h5(path, data_obj, dts):
    """ path: path to save the data
    data_obj: python object that the data live in
    dts: data type of data_obj
    returns: saved path and keys of data_obj in the h5 file
    """
    if len(data_obj) != len(dts):
        raise ValueError("len of data_obj and dts must be the same")

    with h5py.File(path, "w") as data_store:
        for idx, (key, val) in enumerate(data_obj.items()):
            data_set = data_store.create_dataset(
                    key, val.shape, dtype=dts[idx]
                    )
            data_set[...] = val

    return path, data_obj.keys()

def read_h5(path, key):
    """path: path to the saved h5 file
    key: key in the h5 file that you want to load
    returns: data object (numpy array or pd.DataFrame, or list)
    """
    with h5py.File(path, "r") as file_store:
        data = file_store[key][...]

    return data

def save_pickle(path, data_obj):
    """ path: path to save the data to
    data_obj: python object hat the data liv in
    returns: path to where file was written
    """
    with open(path, "wb") as data_store:
        pickle.dump(data_obj, data_store, protocol=pickle.HIGHEST_PROTOCOL)

    return path

def read_pickle(path):
    """ path: path to where the file was written on computer
    return: data file (numpy array, list, pd.DataFrame, etc..)
    """
    with open(path, "rb") as data_obj:
        data = pickle.load(data_obj)

    return data
