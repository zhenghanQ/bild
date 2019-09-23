import subprocess
import pickle

def shell(cmd_split):
    process = subprocess.Popen(cmd_split, stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out.decode("utf-8").split("\n")

def save_pickle(save_path, data):
    with open(save_path, "wb") as file_store:
        pickle.dump(data, file_store, pickle.HIGHEST_PROTOCOL)
    return save_path

def read_pickle(data_path):
    with open(data_path, "rb") as file_store:
        data = pickle.load(file_store)
    return data


