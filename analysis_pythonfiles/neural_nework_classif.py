import os
import utils
import prelim
import models
import pandas as pd
import numpy as np
from collections import Counter

info_dict = dict(
    base_path="/Users/yoelsanchezaraujo/Documents/bild_stuff",
    voxels="fszstatcope2_nvoxels_nz.csv",
    behav="behav.txt",
    data="fszstatcope2_means_nz.csv"
)

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

ds = [
    ("kids_adults", "kids_adults_behav"),
    ("kids_adoles", "kids_adoles_behav"),
    ("adols_adults", "adols_adults_behav")
]

niters = 5
global_results = {}
vxt_vals = [27, 116]
data_sets = prelim.load_data_and_preproc(info_dict, vxt_vals[0])
use_data = data_sets[ds[0][0]]
use_behav = data_sets[ds[0][1]]
r2 = np.zeros(niters)
r2_null = np.zeros(niters)

# to delete this below
X, shape = utils.extract_values(use_data)
X = X.astype(np.float32)
N, P = shape
Y = use_behav.young_kid.values[:, None].astype(np.float32)
lr = 0.01
n_hidden_0 = 10
n_hidden_0 = 10
n_input = shape
n_classes = 2
train_epochs=10
cost_history = np.empty(shape=[1], dtype=float)

# testing
a, b, c, d = simple_ann_model(use_data, use_behav.young_kid.values, 100, 10, 0.01)

