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
X, N, P = utils.extract_values(use_data)
# until here

# making some data that is well explained by a few of the features 
dummy_response = (
    3*np.sin(2*np.pi*data_sets[ds[0][0]].iloc[:, 1]) + data_sets[ds[0][0]].iloc[:, 5]**2 + 
    0.45*data_sets[ds[0][0]].iloc[:, 10] + np.exp(data_sets[ds[0][0]].iloc[:, 15])
)


for idx in np.arange(niters):
    clf = RandomForestRegressor(n_estimators=1000)
    cv = LeaveOneOut()
    max_feat = 10
    thresh = 0.01

    if idx % 5 == 0:
        print("on iteration : {}".format(idx))

    try:
        res_alt, ex_alt, res_null, ex_null = models.rf_feature_selector(
            use_data, dummy_response.values, 
            thresh, cv, clf, max_feat
        )
        global_results["alt_{}".format(idx)] = res_alt
        r2[idx] = ex_alt
        global_results["null_{}".format(idx)] = res_null
        r2_null[idx] = ex_null
    except:
        pass

# need to write code for writing plots and saving results to disk
