import os
import utils
import prelim
import pandas as pd
import numpy as np
from collections import Counter

info_dict = dict(
    base_path="/Users/yoelsanchezaraujo/Documents/bild_stuff",
    voxels="fszstatcope2_nvoxels_nz.csv",
    behav="behav.txt",
    data="fszstatcope2_means_nz.csv"
)

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

ds = [
    ("kids_adults", "kids_adults_behav"),
    ("kids_adoles", "kids_adoles_behav"),
    ("adols_adults", "adols_adults_behav")
]

def rf_feature_selector(data, y, thresh, cv, clf, max_feat=10):
    
    if isinstance(data, pd.DataFrame):
        X = data.values
    elif isinstance(data, np.ndarray):
        X = data

    N, P = X.shape
    results = {"pred":np.zeros(N), "imp":[], "imp_idx":[]}
    results_null = {"pred":np.zeros(N), "imp":[], "imp_idx":[]}

    for idx, (train, test) in enumerate(cv.split(X, y)):
        sfm = SelectFromModel(clf, threshold=thresh)
        sfm.fit(X[train], y[train])
        X_transform = sfm.transform(X[train])
        n_features = X_transform.shape[1]
        
        while n_features > max_feat:
            sfm.threshold += 0.01
            X_transform = sfm.transform(X[train])
            n_features = X_transform.shape[1]

        clf.fit(X_transform, y[train])
        results["pred"][idx] = clf.predict(sfm.transform(X[test]))
        results["imp_idx"].append(sfm.get_support())
        results["imp"].append(clf.feature_importances_)

        try:
            y_shuff = np.copy(y[train])
            np.random.shuffle(y_shuff)
            clf.fit(X_transform, y_shuff)
            results_null["pred"][idx] = clf.predict(sfm.transform(X[test]))
            results_null["imp_idx"].append(sfm.get_support())
            results_null["imp"].append(clf.feature_importances_)
        except:
            print("couldn't compute null model iteration: {}".format(idx))


    return (results, r2_score(y, results["pred"]),
            results_null, r2_score(y, results_null["pred"]))

global_results = {}
vxt_vals = [27, 116]
data_sets = prelim.load_data_and_preproc(info_dict, 27)
use_data = data_sets[ds[0][0]]
use_behav = data_sets[ds[0][1]]
r2 = np.zeros(100)
r2_null = np.zeros(100)
idx = 0

while idx < 2:
    clf = RandomForestRegressor(n_estimators=1000)
    cv = LeaveOneOut()
    max_feat = 10
    thresh = 0.01

    if idx % 5 == 0:
        print("on iteration : {}".format(idx))

    try:
        res_alt, ex_alt, res_null, ex_null = rf_feature_selector(
            use_data, use_behav.old_kid.values, 
            thresh, cv, clf, max_feat
        )
        global_results["alt_{}".format(idx)] = res_alt
        r2[idx] = ex_alt
        global_results["null_{}".format(idx)] = res_null
        r2_null[idx] = ex_null
        idx += 1
    except:
        pass

# need to write code for writing plots and saving results to disk
