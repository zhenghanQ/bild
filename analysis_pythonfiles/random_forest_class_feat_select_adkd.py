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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

ds = [
    ("kids_adults", "kids_adults_behav"),
    ("kids_adoles", "kids_adoles_behav"),
    ("adols_adults", "adols_adults_behav")
]

par_type = ds[0][0]
beh_type = ds[0][1]

def rf_feature_selector(data, y, thresh, cv, clf, max_feat=10):
    
    if isinstance(data, pd.DataFrame):
        X = data.values
    elif isinstance(data, np.ndarray):
        X = data

    N, P = X.shape
    results = {"pred":np.zeros(N), "imp":[], "imp_idx":[]}
    results_null = np.zeros(N)

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
            results_null[idx] = clf.predict(sfm.transform(X[test]))
        except:
            print("couldn't compute null model iteration: {}".format(idx))


    return (results, roc_auc_score(y, results["pred"]),
             roc_auc_score(y, results_null))

global_results = {}
vxt_vals = [27, 116]
data_sets = prelim.load_data_and_preproc(info_dict, 27)
use_data = data_sets[par_type]
use_behav = data_sets[beh_type]
rocs = np.zeros(100)
rocs_null = np.zeros(100)
total_iters, idx = 100, 0
npars = use_data.shape[0]

while idx < 100:
    clf = RandomForestClassifier(n_estimators=1000)
    cv = LeaveOneOut()
    max_feat = 10
    thresh = 0.01

    if idx % 5 == 0:
        print("on iteration : {}".format(idx))

    try:
        res_alt, roc_alt, roc_null = rf_feature_selector(
            use_data, use_behav.young_kid.values, # remember to change this 
            thresh, cv, clf, max_feat
        )
        global_results["alt_{}".format(idx)] = res_alt
        rocs[idx] = roc_alt
        rocs_null[idx] = roc_null
        idx += 1
    except:
        pass

def extract_from_dict(indict, mod_type, use_cols, niters, npars):
    ncols = use_cols.shape[0]
    imps = np.zeros((niters, npars, ncols))
    feats = []
    # get the feature names selected
    for idx in np.arange(niters):
        mfmt = mod_type + "_{}".format(idx)
        for cvidx in np.arange(npars):
            feats.extend(
                use_cols[indict[mfmt]["imp_idx"][cvidx]].tolist()
            )
            # now get the importances
            row_idxs = indict[mfmt]["imp_idx"][cvidx]
            imps[idx, cvidx, row_idxs] = indict[mfmt]["imp"][cvidx]

    return Counter(feats), imps

alt_rdata = extract_from_dict(global_results, "alt", use_data.columns.values, total_iters, npars)

# plot and save roc histogram
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(rocs, edgecolor="black")
plt.hist(rocs_null, edgecolor="black", alpha=.7)
plt.axvline(rocs.mean(), ls="--", color="purple")
plt.axvline(rocs_null.mean(), ls="--", color="red")
plt.legend([
    "alternative model mean = {}".format(rocs.mean()),
    "null model mean = {}".format(rocs_null.mean()),
    "alternative model", 
    "null model"
])
plt.savefig("{}_rochist_rffeatselect_100iters.png".format(par_type), bbox_inches="tight", dpi=300)
plt.close()

# save results as pickle files
save_pkl_name = "feat_counts_and_imps_{}_rffeatselect_100iters.pkl".format(par_type)
utils.save_pickle(save_pkl_name, alt_rdata)
