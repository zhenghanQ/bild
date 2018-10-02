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
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold

global_results = {}

vxt_vals = [27, 116, 127]

ds = [
    ("kids_adults", "kids_adults_behav"),
    ("kids_adoles", "kids_adoles_behav"),
    ("adols_adults", "adols_adults_behav")
]

for vxt in [27, 116, 127]:
    global_results["voxel_thresh_{}".format(vxt)] = {}
    data_sets = prelim.load_data_and_preproc(info_dict, vxt)
    inner_cv = KFold(n_splits=5)
    outer_cv = LeaveOneOut()

    clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", linear_model.RidgeCV(cv=inner_cv))
    ])

    for d in ds:
        X = data_sets[d[0]].values
        y = data_sets[d[1]].age.values
        N, P = X.shape
        results_dict = {"pred":np.zeros(N), "coef":[]}

        for idx, (train, test) in enumerate(outer_cv.split(X, y)):
            clf.fit(X[train], y[train])
            results_dict["pred"][idx] = clf.predict(X[test])
            results_dict["coef"].append(clf.named_steps["lr"].coef_)

        results_dict["r2"] = r2_score(y, results_dict["pred"])
        results_dict["expvar"] = explained_variance_score(y, results_dict["pred"])

        global_results["voxel_thresh_{}".format(vxt)][d[0]] = results_dict

import matplotlib.pyplot as plt
import seaborn as sns

# this plot is to look at the performance measured when varying the voxel cut off 
df_vxt_res = pd.DataFrame(
    dict(
        vtx = ["voxel_threshold_27"]*3 + ["voxel_threshold_116"]*3 + ["voxel_threshold_127"]*3,

        r2 = [ii for xx in [[global_results["voxel_thresh_{}".format(vx)][subtype[0]]["r2"]
              for subtype in ds] for vx in vxt_vals] for ii in xx],

        expvar = [ii for xx in [[global_results["voxel_thresh_{}".format(vx)][subtype[0]]["expvar"]
                  for subtype in ds] for vx in vxt_vals] for ii in xx]
        )
)

sns.boxplot(x="vtx", y="expvar", data=df_vxt_res)
plt.savefig("linear_regression_vtx_var.png", dpi=300, bbox_inchex="tight")

# this one is to look at the performance measured when varying group pairings
df_gp_res = pd.DataFrame(
    dict(
        grouppair = ["kids_adults"]*3 + ["kids_adoles"]*3 + ["adols_adults"]*3,

        r2 = [ii for xx in [[global_results["voxel_thresh_{}".format(vx)][subtype[0]]["r2"] 
              for vx in vxt_vals] for subtype in ds] for ii in xx],

        expvar = [ii for xx in [[global_results["voxel_thresh_{}".format(vx)][subtype[0]]["expvar"]
                  for vx in vxt_vals] for subtype in ds] for ii in xx]
     )
)

sns.boxplot(x="grouppair", y="expvar", data=df_gp_res)
plt.savefig("linear_regression_group_pairs_var.png", dpi=300, bbox_inchex="tight")
