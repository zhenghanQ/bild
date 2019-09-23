import os
import utils
import prelim
import models
import numpy as np
import pandas as pd

info_dict = dict(
    base_path = "/Users/yoelsanchezaraujo/Documents/bild_stuff", 
    voxels="fszstatcope2_nvoxels_nz.csv",
    behav="behav.txt",
    data="fszstatcope2_means_nz.csv"
)

from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import LeaveOneOut, KFold

global_results = {}

vxt_vals = [27, 116, 127]

ds = [
    ("kids_adults", "kids_adults_behav"),
    ("kids_adoles", "kids_adoles_behav"),
    ("adols_adults", "adols_adults_behav")
]

data_sets = prelim.load_data_and_preproc(info_dict, vxt_vals[0])

# within group test kids only
kid_idx = data_sets["behav"].young_kid.values.astype(bool)
kid_roi = data_sets["roi_data"][kid_idx]
kid_behav = data_sets["behav"][kid_idx]
kX = kid_roi.values
ky = utils.rau_transform(kid_behav["total_acc"].values)

kclf = Pipeline([
    ("sc", StandardScaler()),
    ("lr", linear_model.RidgeCV(cv=KFold(n_splits=4)))
])

kxpred = np.zeros(kX.shape[0])

for idx, (train, test) in enumerate(LeaveOneOut().split(kX, ky)):
    kclf.fit(kX[train], ky[train])
    kxpred[idx] = kclf.predict(kX[test])

kr2 = explained_variance_score(ky, kxpred)

#plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))
cvar = kid_behav["age"].values
plt.scatter(
        kid_behav["total_acc"].values, 
        ky, 
        c=cvar,
        cmap=plt.get_cmap("Blues"),
        edgecolor="black"
)

cb = plt.colorbar()
plt.xlabel("total_acc")
plt.ylabel("rau_transform(total_acc)")
plt.show()

# within group adol
adol_idx = data_sets["behav"].old_kid.values.astype(bool)
adol_roi = data_sets["roi_data"][adol_idx]
adol_behav = data_sets["behav"][adol_idx]
adol_X = adol_roi.values
adol_y = adol_behav["total_acc"].values

adol_clf = Pipeline([
    ("sc", StandardScaler()),
    ("lr", linear_model.RidgeCV(cv=KFold(n_splits=4)))
])

adol_xpred = np.zeros(adol_X.shape[0])

for idx, (train, test) in enumerate(LeaveOneOut().split(adol_X, adol_y)):
    adol_clf.fit(adol_X[train], adol_y[train])
    adol_xpred[idx] = adol_clf.predict(adol_X[test])

adol_r2 = explained_variance_score(adol_y, adol_xpred)


# within group adult
adul_idx = data_sets["behav"].adult.values.astype(bool)
adul_roi = data_sets["roi_data"][adul_idx]
adul_behav = data_sets["behav"][adul_idx]
adul_X = adul_roi.values
adul_y = adul_behav["total_acc"].values

adul_clf = Pipeline([
    ("sc", StandardScaler()),
    ("lr", linear_model.RidgeCV(cv=KFold(n_splits=4)))
])

adul_xpred = np.zeros(adul_X.shape[0])

for idx, (train, test) in enumerate(LeaveOneOut().split(adul_X, adul_y)):
    adul_clf.fit(adul_X[train], adul_y[train])
    adul_xpred[idx] = adul_clf.predict(adul_X[test])

adul_r2 = explained_variance_score(adul_y, adul_xpred)


# between dataset test
data_sets = prelim.load_data_and_preproc(info_dict, vxt_vals[0])
use_data = data_sets[ds[0][0]]
use_behav = data_sets[ds[0][1]]

X = use_data.values
y = use_behav["5syl_acc"].values

clf = Pipeline([
    ("sc", StandardScaler()),
    ("lr", linear_model.RidgeCV(cv=KFold(n_splits=4)))
])

xpred = np.zeros(X.shape[0])
gofs = np.zeros(X.shape[0])

for idx, (train, test) in enumerate(LeaveOneOut().split(X, y)):
    clf.fit(X[train], y[train])
    xpred[idx] = clf.predict(X[test])
    gofs[idx] = explained_variance_score(y[test], clf.predict(X[test]))

print("r2_score: {}".format(r2_score(y, xpred)))
print("exp_var_score: {}".format(explained_variance_score(y, xpred)))

