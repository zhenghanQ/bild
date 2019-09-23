import numpy as np
import pandas as pd
import os

t2vol = 50.0
path='/om/project/bild/Analysis/task_openfmri/first_level/output_dir_071616/model01/task002/'
pars = os.listdir(path)
runs = ['run01', 'run02', 'run03']
out_percent_mat = []

def check_all_runs(path, par):
    x_list = []
    for run in ['run01','run02','run03']:
        x_list.append(os.path.isfile(os.path.join(path,
                                  par,'qa/art/',
                                  run+'_art.bold_dtype_mcf_outliers.txt')))
    if np.all(x_list):
        return True
    else:
        return False
    
def num_trs(path, par, run,task):
    f = np.genfromtxt(os.path.join(path,
                                   par,'qa/art/',
                                   run+'_art.bold_dtype_mcf_outliers.txt'),
                     dtype=int)
    try:
        flen_ = len(f)
    except TypeError:
        flen_ = 1.
    flen = flen_ / task
    return flen

par_list_to_use = []
for i in pars:
    if check_all_runs(path, i):
        par_list_to_use.append(i)
print(len(par_list_to_use))  
for par in par_list_to_use:
    for run in runs:
        out_percent_mat.append(num_trs(path,
                                      par,
                                      run,
                                      t2vol))      
to_use = np.array(out_percent_mat)
outlier_matrix = np.reshape(to_use, (234,3))
outlier_df = pd.DataFrame(data=outlier_matrix, columns=['run01','run02','run03'])
outlier_df['id'] = par_list_to_use     
pfa2 = []
pfa3 = []
for i in outlier_df.index:
    if np.all(outlier_df.ix[i:i,:-1] < .2):
        pfa2.append(outlier_df.ix[i:i,-1].values[0])
    if np.all(outlier_df.ix[i:i,:-1] < .3):
        pfa3.append(outlier_df.ix[i:i,-1].values[0])
pfa2df = pd.DataFrame(data=pfa2, columns=['id'])
pfa3df = pd.DataFrame(data=pfa3, columns=['id']) 
pfa2df.to_csv('lessthan20%motiontask2.csv', index=None)
pfa3df.to_csv('lessthan30%motiontask2.csv', index=None)
        
