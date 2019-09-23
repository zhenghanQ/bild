
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# In[81]:


os.chdir('/om/user/cdla/projects/bild/openfmri_4')


# In[89]:


#pars_dir = '/om/user/cdla/projects/bild/openfmri_4/032316_l1output_fwhm6_art3mm_sd/model01/task001'
pars_dir = "/om/project/bild/Analysis/task_openfmri/first_level/output_dir_071616/model01/task001"

#pars = 'BILDC3161 BILDC3139 BILDC3093 BILDC3111 BILDC3223 BILDC3072 BILDC3267 BILDC3083 BILDC3236 BILDC3138 BILDC3237 BILDC3095 BILDC3152 BILDC3296 BILDC3076 BILDC3092 BILDC3089 BILDC3224 BILDC3219 BILDC3074 BILDC3220 BILDC3141 BILDC3165 BILDC3238 BILDC3071 BILDC3143 BILDC3142 BILDC3187 BILDC3213 BILDC3078 BILDC3151 BILDC3167 BILDC3240 BILDC3049 BILDC3159 BILDC3277 BILDC3278 BILDC3130 BILDC3046 BILDC3125 BILDC3266 BILDC3081 BILDC3158 BILDC3205 BILDC3268 BILDC3247 BILDC3244 BILDC3198 BILDC3168 BILDC3189 BILDC3255 BILDC3250 BILDC3119 BILDC3254 BILDA17 BILDA_1 BILDA21 BILDA24 BILDA4 BILDA2 BILDA9 BILDA19 BILDSLI3054 BILDA22 BILDSLI3053 BILDA3 BILDA20 BILDSLI3056 BILDA14 BILDA10 BILDA11 BILDA12 BILDSLI3057 BILDA16 BILDSLI3048 BILDA13'.split(' ')

pars = "BILDC3050 BILDC3144 BILDC3181 BILDC3182 BILDC3263 BILDC3264 BILDC3294 BILDC3295 BILDC3307 BILDC3308 BILDC3311 BILDC3314 BILDC3331 BILDC3332 BILDC3335 BILDC3336".split(" ")

def compute_avg(par, l1dir):
    avg_files = []
    for i in [1,2,3]:
        avg_files.append(np.genfromtxt(os.path.join(l1dir,
                par, 'qa/art/run0{}_norm.bold_dtype_mcf.txt'.format(i))))
    avg_matrix = np.asarray(avg_files)
    avgs_runs = avg_matrix.T.mean(0)
    avg_all = sum(avgs_runs) / 3.0
    return [par, avg_all, avgs_runs]


# In[90]:


store_df = []
for par in pars:  
    avg_data = compute_avg(par,pars_dir)
    data = [avg_data[0], avg_data[1],
                     avg_data[2][0], avg_data[2][1],
                    avg_data[2][2]]
    store_df.append(data)   


# In[91]:


df = pd.DataFrame(data=store_df, columns=['id','overall_avg',
                                         'avg_run01','avg_run02',
                                         'avg_run03'])


# In[92]:

os.chdir("/om/project/bild/Analysis/task_openfmri/scripts/motion/motionfiles")
df.to_csv('task01_motion_norm_avg_valset.csv', sep=',', index=None)

