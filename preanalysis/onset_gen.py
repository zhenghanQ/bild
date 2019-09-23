
# coding: utf-8

# In[1]:


import os
from glob import glob
import csv


# In[2]:


openfmri_dir='/om/project/bild/Analysis/task_openfmri/openfmri'
subjs = next(os.walk(openfmri_dir))[1]


# In[3]:


for i in subjs:
    if not 'BILD' in i:
        subjs.pop(subjs.index(i))


# In[5]:


#####phono task
for run in range(3):
    onsets = [[[38, 44, 98, 104, 122, 128, 194, 200],[50, 56, 110, 116, 158, 164, 206, 212],[2, 8, 74, 80, 146, 152, 230, 236],[14, 20, 62, 68, 170, 176, 182, 188]],
               [[26, 32, 74, 80, 170, 176, 194, 200],[14, 20, 110, 116, 134, 140, 230, 236],[2, 8, 86, 92, 122, 128, 206, 212],[50, 56, 98, 104, 146, 152, 182, 188]],
               [[26, 32, 62, 68, 134, 140, 194, 200],[50, 56, 86, 92, 158, 164, 230, 236],[38, 44, 110, 116, 146, 152, 182, 188],[2, 8, 74, 80, 122, 128, 218, 224]]]
    for sub in subjs:
        onset_dir='/om/project/bild/Analysis/task_openfmri/openfmri/%s/model/model001/onsets/task001_run%03d'%(sub,run+1)
        if not os.path.exists(onset_dir):
            os.makedirs(onset_dir)
            
        for indx, condition in enumerate(['2_syl','3_syl','4_syl','5_syl']):
            with open(os.path.join(onset_dir,'cond%03d.txt'%(indx+1)),'wb') as csvfile:
                writer=csv.writer(csvfile, delimiter='\t')
                for onset in onsets[run][indx]:
                    writer.writerow(['%.1f'%onset, '%.1f'%4, '%.1f'%1])


# In[6]:


#### syntax task
for run in range(3):
    onsets=[[[20, 44, 62, 74, 98, 122, 152, 170, 194, 230, 260, 272],[2, 32, 68, 86, 110, 140, 158, 182, 212, 236, 254, 284],[8, 38, 56, 80, 104, 128, 164, 188, 206, 224, 248, 266]],
        [[20, 44, 56, 74, 110, 140, 164, 182, 194, 224, 260, 284],[8, 26, 50, 86, 116, 134, 146, 188, 206, 218, 242, 278],[2, 38, 68, 80, 98, 122, 152, 170, 200, 236, 254, 266]],
        [[8, 44, 56, 80, 98, 122, 146, 182, 212, 230, 248, 266],[2, 26, 68, 92, 104, 128, 152, 188, 206, 224, 242, 284],[14, 38, 62, 74, 110, 140, 158, 170, 200, 236, 260, 272]]]
    for sub in subjs:
        onset_dir='/om/project/bild/Analysis/task_openfmri/openfmri/%s/model/model001/onsets/task002_run%03d'%(sub,run+1)
        if not os.path.exists(onset_dir):
            os.makedirs(onset_dir)
        
        for idx,condition in enumerate(['GE','OI','C']):
            with open(os.path.join(onset_dir,'cond%03d.txt'%(idx+1)),'wb') as csvfile:
                writer=csv.writer(csvfile, delimiter='\t')
                for onset in onsets[run][idx]:
                    writer.writerow(['%.1f'%onset, '%.1f'%4, '%.1f'%1])


# In[4]:


subjs

