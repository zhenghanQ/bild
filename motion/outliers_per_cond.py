# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:11:22 2016

@author: ysa
"""
import os
import glob
import pandas as pd
base_dir = '/om/user/ysa/bild/data'
l1_dir = '/om/user/ysa/bild/data/first_level/output_dir/model01/task001'
l1_par = os.listdir(l1_dir)
outlier_files = sorted(glob.glob(l1_dir+'/BILDC*/qa/art/run0*_art.bold_dtype_mcf_outliers*'))
cond_files = []
for i in range(1,4):
    cond_files.append(sorted(glob.glob(base_dir+'/BILDC*/model/model001/onsets/task001_run00%1d/*' % i)))
cond_tr = {}
for i in l1_par:
    cond_tr[i] = []
for i in range(3):    
    for k,v in cond_tr.items():
        for j in cond_files[i]:
            if k in j:
                v.append(j)
tr_outliers_from_conds = {} # contains onset - 2 / 6
for i in l1_par:
    tr_outliers_from_conds[i] = []
for i in range(12):
    for k,v in cond_tr.items():
        for h,j in tr_outliers_from_conds.items():
            if k == h and len(v) == 12:
                f = pd.read_csv(v[i], sep='\t', names=['time','duration','not_sure'])
                times = ((f['time']-2)/6)+2
                t = 0
                while t != 12 and len(j) != 12:
                    j.append([])
                    t += 1
                for w in times:
                    j[i].append(w)
for x in range(4):                    
    for k,v in tr_outliers_from_conds.items():
        if len(v) > 0:
            for i in v[x]:
                v[x].insert(v[x].index(i), v[x].pop(v[x].index(i))*10)
for x in range(4):                    
    for k,v in tr_outliers_from_conds.items():
        if len(v) > 0:
            for i in v[x+4]:
                v[x+4].insert(v[x+4].index(i), v[x+4].pop(v[x+4].index(i))*10000) 
for x in range(4):                    
    for k,v in tr_outliers_from_conds.items():
        if len(v) > 0:
            for i in v[x+8]:
                v[x+8].insert(v[x+8].index(i), v[x+8].pop(v[x+8].index(i))*100000000)                  
# multiplying groups of 4 in key:value by some unique identifier
# run 1 * 10, run 2 * 10000, run 3 * 100000000
par_outlier = {}
for i in l1_par:
    par_outlier[i] = []
for x in outlier_files:
    for k,v in par_outlier.items():
        if k in x:
            v.append(x)
tr_outliers_from_qa = {} # contains outliers from art
for i in l1_par:
    tr_outliers_from_qa[i] = []
for i in range(3):
    for k,v in par_outlier.items():
        for h,j in tr_outliers_from_qa.items():
            if k == h and len(v) == 3:
                if i == 0:
                    f = pd.read_csv(v[i], names=['outlier'])
                    trs = f['outlier'] * 10
                    for w in trs:
                        j.append(w)
                elif i == 1:
                    f = pd.read_csv(v[i], names=['outlier'])
                    trs = f['outlier'] * 10000
                    for w in trs:
                        j.append(w)
                elif i == 2:
                    f = pd.read_csv(v[i], names=['outlier'])
                    trs = f['outlier'] * 100000000
                    for w in trs:
                        j.append(w)
                       
big_list = []                    
for k,v in tr_outliers_from_qa.items():
    for x,y in enumerate(tr_outliers_from_conds[k]):
        for j in v:
            if j in y:
                big_list.append(' '.join([str(j),str(x),k])) # j = TR estimate, x = run_cond, k = par
big_list = sorted(big_list)
order_cs_rs = {'run001_cond001':0,'run001_cond002':0,'run001_cond003':0,
               'run001_cond004':0,'run002_cond001':0,'run002_cond002':0,
               'run002_cond003':0,'run002_cond004':0,'run003_cond001':0,
               'run003_cond002':0,'run003_cond003':0,'run003_cond004':0}
for x,y in enumerate(sorted(order_cs_rs)):
    for i in big_list:
        if str(' ' + str(x) + ' ') in i:
            order_cs_rs[y] += 1
outlier_per_par_cond = {x:[] for x in l1_par}
for i in big_list:
    for k,v in outlier_per_par_cond.items():
        if i[-9:] == k:
            v.append(i)           
ordered_dict = {x:[['run001_cond001',0],['run001_cond002',0],['run001_cond003',0],
               ['run001_cond004',0],['run002_cond001',0],['run002_cond002',0],
               ['run002_cond003',0],['run002_cond004',0],['run003_cond001',0],
               ['run003_cond002',0],['run003_cond003',0],['run003_cond004',0]] for x in l1_par}
for i in range(12):
    for k,v in outlier_per_par_cond.items():
        for x,y in ordered_dict.items():
            if k == x:
                for n in v:
                    if str(' ' + str(i) + ' ') in n:
                        y[i][1] += 1
                        
f = open('of1.csv','w')
f.write('ids,r1c1,r1c2,r1c3,r1c4,r2c1,r2c2,r2c3,r2c4,r3c1,r3c2,r3c3,r3c4\n')                       
for k,v in ordered_dict.items():
    f.write(str(k+','+str(v[0][1])+','+str(v[1][1])+','+str(v[2][1])+','+str(v[3][1])+','+str(v[4][1])+','+str(v[5][1])+','+str(v[6][1])+','+str(v[7][1])+','+str(v[8][1])+','+str(v[9][1])+','+str(v[10][1])+','+str(v[11][1])+'\n'))
f.close()    