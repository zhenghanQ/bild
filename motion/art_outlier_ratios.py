# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:06:38 2016

@author: ysa
"""

import os
#import nibabel as nb
import glob 
#import matplotlib as plt
import re

'''
list of steps needed:
    1. get list of all bolds, 
    2. seperate per run
    3. get outliers
    4. seperate as above
    5. create ratios
    6. turn into text files
'''

# do we want to first define participants or just recursively get all participants?
data_dir_1 = '/om/user/ysa/bild/first_level/output_dir/model01/task001'
bold_qa_motion_files_1 = []
runs = [1,2,3,4,5,6]
for i in runs:
    bold_qa_motion_files_1.append(glob.glob(data_dir_1+'/BILDC*/qa/art/run0%1d_art.bold*' % i))
    
num_of_outliers_1 = []
for i in runs:
    for x in bold_qa_motion_files_1[i-1]:
        with open(x) as f:
            qa_file = f.readlines()
            num_of_outliers_1.append([len(qa_file), x])
            
runs_qa_1_t1 = [x for x in num_of_outliers_1 if 'run01' in x[1]]
runs_qa_2_t1 = [x for x in num_of_outliers_1 if 'run02' in x[1]]
runs_qa_3_t1 = [x for x in num_of_outliers_1 if 'run03' in x[1]]           

data_dir_2 = '/om/user/ysa/bild/first_level/output_dir/model01/task002'
bold_qa_motion_files_2 = []
for i in runs:
    bold_qa_motion_files_2.append(glob.glob(data_dir_2+'/BILDC*/qa/art/run0%1d_art.bold*' % i))
    
num_of_outliers_2 = []
for i in runs:
    for x in bold_qa_motion_files_2[i-1]:
        with open(x) as f:
            qa_file = f.readlines()
            num_of_outliers_2.append([len(qa_file), x])
            
runs_qa_1_t2 = [x for x in num_of_outliers_2 if 'run01' in x[1]]
runs_qa_2_t2 = [x for x in num_of_outliers_2 if 'run02' in x[1]]
runs_qa_3_t2 = [x for x in num_of_outliers_2 if 'run03' in x[1]] 

task_1_par_list = os.listdir(data_dir_1)
task_2_par_list = os.listdir(data_dir_2)

task_1_volumes = 42.0
task_2_volumes = 50.0

bold_percent_outliers_t1_r1 = []
bild = re.compile("BILDC[0-9]*")
for i in runs_qa_1_t1:
    if i[0] != 0:
        bold_percent_outliers_t1_r1.append([(i[0]/task_1_volumes), bild.findall(i[1])])
    else:
        bold_percent_outliers_t1_r1.append(['no outliers', bild.findall(i[1])])
    
    
bold_percent_outliers_t1_r2 = []
bild = re.compile("BILDC[0-9]*")
for i in runs_qa_2_t1:
    if i[0] != 0:
        bold_percent_outliers_t1_r2.append([(i[0]/task_1_volumes), bild.findall(i[1])])
    else:
        bold_percent_outliers_t1_r2.append(['no outliers', bild.findall(i[1])])

bold_percent_outliers_t1_r3 = []
bild = re.compile("BILDC[0-9]*")
for i in runs_qa_3_t1:
    if i[0] != 0:
        bold_percent_outliers_t1_r3.append([(i[0]/task_1_volumes), bild.findall(i[1])])
    else:
        bold_percent_outliers_t1_r3.append(['no outliers', bild.findall(i[1])])

bold_percent_outliers_t2_r1 = []
bild = re.compile("BILDC[0-9]*")
for i in runs_qa_1_t2:
    if i[0] != 0:
        bold_percent_outliers_t2_r1.append([(i[0]/task_2_volumes), bild.findall(i[1])])
    else:
        bold_percent_outliers_t2_r1.append(['no outliers', bild.findall(i[1])])
        
bold_percent_outliers_t2_r2 = []
bild = re.compile("BILDC[0-9]*")
for i in runs_qa_2_t2:
    if i[0] != 0:
        bold_percent_outliers_t2_r2.append([(i[0]/task_2_volumes), bild.findall(i[1])])
    else:
        bold_percent_outliers_t2_r2.append(['no outliers', bild.findall(i[1])]) 
        
bold_percent_outliers_t2_r3 = []
bild = re.compile("BILDC[0-9]*")
for i in runs_qa_3_t2:
    if i[0] != 0:
        bold_percent_outliers_t2_r3.append([(i[0]/task_2_volumes), bild.findall(i[1])])
    else:
        bold_percent_outliers_t2_r3.append(['no outliers', bild.findall(i[1])])