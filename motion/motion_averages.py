# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:59:06 2016

@author: ysa
"""

import pandas as pd
import numpy as np
path_to_ASD='/om/user/ysa/bild/data/task_movement_outliers/task002_ASD_motion.csv'
path_to_TD='/om/user/ysa/bild/data/task_movement_outliers/task002_TD_motion.csv'
path_to_DYS='/om/user/ysa/bild/data/task_movement_outliers/task002_DSY_motion.csv'
groups=[[],[],[]]
group_numbers=[('ASD',36),('TD',75),('DYS',27)]
ASD_dataFrame=pd.read_csv(path_to_ASD, index_col=0)
TD_dataFrame=pd.read_csv(path_to_TD, index_col=0)
DYS_dataFrame=pd.read_csv(path_to_DYS, index_col=0)
dFrames = [ASD_dataFrame, TD_dataFrame, DYS_dataFrame]
for x in range(3):
    for i in range(group_numbers[x][1]):
        groups[x].append((dFrames[x]['r1t2'][i]+dFrames[x]['r2t2'][i]+dFrames[x]['r3t2'][i])/3)
df_ASD = np.array(groups[0])
df_TD = np.array(groups[1])
df_DYS = np.array(groups[2])
group_averages=[df_ASD,df_TD,df_DYS]
for i in range(3):
    dFrames[i]['averages']=group_averages[i]
    dFrames[i].to_csv('%s_task002_motion_averaged' % group_numbers[i][0])


