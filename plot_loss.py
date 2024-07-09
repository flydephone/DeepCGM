# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:56:07 2020
1、在原有基础上加入人工激活层，保证基本物理规律（物候正增长，总干物质正增长，各器官质量守恒）
2、参数作为输入，不作为隐藏状态
@author: hanjingye
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import math
import pickle
import random
import copy



import datetime
import time

import utils



   
if __name__ == "__main__":
    # %%load base data
    tra_year = "2018"
    cali = ""
    model_type = "Naive_LSTM_cum_fit_scratch"
    scal_type="nor"
    obs_mask_name = []
    obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    obs_used_name = [name for name in obs_name if name not in obs_mask_name]
    # obs_name = ['WLV']
    units = ['-',"m$^2$/m$^2$","kg/ha","kg/ha","kg/ha","kg/ha","kg/ha"]
    sample_2018, sample_2019 = 65,40
    use_pretrained = False

    model_list = os.listdir("model_weight/%s/"%model_type)
    if cali=="uncali":
        model_list = [tpt for tpt in model_list if tra_year in tpt and "uncali" in tpt]
    else:
        model_list = [tpt for tpt in model_list if tra_year in tpt and "uncali" not in tpt]

    loss = []
    
    for i,model in enumerate(model_list[:1]):
        tra_loss = []
        val_loss = []
        tes_loss = []
        model_path = 'model_weight/%s/%s'%(model_type,model)
        trained_model_names = os.listdir(model_path)
        for tpt in trained_model_names[:700]:
            tra_loss += [float(tpt[:-4].split("_")[-5])]
            # val_loss += [float(tpt[:-4].split("_")[-3])]
            tes_loss += [float(tpt[:-4].split("_")[-1])]
    
        losses = np.array([tra_loss,tes_loss]).T
        fig = plt.figure(dpi = 300)
        plt.plot(losses[:,:],label=["tra","tes"])
        plt.axhline(y=0.035, color='black', linestyle='--')
        plt.axhline(y=0.020, color='black', linestyle='--')
        # plt.text("%s"%chr(97 + tpt_loc-1))
        plt.ylim(0.010,0.08)
        # plt.title("%02d"%i)
        plt.xlabel("Training epoch",fontsize=16)
        plt.ylabel("Fitting loss",fontsize=16)
        # plt.savefig(os.path.join("figure",'%s_%s_%d.png'%(model_type,tra_year,i)), bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
        loss.append(losses)
    loss = np.array(loss)
    min_indices = np.argmin(loss[:,:,0], axis=1)
    result = np.array([loss[i, min_indices[i], 1] for i in range(len(loss))])
    a_loss_sample = result
    ave_loss = result.mean()
    a_std_loss = result.std()