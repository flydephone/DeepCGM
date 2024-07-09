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

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import datetime
import time
from models_aux.MyDataset import MyDataSet
from models_aux.MC_base_prior13_01 import DEEPORYZA
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
if __name__ == "__main__":
    # %%load base data
    tra_year = "2018"
    cali = "2018"
    model_type = "MC_base_prior13_01_convergence_obs_scratch"
    scal_type="nor"
    obs_mask_name = []
    obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    obs_used_name = [name for name in obs_name if name not in obs_mask_name]
    # obs_name = ['WLV']
    units = ['-',"m$^2$/m$^2$","kg/ha","kg/ha","kg/ha","kg/ha","kg/ha"]
    sample_2018, sample_2019 = 65,40
    use_pretrained = False
    
    rea_res_dataset,rea_par_dataset,rea_wea_fer_dataset,rea_obs_dataset,rea_fit_dataset = utils.dataset_loader(data_source="format_dataset/%s/real_%s"%(scal_type,tra_year))
      
    model_list = os.listdir("model_weight/%s/"%model_type)
    model_list = [tpt for tpt in model_list if tra_year in tpt]


    max_min,mean_std,par_col_name,res_col_name = utils.base_dataset_loader()
    res_col_name = ['TIME','DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    # %%obs loc
    obs_loc = [res_col_name.index(name) for name in obs_name]
    #%% import dataset-raw
    res_max,res_min,par_max,par_min,wea_fer_max,wea_fer_min = max_min
    res_mean,res_std,par_mean,par_std,wea_fer_mean,wea_fer_std = mean_std
    
    
    #%% super parameter
    wea_fer_dim = np.shape((utils.pickle_load(rea_wea_fer_dataset[0])))[1]-1-3
    cro_dim = np.shape(utils.pickle_load(rea_res_dataset[0]))[1]
    obs_dim = len(obs_name)
    
    input_dim = wea_fer_dim
    output_dim = obs_dim
    init_dim = obs_dim
        
    #%% generate dataset
    if tra_year == "2018":
        tra_res_dataset,tra_wea_fer_dataset,tra_obs_dataset,tra_fit_dataset = rea_res_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_obs_dataset[:sample_2018],rea_fit_dataset[:sample_2018]
        tes_res_dataset,tes_wea_fer_dataset,tes_obs_dataset,tes_fit_dataset = rea_res_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_obs_dataset[sample_2018:],rea_fit_dataset[sample_2018:]
    elif tra_year == "2019":
        tes_res_dataset,tes_wea_fer_dataset,tes_obs_dataset,tes_fit_dataset = rea_res_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_obs_dataset[:sample_2018],rea_fit_dataset[:sample_2018]
        tra_res_dataset,tra_wea_fer_dataset,tra_obs_dataset,tra_fit_dataset = rea_res_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_obs_dataset[sample_2018:],rea_fit_dataset[sample_2018:]

    batch_size = 128
    tra_set = MyDataSet(obs_loc=obs_loc, res=tra_res_dataset, wea_fer=tra_wea_fer_dataset, obs=tra_obs_dataset, fit=tra_fit_dataset, max_min=max_min, mean_std=mean_std, scal_type=scal_type, batch_size=batch_size, aug=False)
    tra_DataLoader = DataLoader(tra_set, batch_size=batch_size, shuffle=False)
    tes_set = MyDataSet(obs_loc=obs_loc, res=tes_res_dataset, wea_fer=tes_wea_fer_dataset, obs=tes_obs_dataset, fit=tes_fit_dataset, max_min=max_min, mean_std=mean_std, scal_type=scal_type, batch_size=batch_size, aug=False)
    tes_DataLoader = DataLoader(tes_set, batch_size=batch_size, shuffle=False)

    # %% creat instances from class_LSTM
    hidden_dim_dvs = 64
    layer_dim_dvs = 1
    
    np_pre_models = []
    for model_index in range(25):
        model = model_list[model_index]
        model_path = 'model_weight/%s/%s'%(model_type,model)
        tra_loss = []
        tes_loss = []
        trained_model_names = os.listdir(model_path)
        for tpt in trained_model_names[:]:
            tra_loss += [float(tpt[:-4].split("_")[-5])]
            tes_loss += [float(tpt[:-4].split("_")[-1])]
        loss = np.array([tra_loss,tes_loss]).T
        min_indices = np.argmin(loss[:,0], axis=0)
        trained_model_name = trained_model_names[min_indices]
    

        
        # dvs super parameter  
        model = DEEPORYZA(input_dim = input_dim, hidden_dim = hidden_dim_dvs,layer_dim = layer_dim_dvs, output_dim = output_dim)   
        model.to(device) 
        model_to_load = torch.load(os.path.join(model_path,trained_model_name))
        model.load_state_dict(model_to_load,strict=True)  
    
        #%% -----------------------------------fit------------------------------------
        
        mode = "tes"
        for n,(x,y,o,f) in enumerate(tes_DataLoader):
            var_x, var_y, var_o, var_f = x.to(device), y.to(device), o.to(device), f.to(device)
            mask_res = var_y.ne(-10000)
            mask_obs = var_o.ne(-10000)
            var_x = var_x.requires_grad_(True)
            var_out_all, aux_all = model(var_x[:,:,[1,2,3,7,8]],var_y)
            np_wea_fer = utils.unscalling(scal_type,utils.to_np(var_x),wea_fer_max,wea_fer_min,wea_fer_mean,wea_fer_std)
            np_res = utils.unscalling(scal_type,utils.to_np(var_y),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_pre = utils.unscalling(scal_type,utils.to_np(var_out_all),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_obs = utils.unscalling(scal_type,utils.to_np(var_o),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_fit = utils.unscalling(scal_type,utils.to_np(var_f),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_mask = utils.to_np(mask_obs)         
    
        np_pre_models.append(np_pre)
        # # %%
    rmse_values_vars_models = []
    for np_pre in np_pre_models:
        data = np.stack([np_pre,np_res,np_obs,np_mask])
        prediction  = data[0]
        res = data[1]
        observation = data[2]
        mask = data[3]
        
        # 定义计算 RMSE 的函数
        def calculate_rmse(pred, obs, mask):
            masked_pred = pred[mask == 1]
            masked_obs = obs[mask == 1]
            return np.sqrt(np.mean((masked_pred - masked_obs)**2))
        
        
        # 分段界限
        bins = [0.0, 0.65, 1.0, 1.4, 2.3]
        bins_phe = ["出苗期","拔节初期","扬花期", "成熟期", "衰亡期"]
        
        rmse_values_vars = []
        mask_segment_vars = []
        rmse_res_values_vars = []
        sample_counts_vars = []
        for j in range(1, 7):  # 遍历最后一维的变量（从第二个开始）
            rmse_values = [] 
            rmse_res_values = [] 
            sample_counts = []  # 存储每个区间的样本数量
            
            
            for i in range(len(bins) - 1):
                lower_bound = max(bins[i],0.01)
                upper_bound = bins[i + 1]
                
                
                
                mask_segment = np.logical_and(prediction[..., 0] >= lower_bound, prediction[..., 0] < upper_bound)
                        
                # 计算并存储具有观测值的样本数量
                sample_count = np.sum(mask[..., j][mask_segment])
                sample_counts.append(sample_count)

                
                rmse = calculate_rmse(prediction[..., j][mask_segment], observation[..., j][mask_segment], mask[..., j][mask_segment])
                rmse_values.append(rmse)

                
                rmse_res = calculate_rmse(res[..., j][mask_segment], observation[..., j][mask_segment], mask[..., j][mask_segment])
                rmse_res_values.append(rmse_res)
 
            mask_segment_vars.append(np.array(mask_segment))
            rmse_res_values_vars.append(np.array(rmse_res_values))
            sample_counts_vars.append(np.array(sample_counts))       
 
            rmse_values_vars.append(np.array(rmse_values))
        rmse_values_vars = np.array(rmse_values_vars)
        rmse_values_vars_models.append(rmse_values_vars)
        
    rmse_values_vars_models = np.array(rmse_values_vars_models)
    mask_segment_vars = np.array(mask_segment_vars)
    rmse_res_values_vars = np.array(rmse_res_values_vars)
    sample_counts_vars = np.array(sample_counts_vars)
    rmse_values_vars_models_ave = np.mean(rmse_values_vars_models,0)
    rmse_values_vars_models_std = np.std(rmse_values_vars_models,0)
        
    units = ["m²/m²", "kg/ha", "kg/ha", "kg/ha", "kg/ha", "kg/ha"]
    variables = ["DVS", "PAI", "WLV", "WST", "WSO", "WAGT", "YIELD"]
    x_pos = np.arange(len(bins))*2
    width = 0.35
    # 创建一个 2x3 的子图网格
    nrows = 1
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols, dpi=300, figsize=(12, 2.4))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.8,
                        top=0.9,
                        wspace=0.25,
                        hspace=0.1)
    for j in range(1, 7):
        row = (j - 1) // ncols
        col = (j - 1) % ncols
        if nrows==1 or ncols==1:    
            ax = axes[row*ncols + col]
        else:
            ax = axes[row,col]
        
        # x_labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
        x_labels_mid = [i*2+0.5 for i in range(len(bins) - 1)]
        x_labels_tra = [i*2+0.1 for i in range(len(bins) - 1)]
        x_labels_tes = [i*2+0.9 for i in range(len(bins) - 1)]
        
        # ax.text(0.4, 0.98, variables[j], transform=ax.transAxes, va='top', ha='left',fontsize=20)
    
    
        # ax.bar(x_labels_tes, sample_counts_vars[j-1], alpha=0.4, color='b')  # 2019数据集
        # ax.bar(x_labels_tra, [x for x in sample_counts_2019], alpha=0.4, color='r')  # 2018数据集
        
        # ax = ax.twinx()
        ax.plot(x_labels_mid, [x for x in rmse_values_vars_models_ave[j-1]], alpha=1, linewidth=3,color='green')  # 2018模型
        if len([1 for tpt in rmse_values_vars_models_ave[j-1] if tpt>=0])==1:
            ax.scatter(x_labels_mid, [x for x in rmse_values_vars_models_ave[j-1]],marker="s",linewidths=0.2,color='green')
            
        ax.fill_between(x_labels_mid, rmse_values_vars_models_ave[j-1]-rmse_values_vars_models_std[j-1], rmse_values_vars_models_ave[j-1]+rmse_values_vars_models_std[j-1], alpha=0.4, linewidth=3,color='green')  # 2018模型
        # if len([1 for tpt in rmse_values_vars_models_ave[j-1] if tpt>=0])==1:
        #     ax.scatter(x_labels_mid, [x for x in rmse_values_vars_models_ave[j-1]],marker="s",linewidths=0.2,color='green')
              
        ax.plot(x_labels_mid, [x for x in rmse_res_values_vars[j-1]], alpha=1, linewidth=3,color='black')  # 2018模型
        if len([1 for tpt in rmse_res_values_vars[j-1] if tpt>=0])==1:
            ax.scatter(x_labels_mid, [x for x in rmse_res_values_vars[j-1]],marker="s",linewidths=0.2,color='black')
        
        
        y_min, y_max = ax.get_ylim()
        y_lim = max(abs(y_min),abs(y_max)*1.1)
        ax.set_ylim(0,y_lim)
       
        if col == 0:
            ax.set_ylabel('RMSE', rotation=90)
        ax.set_yticks(ax.get_yticks())
        if j==1:
            ax.set_yticklabels(["%.1f"%(abs(x)) for x in ax.get_yticks()], rotation=90)  # 整数标签
        else:
            ax.set_yticklabels([int(abs(x)) for x in ax.get_yticks()], rotation=90)  # 整数标签

        ax.set_xticks(x_pos -0.5)
        ax.set_xticklabels(bins_phe, rotation=90)
    # plt.tight_layout()
    plt.show() 
                
            
                
    # if col == 0:
    #     ax.set_ylabel(f'RMSE\n({units[j-1]})')
    #     for label in ax.get_yticklabels():
    #         label.set_rotation(90)
    # else:
    #     ax.set_yticklabels([])
    
