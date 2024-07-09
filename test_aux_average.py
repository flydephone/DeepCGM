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
from models_aux.MC_base_prior13 import DEEPORYZA
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    # %%load base data
    tra_year = "2019"
    cali = "2018"
    model_type = "MC_base_prior13_fit_scratch"
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


    r2_list, rmse_list,np_pre_list = [], [], []
    for model in model_list[:1]:
        model_path = 'model_weight/%s/%s'%(model_type,model)

        # trained_model_name = os.listdir(model_path)[epoch]
        tra_loss = []
        tes_loss = []
        trained_model_names = os.listdir(model_path)
        for tpt in trained_model_names[:]:
            tra_loss += [float(tpt[:-4].split("_")[-5])]
            tes_loss += [float(tpt[:-4].split("_")[-1])]
        loss = np.array([tra_loss,tes_loss]).T
        min_indices = np.argmin(loss[:,0], axis=0)
        # trained_model_name = trained_model_names[-1]
        trained_model_name = trained_model_names[min_indices]
       
        # model_path = 'E:/pytorch/model_wtight/my_model/ORYZA_LSTM'
        # trained_model_name = "2022-07-08 05-09-51_uniform_structure_two_model3_2018_2019.pkl"
        rea_res_dataset,rea_par_dataset,rea_wea_fer_dataset,rea_obs_dataset,rea_fit_dataset = utils.dataset_loader(data_source="format_dataset/%s/real_%s"%(scal_type,tra_year))
      
        if tra_year == "2018":
            tra_res_dataset,tra_wea_fer_dataset,tra_obs_dataset,tra_fit_dataset = rea_res_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_obs_dataset[:sample_2018],rea_fit_dataset[:sample_2018]
            tes_res_dataset,tes_wea_fer_dataset,tes_obs_dataset,tes_fit_dataset = rea_res_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_obs_dataset[sample_2018:],rea_fit_dataset[sample_2018:]
        elif tra_year == "2019":
            tes_res_dataset,tes_wea_fer_dataset,tes_obs_dataset,tes_fit_dataset = rea_res_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_obs_dataset[:sample_2018],rea_fit_dataset[:sample_2018]
            tra_res_dataset,tra_wea_fer_dataset,tra_obs_dataset,tra_fit_dataset = rea_res_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_obs_dataset[sample_2018:],rea_fit_dataset[sample_2018:]
    
            
        max_min,mean_std,par_col_name,res_col_name = utils.base_dataset_loader()
        res_col_name = ['TIME','DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
        # %%obs loc
        obs_loc = [res_col_name.index(name) for name in obs_name]
        #%% import dataset-raw
        res_max,res_min,par_max,par_min,wea_fer_max,wea_fer_min = max_min
        res_mean,res_std,par_mean,par_std,wea_fer_mean,wea_fer_std = mean_std
        
        
        #%% super parameter
        wea_fer_dim = np.shape((utils.pickle_load(tra_wea_fer_dataset[0])))[1]-1-3
        cro_dim = np.shape(utils.pickle_load(tra_res_dataset[0]))[1]
        obs_dim = len(obs_name)
        
        input_dim = wea_fer_dim
        output_dim = obs_dim
        init_dim = obs_dim
            
        #%% generate dataset
        batch_size = 128
        tra_set = MyDataSet(obs_loc=obs_loc, res=tra_res_dataset, wea_fer=tra_wea_fer_dataset, obs=tra_obs_dataset, fit=tra_fit_dataset, max_min=max_min, mean_std=mean_std, scal_type=scal_type, batch_size=batch_size, aug=False)
        tra_DataLoader = DataLoader(tra_set, batch_size=batch_size, shuffle=False)
        tes_set = MyDataSet(obs_loc=obs_loc, res=tes_res_dataset, wea_fer=tes_wea_fer_dataset, obs=tes_obs_dataset, fit=tes_fit_dataset, max_min=max_min, mean_std=mean_std, scal_type=scal_type, batch_size=batch_size, aug=False)
        tes_DataLoader = DataLoader(tes_set, batch_size=batch_size, shuffle=False)
    
        # %% creat instances from class_LSTM
        hidden_dim_dvs = 64
        layer_dim_dvs = 1
        
        # dvs super parameter  
        model = DEEPORYZA(input_dim = input_dim, hidden_dim = hidden_dim_dvs,layer_dim = layer_dim_dvs, 
                            output_dim = output_dim)
        model_ref = DEEPORYZA(input_dim = input_dim, hidden_dim = hidden_dim_dvs,layer_dim = layer_dim_dvs, 
                            output_dim = output_dim)
        
        model.to(device) 
     
        model_to_load = torch.load(os.path.join(model_path,trained_model_name))
        model.load_state_dict(model_to_load,strict=True)  

        model_ref.to(device) 
        model_ref.load_state_dict(model_to_load,strict=True)  
        
        criterion = nn.MSELoss()
        def mask_mse(pred,real,mask):
            weights =  [1,1,5,2,2,1,2]
            mse_loss = nn.MSELoss(reduction='none')
            loss = mse_loss(pred, real)
            # loss_dvs = layer(pred[:,:-1,0]-pred[:,1:,0]-0.00000).mean()
            # loss_mask = loss.masked_select(mask).mean()
            loss_split_mask = [loss[:,:,i].masked_select(mask[:,:,i]).mean()*weights[i] for i in range(loss.shape[2])]
            # np_mask = mask.clone().data.cpu().numpy()
            return sum(loss_split_mask),torch.Tensor(loss_split_mask)
        #%% -----------------------------------fit------------------------------------
    
        np_wea_fer_batchs, np_res_batchs, np_pre_batchs, np_obs_batchs, np_fit_batchs = [],[],[],[], []
        mode = "tes"
        for n,(x,y,o,f) in enumerate(tes_DataLoader):
            var_x, var_y, var_o, var_f = x.to(device), y.to(device), o.to(device), f.to(device)
            mask_res = var_y.ne(-10000)
            var_x = var_x.requires_grad_(True)
            var_out_all, aux_all = model(var_x[:,:,[1,2,3,7,8]],var_y)
            np_wea_fer = utils.unscalling(scal_type,utils.to_np(var_x),wea_fer_max,wea_fer_min,wea_fer_mean,wea_fer_std)
            np_res = utils.unscalling(scal_type,utils.to_np(var_y),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_pre = utils.unscalling(scal_type,utils.to_np(var_out_all),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_obs = utils.unscalling(scal_type,utils.to_np(var_o),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_fit = utils.unscalling(scal_type,utils.to_np(var_f),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            
            # C_cell_all,A_cell_all,C_redistribution_mat_all,C_partitation_mat_all,C_consuming_mat_all,C_assimilate_ratio_all,C_partitation_mat_convergence_all,C_redistribution_mat_convergence_all,C_cell_convergence_all,C_consuming_all = aux_all
            # dims = [8,8,8,1]
            # dim_segment = np.cumsum([0]+dims)
            # lea = C_cell_all[:,:,dim_segment[0]:dim_segment[1]].sum(-1)
            # ste = C_cell_all[:,:,dim_segment[1]:dim_segment[2]].sum(-1)
            # gra = C_cell_all[:,:,dim_segment[2]:dim_segment[3]].sum(-1)
            # roo = C_cell_all[:,:,dim_segment[3]:dim_segment[4]].sum(-1)
            # agb = lea + ste + gra
            # d_lea = torch.cat([lea[:,[0]],lea[:,1:] - lea[:,:-1]],1)
            # d_ste = torch.cat([ste[:,[0]],ste[:,1:] - ste[:,:-1]],1)
            # d_gra = torch.cat([gra[:,[0]],gra[:,1:] - gra[:,:-1]],1)
            # d_roo = torch.cat([roo[:,[0]],roo[:,1:] - roo[:,:-1]],1)
            # d_agb = torch.cat([agb[:,[0]],agb[:,1:] - agb[:,:-1]],1)
            # C_consuming_maintenance = lea*0.02 + ste*0.01+gra*0.003+roo*0.01
            # C_consuming_growth  = F.relu(d_lea*(1.326-1))+F.relu(d_ste*(1.326-1))+F.relu(d_gra*(1.426-1))+F.relu(d_roo*(1.326-1))
            # C_consuming_knowledge = C_consuming_maintenance+C_consuming_growth
            
            # C_consuming_knowledge_np = utils.to_np(C_consuming_knowledge)
            
            a = res_min[obs_loc]
            b = res_max[obs_loc]
            np_wea_fer_batchs.append(np_wea_fer)
            np_res_batchs.append(np_res)
            np_pre_batchs.append(np_pre)
            np_obs_batchs.append(np_obs)
            np_fit_batchs.append(np_fit)
            mask_obs = var_o.ne(-10000)
            loss_res,loss_res_split = mask_mse(var_out_all, var_o, mask_obs)
            loss_ory,loss_res_split = mask_mse(var_y, var_o, mask_obs)
            grad = utils.gradients(var_out_all[:,110,-1],var_x)
            grad_np = utils.to_np(grad)
            
            # grad = [utils.gradients(var_out_all[:,i,-2],X_t_all[i]) for i in range(120)]
            # grad_np = utils.to_np(F.relu(-torch.stack(grad,1)[:,:,[0,3]]))
            # loss_pinn = (F.relu(-torch.stack(grad,1)[:,:,[0,3]])).masked_select(mask_res[:,:120,0:2]).mean()*2000
            # print("%s loss_pinn: %f"%(mode,loss_pinn))

            print('tes: %.8f'%(loss_res.data))
            print('ory: %.8f'%(loss_ory.data))

        
        np_wea_fer_dataset = np.concatenate(np_wea_fer_batchs,0)
        np_res_dataset = np.concatenate(np_res_batchs,0)
        np_pre_dataset = np.concatenate(np_pre_batchs,0)
        np_obs_dataset = np.concatenate(np_obs_batchs,0)
        np_fit_dataset = np.concatenate(np_fit_batchs,0)
        # np_pre_ref_dataset = np.concatenate(np_pre_ref_batchs,0)
        np_res_points = np_res_dataset.reshape(-1,obs_dim)
        np_pre_points = np_pre_dataset.reshape(-1,obs_dim)
        np_obs_points = np_obs_dataset.reshape(-1,obs_dim)
        np_fit_points = np_fit_dataset.reshape(-1,obs_dim)
        a = np_pre_dataset[:,50:100,1]
        np_pre_list.append(np_pre_points)
        # a = np_pre_dataset[np_pre_points]
    np_pre_points = np.stack(np_pre_list,0).mean(0)
        
    alpha, fontsize = 1, 14
    i = 0
    r2_obs = []
    rmse_obs = []
    for i,(title,unit) in enumerate(zip(obs_name,units)):
        x = np_obs_points[:,i]
        f = np_fit_points[:,i]
        y = np_pre_points[:,i]
        z = np_res_points[:,i]

        y = y[x>=0]
        z = z[x>=0]
        f = f[x>=0]
        x = x[x>=0]
        if title=="WRR14": title = "YIELD"
        R2, RMSE = utils.show_fit(x,y,title,24,max(max(x),max(z)),1,1,fig_type = "normal",markersize = 1, unit = unit,save_dir="figure/paper/ORYZA2000_2018-2018")
        r2_obs.append(R2)
        rmse_obs.append(RMSE)

        