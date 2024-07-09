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
from torch.autograd import Variable
from torch.utils.data import DataLoader

import datetime
import time
from models_aux.MyDataset import MyDataSet
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


   
if __name__ == "__main__":
    # %%load base data
    fig_num = 4
    for seed in range(0,25):
        tra_year = "2018"
        cali = ""
        model_type_list = [
            # "Naive_LSTM_cum_fit_scratch",
                            # "Naive_LSTM_cum_obs_scratch",
                            "MC_base_prior13_01_convergence_fit_scratch",
                           ]
        model_py_list = [
            # "Naive_LSTM_cum",
                            # "Naive_LSTM_cum",
                            "MC_base_prior13_01",
                           ]
        
        scal_type="nor"
        obs_mask_name = []
        obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
        obs_used_name = [name for name in obs_name if name not in obs_mask_name]
        # obs_name = ['WLV']
        units = ['-',"m$^2$/m$^2$","kg/ha","kg/ha","kg/ha","kg/ha","kg/ha"]
        sample_2018, sample_2019 = 65,40
        use_pretrained = False
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
        pre_list = []
        for model_type,model_py in zip(model_type_list,model_py_list):
            exec("from models_aux.%s import DEEPORYZA"%model_py)
            model_list = os.listdir("model_weight/%s/"%model_type) 
            model_list = [tpt for tpt in model_list if tra_year in tpt]
            
            model = model_list[seed]
            model_path = 'model_weight/%s/%s'%(model_type,model)
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
            # dvs super parameter  
            model = DEEPORYZA(input_dim = input_dim, hidden_dim = hidden_dim_dvs,layer_dim = layer_dim_dvs, 
                                output_dim = output_dim)
            model.to(device) 
         
            model_to_load = torch.load(os.path.join(model_path,trained_model_name))
            model.load_state_dict(model_to_load,strict=True)  
    
          
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
                var_out_all, aux_all = model(var_x[:,:,[1,2,3,7,8]],var_y)
                np_wea_fer = utils.unscalling(scal_type,utils.to_np(var_x),wea_fer_max,wea_fer_min,wea_fer_mean,wea_fer_std)
                np_res = utils.unscalling(scal_type,utils.to_np(var_y),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
                np_pre = utils.unscalling(scal_type,utils.to_np(var_out_all),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
                np_obs = utils.unscalling(scal_type,utils.to_np(var_o),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
                np_fit = utils.unscalling(scal_type,utils.to_np(var_f),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
    
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
            pre_list.append(np_pre_dataset)
            
            from matplotlib import rcParams
            config = {
                    "font.family": 'serif',
                    "font.size": 10,# 相当于小四大小
                    "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
                    "font.serif": ['SimSun'],#宋体
                    'axes.unicode_minus': False, # 处理负号，即-号
                  }
            from matplotlib.ticker import FuncFormatter
            def to_integer(x, pos):
                return '%d' % x
            rcParams.update(config)
            obs_name = ['DVS',"PAI","WLV","WST","WSO","WAGT","YIELD"]
            title_list =[
                # "NAIVE_LSTM",
                            # "MC_base_prior13_00",
                            "MC_base_prior13_01",]
            # colors = ['blue', "green",'red', ]
            colors = ['red', ]
            lines = ['--','--','--', '-', '-', '-']
            case_names = [" Train with RAW-dataset "," Train with interpolated-dataset "]
            formatter = FuncFormatter(to_integer)
            nrows = 6
            ncols = 1
            fig, axs = plt.subplots(dpi = 300,nrows=nrows, ncols=ncols, figsize=(2.5, 8))
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.8,
                                top=0.9,
                                wspace=0.3,
                                hspace=0.1)
            sample_loc = -3
            # 2018-2019: 4, -1
            # 2019-2018: -2, -3
            obs_tpt = ["PAI","WLV","WST","WSO","WAGT","YIELD"]
            titles = ['植被面积指数','叶质量','茎质量','穗质量','地上生物量',"产量"]
            max_values = [8,6000,6000,8000,14000,10000]
            # for i,obs_tpt in enumerate( ["PAI","WLV","WST","WSO","WAGT","YIELD"]):
            for i in range(nrows):
                for j in range(ncols):
                    tpt_loc = obs_name.index(obs_tpt[i*ncols + j])
                    print(tpt_loc)
                    day = np_wea_fer_dataset[sample_loc,:,0]
                    res = np_res_dataset[sample_loc,:,tpt_loc]
                    obs = np_obs_dataset[sample_loc,:,tpt_loc]
                    
                    if nrows==1 or ncols==1:    
                        axs_ij = axs[i*ncols + j]
                    else:
                        axs_ij =  axs[i,j]
                    axs_ij.scatter(day[(obs>=0)*(day>=0)],obs[(obs>=0)*(day>=0)],s=5,c='black',label="obseravation")
                    axs_ij.plot(day[(res>=0)*(day>=0)],res[(res>=0)*(day>=0)],c='black',linewidth=1,label="ORYZA2000")
                    
        
                    for n,(title,np_pre_dataset,line,color) in enumerate(zip(title_list,pre_list,lines,colors)):
                        pre = np_pre_dataset[sample_loc,:,tpt_loc]
                        if "00" in title:
                            alpha = 0.5
                        else:
                            alpha = 1
                        axs_ij.plot(day[(res>=0)*(day>=0)],pre[(res>=0)*(day>=0)],c=color,linewidth=0.75,alpha = alpha, label=title)
                            
        
        
                    # if j == 0:  # Set ylabel only for the first column
                    axs_ij.set_ylabel("%s(%s)"%(titles[i*ncols + j],units[tpt_loc]))  # Add y-axis label
                    axs_ij.set_yticklabels(axs_ij.get_yticks(), rotation=90)
                    #     # if i>1:
                    axs_ij.yaxis.set_major_formatter(formatter)
                    axs_ij.set_ylim(top=max_values[i*ncols + j])
                    # # else:  # Hide yticks for other columns
                    #     # axs_ij.set_yticklabels([])
                    if i == nrows-1:  # Set xlabel only for the last row
                        axs_ij.set_xlabel("日序数")
                    else:
                        axs_ij.set_xticklabels([])
                    axs_ij.text(0.03, 0.95, "(%s%d)"%(chr(97 + tpt_loc-1),fig_num), transform=axs_ij.transAxes,
                    fontsize=14, fontweight='bold', va='top')
                    
                    # axs_ij.text(0.25, 0.95, "%s"%(obs_tpt[i*ncols + j]), transform=axs_ij.transAxes,
                    # fontsize=14, fontweight='bold', va='top')
        
                    # plt.legend()
                    title = "%s_%d"%(obs_tpt[i*ncols + j],sample_loc)
                    # plt.title(title)
                    utils.find_or_make("figure/%s"%model_type)
                    
            # plt.legend()
            plt.savefig('figure/%s/%s_%02d.png'%(model_type,tra_year,seed), bbox_inches='tight')
            plt.show()
            plt.close()
            
            
