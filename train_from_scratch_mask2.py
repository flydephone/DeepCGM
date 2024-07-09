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
from torch.utils.data import DataLoader, Dataset

import datetime
import time
from models_aux.MyDataset import MyDataSet
from models_aux.MC_base_prior13_01 import DEEPORYZA
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def run(loss_target,obs_mask,DataLoader,mode,model,optimizer,scheduler,e):
    def forward(loss_target,obs_mask,DataLoader,mode,model,optimizer,scheduler,e):
        loss_res_dataset, loss_obs_dataset, loss_fit_dataset = 0, 0, 0
        loss_res_split_dataset, loss_obs_split_dataset, loss_fit_split_dataset = 0, 0, 0
        for n,(x,y,o,f) in enumerate(DataLoader):
            var_x, var_y, var_o, var_f = x.to(device), y.to(device), o.to(device), f.to(device)
            var_l = utils.to_np(var_o)
            # a = utils.to_np(var_o)
            # aa = utils.interpolate_missing_values_3d(a)
            mask_res = var_y.ne(-10000)
            mask_obs = var_o.ne(-10000)
            mask_obs[:,1:50,:] = False
            for tpt in obs_mask:
                mask_obs[:,:,tpt] = False
            # 前向传播
            var_out_all,aux_all = model(var_x[:,:,[1,2,3,7,8]],var_y)
            loss_res,loss_res_split = mask_mse(var_out_all, var_y, mask_res)
            loss_fit,loss_fit_split = mask_mse(var_out_all, var_f, mask_res)
            loss_obs,loss_obs_split = mask_mse(var_out_all, var_o, mask_obs)
            if len(aux_all)>0:
                loss_kgml = kgml_loss(var_out_all, var_y, mask_res, aux_all,var_x[:,:,[1,2,3,7,8]])
                print("%s kgml: %f"%(mode,loss_kgml))
            loss_res_dataset += loss_res  
            loss_fit_dataset += loss_fit
            loss_obs_dataset += loss_obs  
            loss_res_split_dataset += loss_res_split
            loss_fit_split_dataset += loss_fit_split            
            loss_obs_split_dataset += loss_obs_split
            
            if mode=="tra": 
                optimizer.zero_grad()
                if loss_target == "res":
                    loss = loss_res
                elif loss_target == "obs":
                    loss = loss_obs
                elif loss_target == "fit":
                    loss = loss_fit
                alpha = 1
                # alpha = 0.1
                # if loss_res<0.04:
                #     alpha = 1-(1-0.1)*(loss.item()/0.05)
                if len(aux_all)>0:
                    (loss+loss_kgml*alpha*kgml_trigger).backward()
                else:
                    (loss).backward()
                optimizer.step()
        if mode=="tra": 
            scheduler.step()
        loss_res, loss_res_split = loss_res_dataset/len(DataLoader), loss_res_split_dataset/len(DataLoader)
        loss_obs, loss_obs_split = loss_obs_dataset/len(DataLoader), loss_obs_split_dataset/len(DataLoader)
        loss_fit, loss_fit_split = loss_fit_dataset/len(DataLoader), loss_fit_split_dataset/len(DataLoader)
        return loss_res,loss_obs,loss_fit,loss_res_split,loss_obs_split,loss_fit_split
    
    if mode=="tra":
        model.train()
        loss_res,loss_obs,loss_fit,loss_res_split,loss_obs_split,loss_fit_split = forward(loss_target,obs_mask,DataLoader,mode,model,optimizer,scheduler,e)
    elif mode=="tes" or mode=="val":
        model.eval()
        with torch.no_grad():
            loss_res,loss_obs,loss_fit,loss_res_split,loss_obs_split,loss_fit_split = forward(loss_target,obs_mask,DataLoader,mode,model,optimizer,scheduler,e)
    return loss_res,loss_obs,loss_fit,loss_res_split,loss_obs_split,loss_fit_split
   
if __name__ == "__main__":
    # %%load base data
    tra_year = "2019"
    target = "obs"
    model_name = "MC_base_prior13_01_convergence_mask"
    model_name = "%s_%s"%(model_name,target)
    kgml_trigger = 1
    tra_ratio = 1
    model_name = "%s_scratch"%model_name
    fig_dir = "%s_%s"%(model_name,tra_year)
    utils.find_or_make("figure/%s"%fig_dir)
    scal_type="nor"
    obs_mask_name = []
    obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]
    obs_used_name = [name for name in obs_name if name not in obs_mask_name]
    units = ['-',"m$^2$/m$^2$","kg/ha","kg/ha","kg/ha","kg/ha","kg/ha"]
    sample_2018, sample_2019 = 65,40
    use_pretrained = False

    r2_list, rmse_list = [], []

    rea_res_dataset,rea_par_dataset,rea_wea_fer_dataset,rea_obs_dataset,rea_fit_dataset = utils.dataset_loader(data_source="format_dataset/%s/real_%s"%(scal_type,tra_year))

    for seed in range(0,1):
        utils.setup_seed(seed)
        robost_name = "%s_%s_%02d"%(tra_year,tra_year,seed)
        if tra_year == "2018":
            tra_res_dataset,tra_wea_fer_dataset,tra_obs_dataset,tra_fit_dataset = rea_res_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_obs_dataset[:sample_2018],rea_fit_dataset[:sample_2018]
            tes_res_dataset,tes_wea_fer_dataset,tes_obs_dataset,tes_fit_dataset = rea_res_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_obs_dataset[sample_2018:],rea_fit_dataset[sample_2018:]
        elif tra_year == "2019":
            tes_res_dataset,tes_wea_fer_dataset,tes_obs_dataset,tes_fit_dataset = rea_res_dataset[:sample_2018],rea_wea_fer_dataset[:sample_2018],rea_obs_dataset[:sample_2018],rea_fit_dataset[:sample_2018]
            tra_res_dataset,tra_wea_fer_dataset,tra_obs_dataset,tra_fit_dataset = rea_res_dataset[sample_2018:],rea_wea_fer_dataset[sample_2018:],rea_obs_dataset[sample_2018:],rea_fit_dataset[sample_2018:]
    
        tra_num = int(len(tra_res_dataset)*tra_ratio)
        val_num = len(tra_res_dataset) - tra_num
        combined = list(zip(tra_res_dataset,tra_wea_fer_dataset,tra_obs_dataset,tra_fit_dataset))
        tra_res_dataset,tra_wea_fer_dataset,tra_obs_dataset,tra_fit_dataset = zip(*combined[:tra_num])
        if tra_ratio<1:
            random.shuffle(combined)
            val_res_dataset,val_wea_fer_dataset,val_obs_dataset,val_fit_dataset = zip(*combined[tra_num:])
        
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
        if tra_ratio<1:
            val_set = MyDataSet(obs_loc=obs_loc, res=val_res_dataset, wea_fer=val_wea_fer_dataset, obs=val_obs_dataset, fit=val_fit_dataset, max_min=max_min, mean_std=mean_std, scal_type=scal_type, batch_size=batch_size, aug=False)
            val_DataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        tes_set = MyDataSet(obs_loc=obs_loc, res=tes_res_dataset, wea_fer=tes_wea_fer_dataset, obs=tes_obs_dataset, fit=tes_fit_dataset, max_min=max_min, mean_std=mean_std, scal_type=scal_type, batch_size=batch_size, aug=False)
        tes_DataLoader = DataLoader(tes_set, batch_size=batch_size, shuffle=False)
    
        # %% creat instances from class_LSTM
        hidden_dim_dvs = 64
        layer_dim_dvs = 1
        
        # dvs super parameter  
        model = DEEPORYZA(input_dim = input_dim, hidden_dim = hidden_dim_dvs,layer_dim = layer_dim_dvs, 
                            output_dim = output_dim)
        model.to(device) 
      
        criterion = nn.MSELoss()
        def projection_M(M, lambda_=1):
            # 计算M的转置
            M_transpose = M.transpose(-1, -2)
            # 计算M的平方
            M_square = torch.einsum('...ji,...jk->...ik', M, M)
            # 计算损失
            loss = torch.norm(M_square - M, dim=(-2, -1))**2
            # a = utils.to_np(M - M_transpose)[0]
            return loss

        def kgml_loss(pred,res,mask,aux_all,X):
            dims = [8,8,8]
            dim_segment = np.cumsum([0]+dims)
            layer = nn.ReLU()
            mse_loss = nn.MSELoss(reduction='none')
            batch_size = res.shape[0]
            avaible_length = torch.sum(mask[:,:,1],1)   
            mask = mask[:,:-1,:]
            a = utils.to_np(mask)
            # mask_fertilization = (X[:,:,[-2]]>0)[:,:,None,:]
            mask_redistribution_mat = mask[:,:,[0],None].repeat(1,1,dim_segment[-1],dim_segment[-1])
            mask_partitation_mat = mask[:,:,[0],None].repeat(1,1,1,dim_segment[-1])
            
            C_cell_all,C_cell_convergence_all = aux_all

            # %% convergence loss
            identity_matrix = torch.eye(dim_segment[-1]).to(device)
            C_converge_loss_2 = mse_loss(C_cell_convergence_all,C_cell_all).masked_select(mask_partitation_mat[:,:,0,:]).mean()*100000
            return C_converge_loss_2
            # return sa_all*0
        
                
        def mask_mse(pred,real,mask):
            weights =  [1,1,5,2,2,1,2]
            mse_loss = nn.MSELoss(reduction='none')
            loss = mse_loss(pred, real)
            loss_split_mask = [loss[:,:,i].masked_select(mask[:,:,i]).mean()*weights[i] for i in range(loss.shape[2])]
            a = utils.to_np(real)
            return sum(loss_split_mask),torch.Tensor(loss_split_mask)
        
        print("one Epoch has %d sample"%len(tra_set))
        
        def trained_dict_fun(model,pool):
            model_dict = model.state_dict()
            if pool == 1: trained_dict = {k: v for k, v in model_dict.items()}
            if pool == 2: trained_dict = {k: v for k, v in model_dict.items() if 'init' not in k and "R2C" not in k and "lstm_control" not in k}
            if pool == 3: trained_dict = {k: v for k, v in model_dict.items() if 'lstm' not in k and "redistribution" not in k and "consuming" not in k}
            trained_params = [param for name, param in model.named_parameters() if name in trained_dict]
            return trained_dict, trained_params
        
        trained_dict, trained_params = trained_dict_fun(model,1)
        
        if "Naive" in model_name:
            lr = 0.005    # optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.0)
        else:
            lr = 0.1
        # optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.0)
        optimizer = torch.optim.Adam(trained_params, lr=lr, weight_decay=0.0000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1**(int(epoch/250)))
        
        obs_mask = [i for i,obs in enumerate(obs_name) if obs in obs_mask_name]
        obs_used = [i for i,obs in enumerate(obs_name) if obs in obs_used_name]
        # %% dvs model train and test
        val_loss_min = 10
        epoch_model = 0
        max_epoch = 700
        log = {"tra_res_loss":[],"tra_obs_loss":[],"tra_fit_loss":[],"tra_obs_loss_split":[],"tra_res_loss_split":[],"tra_fit_loss_split":[],
               "val_res_loss":[],"val_obs_loss":[],"val_fit_loss":[],"val_obs_loss_split":[],"val_res_loss_split":[],"val_fit_loss_split":[],
               "tes_res_loss":[],"tes_obs_loss":[],"tes_fit_loss":[],"tes_obs_loss_split":[],"tes_res_loss_split":[],"tes_fit_loss_split":[],}
        
        start_time = time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time()))
        for e in range(max_epoch):
            # 计算训练集损失
            tra_res_loss,tra_obs_loss,tra_fit_loss, tra_res_loss_split,tra_obs_loss_split,tra_fit_loss_split = run(target,obs_mask,tra_DataLoader,'tra',model,optimizer,scheduler,e)
            # 计算验证集损失
            tes_res_loss,tes_obs_loss,tes_fit_loss, tes_res_loss_split,tes_obs_loss_split,tes_fit_loss_split = run("obs",obs_mask,tes_DataLoader,'tes',model,optimizer,scheduler,e)
            if tra_ratio<1:
                val_res_loss,val_obs_loss,val_fit_loss, val_res_loss_split,val_obs_loss_split,val_fit_loss_split = run("obs",obs_mask,val_DataLoader,'val',model,optimizer,scheduler,e)
            else:
                val_res_loss,val_obs_loss,val_fit_loss, val_res_loss_split,val_obs_loss_split,val_fit_loss_split =  tes_res_loss,tes_obs_loss,tes_fit_loss, tes_res_loss_split,tes_obs_loss_split,tes_fit_loss_split
            log["tra_res_loss"].append(tra_res_loss.clone().data.cpu().numpy())
            log["tra_obs_loss"].append(tra_obs_loss.clone().data.cpu().numpy())
            log["tra_fit_loss"].append(tra_fit_loss.clone().data.cpu().numpy())
            log["tra_res_loss_split"].append(tra_res_loss_split.clone().data.cpu().numpy())
            log["tra_obs_loss_split"].append(tra_obs_loss_split.clone().data.cpu().numpy())
            log["tra_fit_loss_split"].append(tra_fit_loss_split.clone().data.cpu().numpy())
            log["val_res_loss"].append(val_res_loss.clone().data.cpu().numpy())
            log["val_obs_loss"].append(val_obs_loss.clone().data.cpu().numpy())
            log["val_fit_loss"].append(val_fit_loss.clone().data.cpu().numpy())
            log["val_res_loss_split"].append(val_res_loss_split.clone().data.cpu().numpy())
            log["val_obs_loss_split"].append(val_obs_loss_split.clone().data.cpu().numpy())
            log["val_fit_loss_split"].append(val_fit_loss_split.clone().data.cpu().numpy())
            log["tes_res_loss"].append(tes_res_loss.clone().data.cpu().numpy())
            log["tes_obs_loss"].append(tes_obs_loss.clone().data.cpu().numpy())
            log["tes_fit_loss"].append(tes_fit_loss.clone().data.cpu().numpy())
            log["tes_res_loss_split"].append(tes_res_loss_split.clone().data.cpu().numpy())
            log["tes_obs_loss_split"].append(tes_obs_loss_split.clone().data.cpu().numpy())
            log["tes_fit_loss_split"].append(tes_fit_loss_split.clone().data.cpu().numpy())
            print("第%d个epoch的学习率：%f" %(e, optimizer.param_groups[0]['lr']))
            print('【OBS】Epoch: %s lr:%f, tra: %.8f val: %.8f, tes: %.8f'%(e,optimizer.param_groups[0]['lr'], tra_obs_loss.data, val_obs_loss.data, tes_obs_loss.data))
            print('【FIT】Epoch: %s lr:%f, tra: %.8f val: %.8f, tes: %.8f'%(e,optimizer.param_groups[0]['lr'], tra_fit_loss.data, val_fit_loss.data, tes_fit_loss.data))
            now = time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time()))
            save_path = os.path.join('model_weight',"%s"%model_name,"%s_%s"%(robost_name,start_time))
            utils.find_or_make(save_path)
            save_name = '%s_%s_%02d_tra_%.4f_val_%.4f_tes_%.4f.pkl'%(now,tra_year,e,tra_obs_loss,val_obs_loss,tes_obs_loss)
            model_to_save = copy.deepcopy(model.state_dict())
            epoch_model = copy.deepcopy(e)
            val_loss_min=val_obs_loss
            torch.save(model_to_save, os.path.join(save_path,save_name))
    
        # # %%plot
        # -----------------------------------loss------------------------------------
    
        loss_pool = ["tra_res_loss","val_res_loss",
                     "tra_obs_loss","val_obs_loss",
                     ]
        loss_split_pool = ["tra_obs_loss_split","val_obs_loss_split",]
    
        loss_pool = [tpt for tpt in loss_pool if "obs" in tpt]
        fig = plt.figure(dpi = 300)
        max_lim = 0
        for tpt in loss_pool:
            max_lim = max(max_lim,max(log[tpt]))
            plt.plot(log[tpt], label = tpt)
            plt.legend()
            plt.ylim(0,max_lim)
            plt.ylabel("Loss")
            plt.xlabel("Epoch number")
            title_tpt = [tpt for tpt in obs_name if tpt not in obs_mask_name]
            title = "Loss of %s"%("_".join(title_tpt))
            plt.title(title)
        plt.savefig('figure/%s/%s.png'%(fig_dir,title), bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        for tpt in loss_split_pool:
            fig = plt.figure(dpi = 300)
            max_lim = 0
            loss_split_value = np.array(log[tpt])
            for loc in obs_used:
                y = loss_split_value[:,loc]
                max_lim = max(max_lim,max(y))
                plt.plot(y, label = obs_name[loc])
                plt.legend()
                plt.ylim(0,max_lim)
                plt.ylabel("Loss")
                plt.xlabel("Epoch number")
                title = "Loss of %s"%(tpt)
                plt.title(title)
            plt.savefig('figure/%s.png'%title, bbox_inches='tight')
            plt.show()
            plt.close(fig)
    
        # # -----------------------------------fit------------------------------------
        # model.load_state_dict(model_to_save,strict=True)  
        np_wea_fer_batchs, np_res_batchs, np_pre_batchs, np_obs_batchs, np_pre_ref_batchs = [],[],[], [], []
        mode = "tes"
        for n,(x,y,o,f) in enumerate(tes_DataLoader):
            var_x, var_y, var_o, var_f = x.to(device), y.to(device), o.to(device), f.to(device)
            var_out_all, aux_all = model(var_x[:,:,[1,2,3,7,8]],var_y)
            np_wea_fer = utils.unscalling(scal_type,utils.to_np(var_x),wea_fer_max,wea_fer_min,wea_fer_mean,wea_fer_std)
            np_res = utils.unscalling(scal_type,utils.to_np(var_y),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_pre = utils.unscalling(scal_type,utils.to_np(var_out_all),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            np_obs = utils.unscalling(scal_type,utils.to_np(var_o),res_max[obs_loc],res_min[obs_loc],res_mean[obs_loc],res_std[obs_loc])
            a = res_min[obs_loc]
            b = res_max[obs_loc]
            np_wea_fer_batchs.append(np_wea_fer)
            np_res_batchs.append(np_res)
            np_pre_batchs.append(np_pre)
            np_obs_batchs.append(np_obs)
        
        np_wea_fer_dataset = np.concatenate(np_wea_fer_batchs,0)
        np_res_dataset = np.concatenate(np_res_batchs,0)
        np_pre_dataset = np.concatenate(np_pre_batchs,0)
        np_obs_dataset = np.concatenate(np_obs_batchs,0)
        np_res_points = np_res_dataset.reshape(-1,obs_dim)
        np_pre_points = np_pre_dataset.reshape(-1,obs_dim)
        np_obs_points = np_obs_dataset.reshape(-1,obs_dim)

        
        alpha, fontsize = 1, 14
        i = 0
        r2_obs = []
        rmse_obs = []
        for i,(title,unit) in enumerate(zip(obs_name,units)):
            x = np_obs_points[:,i]
            y = np_pre_points[:,i]
            z = np_res_points[:,i]
    
            y = y[x>=0]
            z = z[x>=0]
            x = x[x>=0]
            # if title == "DVS":
            #     y = y[(x-1.2)*(x-1.8)>=0]
            #     z = z[(x-1.2)*(x-1.8)>=0]
            #     r = r[(x-1.2)*(x-1.8)>=0]
            #     x = x[(x-1.2)*(x-1.8)>=0]
            save_dir = "figure/%s"%fig_dir
            # R2, RMSE = utils.show_fit(x,y,title,14,max(max(x),max(y)),1,1,fig_type = "normal",markersize = 1, unit = unit,save_dir=save_dir)
            # # R2, RMSE = utils.show_fit(x,f,title,14,max(max(x),max(r)),1,1,fig_type = "normal",markersize = 1, unit = unit,save_dir=None)
            # # R2, RMSE = utils.show_fit(x,z,title,14,max(max(x),max(z)),1,1,fig_type = "normal",markersize = 1, unit = unit,save_dir="figure/paper/ORYZA2000calibration_2019_training")
            # r2_obs.append(R2)
            # rmse_obs.append(RMSE)
        # obs_name = ['WLV']
        obs_name = ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"];
        sample_loc = 9
        # for sample_loc in range(0,1):
            
        #     for obs_tpt in ['DVS','PAI','WLV','WST','WSO','WAGT',"WRR14"]:
        #         tpt_loc = obs_name.index(obs_tpt)
                
        #         day = np_wea_fer_dataset[sample_loc,:,0]
        #         res = np_res_dataset[sample_loc,:,tpt_loc]
        #         pre = np_pre_dataset[sample_loc,:,tpt_loc]
        #         obs = np_obs_dataset[sample_loc,:,tpt_loc]
        #         plt.plot(day[(pre>=0)*(day>=0)],pre[(pre>=0)*(day>=0)],c='red',label="pre")
        #         plt.scatter(day[(obs>=0)*(day>=0)],obs[(obs>=0)*(day>=0)],c='black',label="obs")
        #         # plt.plot(day[(fit>=0)*(day>=0)],fit[(fit>=0)*(day>=0)],c='orange',label="fit")
        #         plt.plot(day[(res>=0)*(day>=0)],res[(res>=0)*(day>=0)],c='blue',label="res")
        #         plt.legend()
        #         title = "%s_%d"%(obs_tpt,sample_loc)
        #         plt.title(title)
        #         # utils.find_or_make("figure")
        #         plt.savefig('figure/%s/%s_%s.png'%(fig_dir,obs_tpt,sample_loc), bbox_inches='tight')
        #         plt.show()
        
        #         plt.close()
        r2_list.append(r2_obs)
        rmse_list.append(rmse_obs)
        
        r2_array = np.array(r2_list)
        rmse_array = np.array(rmse_list)
        
