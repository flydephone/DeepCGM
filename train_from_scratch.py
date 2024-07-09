 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:56:07 2020
1、在原有基础上加入人工激活层，保证基本物理规律（物候正增长，总干物质正增长，各器官质量守恒）
2、参数作为输入，不作为隐藏状态
@author: hanjingye
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import random
import copy
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from models_aux.MyDataset import MyDataSet
from models_aux.MC_base_prior13 import DEEPORYZA
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def run(loss_target,obs_mask,DataLoader,mode,model,optimizer,scheduler,e):
    def forward(loss_target,obs_mask,DataLoader,mode,model,optimizer,scheduler,e):
        loss_res_dataset, loss_obs_dataset, loss_fit_dataset = 0, 0, 0
        loss_res_split_dataset, loss_obs_split_dataset, loss_fit_split_dataset = 0, 0, 0
        for n,(x,y,o,f) in enumerate(DataLoader):
            var_x, var_y, var_o, var_f = x.to(device), y.to(device), o.to(device), f.to(device)
            mask_res = var_y.ne(-10000)
            mask_obs = var_o.ne(-10000)
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
    model_name = "MC_base_prior13_convergence"
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

    for seed in range(0,25):
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

        def kgml_loss(pred,res,mask,aux_all,X):
            dims = [8,8,8]
            dim_segment = np.cumsum([0]+dims)
            mse_loss = nn.MSELoss(reduction='none')
            mask = mask[:,:-1,:]
            mask_partitation_mat = mask[:,:,[0],None].repeat(1,1,1,dim_segment[-1])
            C_cell_all,C_cell_convergence_all = aux_all

            # %% convergence loss
            C_converge_loss_2 = mse_loss(C_cell_convergence_all,C_cell_all).masked_select(mask_partitation_mat[:,:,0,:]).mean()*100000
            return C_converge_loss_2
                
        def mask_mse(pred,real,mask):
            weights =  [1,1,5,2,2,1,2]
            mse_loss = nn.MSELoss(reduction='none')
            loss = mse_loss(pred, real)
            loss_split_mask = [loss[:,:,i].masked_select(mask[:,:,i]).mean()*weights[i] for i in range(loss.shape[2])]
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
        
