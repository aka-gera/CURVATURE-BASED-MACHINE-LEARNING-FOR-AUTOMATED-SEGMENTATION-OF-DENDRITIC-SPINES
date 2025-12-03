 
 # 
import sys,os
file_path_org=os.getcwd()   
DTYPE='float32'  
from dend_fun_0.main_0 import app_run_param,algorithm,algorithm_param,get_data, get_data_all 
 
from dend_fun_0.get_path import get_name 
import time 

file_path_data=r'/Users/akag/Desktop/'     # Change this directory to the path dataset 
gdas=get_data(file_path_data) 
dend_names_chld={} 
gdas=get_data_all(names_dic=dend_names_chld,dend_data=gdas.dend_data,file_path_data=file_path_data)

# choose dataset to train
nam = 'tom_dend_2'
nam = 'neuropil_recon_0' 
nam = 'p21-dendrites_init'
nam = 'p21-dendrites' 

gnam = get_name() 
dend_data = gdas.part(nam)  
 
  
path_heads = ['pinn', 'cML', 'pinn_old']

model_type = 'ML'  
model_type = 'cML' 
model_type = 'pinn_old'

path_display = ['dest_shaft_path']
path_display = ['dest_shaft_path', 'dest_spine_path']

param = algorithm_param() 
mapp = app_run_param(param)
param=mapp.emerge_param()

param['smooth']['tf'] = False       # True/False: choose whether to smooth the data
param['dendrite_pred']['tf'] = True # True/False:  dendrite prediction
param['annotations']['tf'] = True  #False  # True/False: generate annotations for training, accuracy, and recall
param['get_training']['tf'] = True  # True/False: enable training data generation
param['clean_path_dir']['tf'] = False          # True/False: delete computed data

alg = algorithm(param)
alg.train(   
    dend_data = dend_data,  
    true_name = 'true_0', 
    dnn_mode = 'mode1',
    model_type = model_type,  
    path_display = path_display, 
)
