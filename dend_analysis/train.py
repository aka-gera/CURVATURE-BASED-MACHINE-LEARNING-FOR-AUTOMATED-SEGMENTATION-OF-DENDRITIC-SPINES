 
 # 
import sys,os
file_path_org=os.getcwd()   
DTYPE='float32'  
from dend_fun_0.main_0 import app_run_param,algorithm,algorithm_param,get_data, get_data_all  ,get_dict_param
from dend_fun_0.get_path import get_name 
import time 
import random#
import numpy as np


def get_aug_path(model_type, default=None):
    aug_dic = {
        'sm00000' :'smooth_pca_scaling_1_angle_1_translate_0',
        'sm07000' :'smooth_resize_7000_pca_scaling_1_angle_1_translate_0',
        'sm05000' :'smooth_resize_5000_pca_scaling_1_angle_1_translate_0',
        'nsm05000':'resize_5000_pca_scaling_1_angle_1_translate_0',
    } 
    model_type_split = model_type.lower().split('_')

    for key, path in aug_dic.items():
        for mm in model_type_split:
            if key ==mm:
                return path 
    return default



 
 
file_path_org=os.getcwd()  
file_path_data=os.path.dirname(file_path_org)  
gdas=get_data(file_path_data) 
dend_names_chld={} 
gdas=get_data_all(names_dic=dend_names_chld,dend_data=gdas.dend_data,file_path_data=file_path_data)
 
nam_gen,action='p21-dendrites','resize'


dend_names = ['BTLNG_d004_RECON','BTLNG_d005_RECON','BTLNG_d010_RECON','CNJVY_d001_RECON','CNJVY_d003_RECON','CNJVY_d005_RECON',]   
dend_last = ['BTLNG', 'BTLNG', 'BTLNG', 'CNJVY', 'CNJVY', 'CNJVY', ] 
dend_first = ['d004', 'd005',  'd010',  'd001', 'd003',  'd005', ]  
dend_namess = [[f'{dn}_',df,f'{df}sp'] for dn,df in zip(dend_names,dend_first)]
extraa=[f'{nam_gen}_{action}_{mm}' for mm in [1000,5000,7000,15000]]
extraaaa=[f'{nam_gen}_smooth_{action}_{mm}' for mm in [1000,5000,7000,10000,15000,25000]]
for nam in [ nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth', ]+extraa+extraaaa:
    obj_org_path= os.path.join(file_path_data,nam_gen, nam )
    dend_names_chld[nam]=dict( 
        dend_namess=dend_namess ,
        dend_names=dend_names ,
        dend_last=dend_last ,
        dend_first=dend_first ,
        dend_path_inits=[nam  for _ in range(len(dend_names))],
        name_spine_id='sp',
        name_head_id='hsp',
        name_neck_id='nsp',
        name_shaft_id='shsp',
        objs_path_org=None,
        weights=[[3.,0.81]],
        obj_org_path=obj_org_path,
        )

gdas=get_data_all(names_dic=dend_names_chld,dend_data=gdas.dend_data,file_path_data=file_path_data)

# choose dataset to train 
 
nam ='p21-dendrites_resized_7000' 
nam_gen,action,nnbb='neuropil_recon_0','resized', 5000
nam_gen,action,nnbb='p21-dendrites','resize', 5000  
nam=f'{nam_gen}_{action}_{nnbb}'

nam = 'p21-dendrites' 
nam = 'p21-dendrites_resize_5000' 
nam = 'p21-dendrites_smooth_resize_5000'
nam = 'p21-dendrites_smooth_resize_15000'
nam = 'p21-dendrites_smooth_resize_10000'
nam = 'p21-dendrites_smooth'  
 
  

 # Pick name for the model 

nnbb=5000 
model_type=f'vol_VGG16_FCN3D_{nnbb}_hpcc_crop' 
model_type=f'vol_3UNet3D3_{nnbb}_hpcc_crop'
model_type=f'vol_VoxNetSeg_{nnbb}_hpcc_crop'  

model_type='vol_VoxNetSeg_SM07000_AUG' 
model_type='vol_UNet3D_NSM05000_AUG'
model_type='cml_cML'
model_type='gcn_UNet_SM10000_LOC' 
model_type='dnn_GINN_SM00000_LOC_AUG' 


# Pick feature type
dnn_mode = 'DNN-0'         
dnn_mode = "DNN-4"          
dnn_mode = 'DNN-1'          
dnn_mode = 'DNN-2'          
dnn_mode = 'DNN-3'          





path_heads_show=[ 
            f'vol_3UNet3D3_{nnbb}_hpcc_crop',
            f'vol_VGG16_FCN3D_{nnbb}_hpcc_crop',
            f'vol_VoxNetSeg_{nnbb}_hpcc_crop',  
            'dnn_GINN_SM00000_LOC_AUG',
] 


path_heads_show=path_heads_show if model_type in path_heads_show else path_heads_show+[model_type]
path_heads=[  ] 
path_heads = list(set(path_heads+path_heads_show))
if model_type not in path_heads:
    path_heads.append(model_type) 
 
  
path_display = ['dest_shaft_path', ]

 
n_step = 0       # smoothing step size'resize_10000','resize_10000_pca'
weight=3          # Decrease for less spine mesh/Increase for more spine mesh
size_threshold=200   # mesh size threshold above which a spine is consider 
weight=0.2


weight=16# smoothing step size

if nam.startswith("p21"):
    size_threshold = 300
elif nam.startswith("vol"):
    size_threshold = 10
else:
    size_threshold = 100

weight_pre=weight
path_dir_nam=f'dice'
path_dir=f'{path_dir_nam}_we_{weight}'
data_dir=f'{path_dir_nam}_we_3'
data_dir=path_dir


dict_param=get_dict_param(nam=nam,
                    n_step =n_step,
                    weight=weight,
                    size_threshold=size_threshold,
                    path_heads_show=path_heads_show,
                    path_heads=path_heads,
                    )  
dict_param['model_type_data']=model_type 
param = algorithm_param(**dict_param)



dend_data = gdas.part(nam)  
mapp = app_run_param(param)
param=mapp.emerge_param()
param['Smooth']['tf'] = False   #  False   #False   # False   # True/False: choose whether to smooth the data
param['annotations']['tf'] =  False   # True   # True/False: choose whether to generate annotations for training, accuracy, and recall 
param['get_training']['tf'] = True  # True/False: enable training data generation
param['clean_path_dir']['tf'] = False          # True/False: delete computed data
mj = [x / 1000 for x in range(1, 1000, 200)]  
param['get_training']['param']['weight']=[[a, b] for a, b in zip(mj, mj[::-1])]
param['dendrite_pred']['param']['param_dic']['tf_restart']['get_pinn_features']= False   #True   #

entry_names=[None,]
ls =[1,2,3,4,] 
 
if model_type.endswith(('AUG',)):
    uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),f'{get_aug_path(model_type)}.txt'),dtype=str)  
    tott=(len(uyu)+1)*len(dend_data['dend_names'])
    entry_names.extend(uyu)
    print(entry_names)
    lss=list(set(range(tott-2))-set(ls))
    ls=np.unique(ls+random.sample(lss, int(len(lss) * 0.7)))
    print('[[[[',ls,len(ls),tott)
    print(max(ls),max(set(range(tott))-set(ls)))  
 
param['get_training']['param']['ls']=ls
param['get_training']['param']['itime']=100000
alg = algorithm(param)
alg.train(
    dend_data = dend_data,  
    true_name = 'true_0', 
    dnn_mode = dnn_mode,
    model_type = model_type,  
    path_display = path_display, 
    entry_names=entry_names,
    path_dir=path_dir,
    path_heads_show=path_heads,
)
