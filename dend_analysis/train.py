 
 # 
import sys,os
file_path_org=os.getcwd()   
DTYPE='float32'  
from dend_fun_0.main_0 import app_run_param,algorithm,algorithm_param,get_data, get_data_all  ,get_dict_param
 
from dend_fun_0.get_path import get_name 
import time 

file_path_data=r'/Users/akag/Desktop/'     # Change this directory to the path dataset 

file_path_org=os.getcwd()  
file_path_data=os.path.dirname(file_path_org)  
gdas=get_data(file_path_data) 
dend_names_chld={} 
gdas=get_data_all(names_dic=dend_names_chld,dend_data=gdas.dend_data,file_path_data=file_path_data)

'''
nam='p21-dendrites'

dend_names = ['BTLNG_d004_RECON','BTLNG_d005_RECON','BTLNG_d010_RECON','CNJVY_d001_RECON','CNJVY_d003_RECON','CNJVY_d005_RECON',]   
dend_last = ['BTLNG', 'BTLNG', 'BTLNG', 'CNJVY', 'CNJVY', 'CNJVY', ] 
dend_first = ['d004', 'd005',  'd010',  'd001', 'd003',  'd005', ] 

dend_namess = [[f'{dn}_',df,f'{df}sp'] for dn,df in zip(dend_names,dend_first)]
extraa=[f'p21-dendrites_resized_{mm}' for mm in [1000,5000,7000,15000]]
for nam in [ 'p21-dendrites','p21-dendrites_resized', ]+extraa:
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
        )

'''
nam_gen='diane_raw_0_wrap'   
nam='d001'
nami='train'
obj_org_path = os.path.join(file_path_data,'diane',nam_gen,nam)
# dend_names=[ff for ff in os.listdir(obj_org_path) if ff.startswith(('BTLNG', 'CNJVY'))]  [:10]
dend_names=[ff for ff in os.listdir(obj_org_path) if ff.startswith((nam,'BTLNG', 'CNJVY'))]
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]


dend_names_chld[nam]= dict(  
                dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits,
                name_spine_id='p_sps',
                name_head_id='p_sphs',
                name_neck_id='p_spns',
                name_shaft_id='p_spsh',
                weights=[[16.,0.81]],
                )  



nam_gen,action='neuropil_recon_0','resized'
obj_list=[ls for ls in range(151) if ls not in [129]] #[7,8,9,10,11,14,16]#[2,3,4]#,[30,31,32,63]#[20,22]#
# obj_list=[0,1,2,3,4,7,8,9,10,11,14,16,20,22,30,31,32,59,63] 
obj_list1=[13,18,23,24,25,34,36,39,42,49,50,
           53,56,57,60,61,65,66,68,71,75,
           79,80,84,85,88,89,91,92,93,97,
           100,107,108,116,120,121,123,127,129,133,
           135,136,138,141,142,144,145,146,148,150]
obj_list2=[5,6,15,17,26,27,28,29,33,35,41,43,44,45,46,47,48,51,52,54,55,58,64,67,69,70,72,73,74,76,77,78,81,82,83,86,87,90,94,95,96,98,99,101,102,103,105,106,110,111,112,115,118 ,122,124,125,132,139,140,143,147,]
obj_list3=[37,38,114,119,126,128,130,131,134,137,149]
obj_list4=[117,113,62,63,40] # My additional selection
obj_list=list(set(obj_list)-set(obj_list1)) 
obj_list=list(set(obj_list)-set(obj_list2))
obj_list=list(set(obj_list)-set(obj_list3))
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 

extraa=[f'{nam_gen}_{action}_{mm}' for mm in [5000,7000,]]
for nam in [nam_gen,f'{nam_gen}_{action}', ]+extraa:
    obj_org_path= os.path.join(file_path_data,'neuropil_0', nam )
    dend_names_chld[nam]= dict(  
                    dend_namess=dend_namess ,
                    dend_names=dend_names , 
                    obj_org_path=obj_org_path,
                    dend_last=dend_last ,
                    dend_first=dend_first ,
                    dend_path_inits=dend_path_inits,
                    name_spine_id='p_sps',
                    name_head_id='p_sphs',
                    name_neck_id='p_spns',
                    name_shaft_id='p_spsh',
                    ) 




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
nam = 'tom_dend_2'
nam = 'neuropil_recon_0' 
nam = 'p21-dendrites_init'
nam='d001'

nam ='p21-dendrites_resized'
nam ='p21-dendrites_resized_7000'
nam='p21-dendrites_resized_1000'
nam_gen,action,nnbb='neuropil_recon_0','resized', 5000
nam_gen,action,nnbb='p21-dendrites','resize', 5000  
nam=f'{nam_gen}_{action}_{nnbb}'

nam = 'p21-dendrites' 
nam='p21-dendrites_resize_5000' 
nam = 'p21-dendrites_smooth_resize_5000'
nam = 'p21-dendrites_smooth_resize_15000'
nam = 'p21-dendrites_smooth_resize_10000'
nam = 'p21-dendrites_smooth' 
dend_data = gdas.part(nam)  
 
  


model_type='vol_VGG16_FCN3D_5000_hpcc_fill'
model_type='vol_UNet3D_5000_hpcc_fill'
model_type='vol_UNet3D_5000_hpcc' 

model_type='vol_VoxNetSeg_5000_hpcc'
model_type='vol_2UNet3D2_5000_hpcc'
model_type='vol_VoxNetSeg_5000_hpcc_fill'

model_type='vol_3UNet3D3_5000_hpcc' 

nnbb=5000
model_type='vol_VGG16_FCN3D_5000_hpcc'

model_type='dnn_GINN_1'
model_type = 'pinn'
model_type = 'pinn_NN'
model_type=f'dnn_GINN_0'
model_type='dnn_GINN_1_AUG'
model_type='dnn_1GINN_SM_SMOOTH'
model_type='dnn_GINN_DICE_AUG'
model_type='dnn_GINN_AUG'
model_type='vol_UNet3D_DICE_AUG'
model_type='dnn_GINN_SMOOTH_AUG'
model_type='dnn_GINN_SMOOTH'
model_type='dnn_GINN_SMOOTH'
model_type='dnn_GINN_DICE_SMOOTH_AUG'
model_type='dnn_GINN_DICE_SMOOTH'
model_type='pinn_GINN_ORG'
model_type='vol_VoxNetSeg_DICE'
model_type='vol_VGG16_FCN3D_DICE'
model_type=f'vol_VGG16_FCN3D_{nnbb}_hpcc_crop'
model_type='vol_UNet3D_DICE'
model_type='vol_3UNet3D3_5000_hpcc'
model_type=f'vol_3UNet3D3_{nnbb}_hpcc_crop'
model_type=f'vol_VoxNetSeg_{nnbb}_hpcc_crop'
model_type='dnn_1GINN_SM'
model_type='vol_FastFCN3D_5000'
model_type='dnn_1GINN_SM_ORG_AUG'
model_type='vol_FastFCN3D_ORG_AUG'
model_type='vol_FastFCN3D_7000ORG_AUG'
model_type='dnn_1GINN_SM_ORG.0_AUG'
model_type='vol_FastFCN3D_5000ORG_AUG'

model_type='vol_VoxNetSeg_SM07000_AUG'
model_type='vol_UNet3D_SM07000_AUG'
model_type='vol_UNet3D_SM07000'
model_type='vol_UNet3D_NSM05000_AUG'
model_type='dnn_GINN_SM00000_LOC_AUG'
model_type='gcn_UNet_SM05000_LOC_AUG'
model_type='dnn_GINN_SM05000_LOC'
model_type='gcn_UNet_SM05000_LOC'
model_type='gcn_UNet_SM25000_LOC'
model_type='gcn_UNet_SM10000_LOC'
model_type='cml_cML'

dnn_mode = 'DNN-0'         # choose neural network type
dnn_mode = "DNN-4"         # choose neural network type
dnn_mode = 'DNN-3'         # choose neural network type
dnn_mode = 'DNN-1'         # choose neural network type  
dnn_mode = 'DNN-2'         # choose neural network type






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




path_heads_show=[
            # 'vol_VGG16_FCN3D_5000_hpcc',
            # 'vol_VoxNetSeg_5000_hpcc',
            # 'vol_UNet3D_5000_hpcc',
            # 'vol_VoxNetSeg_5000_hpcc_fill',
            # 'vol_2UNet3D2_5000_hpcc',
            # 'vol_3UNet3D3_5000_hpcc',
            f'vol_3UNet3D3_{nnbb}_hpcc_crop',
            f'vol_VGG16_FCN3D_{nnbb}_hpcc_crop',
            f'vol_VoxNetSeg_{nnbb}_hpcc_crop',
            # f'dnn_GINN_0',
            'dnn_GINN_DICE_SMOOTH',
            'dnn_GINN_DICE_SMOOTH_AUG',
            'vol_UNet3D_DICE',
            'pinn_GINN_ORG',
            # 'dnn_GINN_1', 
]
path_heads_show=path_heads_show if model_type in path_heads_show else path_heads_show+[model_type]
path_heads=[ 
    # 'ML','cML', 'unet3d','gcn_UNet','gcn_UNet_15000',
    #         'vol_VoxNetSeg_5000_hpcc_fill',
    #         'vol_VGG16_FCN3D_5000_hpcc_fill',
    #         'vol_UNet3D_5000_hpcc_fill',
    #         'vol_VGG16_FCN3D_5000_hpcc',
    #         'vol_VoxNetSeg_5000_hpcc',
    #         'vol_UNet3D_5000_hpcc',
    #         'pnet_PointNet_7000_0',
    #         'vol_2UNet3D2_5000_hpcc',
    #         'vol_3UNet3D3_5000_hpcc' ,
    #         'save',
            # 'pinn_NN',
            # 'dnn_GINN_AUG',
            # 'dnn_GINN_1', 
            # 'dnn_GINN_0', 
            # 'dnn_1GINN_SM',
            # 'dnn_1GINN_SM_SMOOTH',
            # 'dnn_GINN_DICE_AUG',
            # 'vol_UNet3D_DICE_AUG',
            # 'dnn_GINN_SMOOTH_AUG',
            # 'dnn_GINN_SMOOTH',
            # 'dnn_GINN_1_AUG',
            # 'dnn_GINN_DICE_SMOOTH_AUG',
            # 'vol_UNet3D_DICE',
            ] 
path_heads = list(set(path_heads+path_heads_show))
if model_type not in path_heads:
    path_heads.append(model_type) 

dnn_modes=['DNN-0','DNN-1','DNN-2' , 'DNN-3',]
dnn_modes += [dnn_mode] if dnn_mode not in dnn_modes else []

 
path_display = ['dest_shaft_path', 'dest_spine_path', ] 
path_display = ['dest_shaft_path', ]


# param = algorithm_param() 
# mapp = app_run_param(param)
# param=mapp.emerge_param()
n_step = 2       # smoothing step size'resize_10000','resize_10000_pca'
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
                    n_step = 0,
                    weight=weight,
                    size_threshold=size_threshold,
                    path_heads_show=path_heads_show,
                    path_heads=path_heads,
                    )
 
# if model_type.startswith(('vol')):
#     dict_param['param_dic']['data']['get_dend_name']=dict(dict_dend_path='old',
#                                     drop_dic_name='resize_5000') 
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

import random#
import numpy as np
entry_names=[None,]
ls =[1,2,3,4,]# trn.tolist()+random.sample(lss, int(len(lss) * 0.6))8,9,10,15,17


'''
tst=np.where(np.isin(dend_names, ['CNJVY_d003_RECON', 'CNJVY_d005_RECON']))[0]
trn=np.where(np.isin(dend_names, ['BTLNG_d004_RECON','BTLNG_d005_RECON','BTLNG_d010_RECON','CNJVY_d001_RECON']))[0]
lss=list(set([ii for ii in range(len(dend_names))])-set(tst)-set(trn) )
entry_names=[None,]
ls =[1,2,3,4,]# trn.tolist()+random.sample(lss, int(len(lss) * 0.6))8,9,10,15,17
if model_type.endswith(('AUG',)):
    uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),'name_all_0.txt'),dtype=str) 
    pom=dend_data['obj_org_path']
    tott=sum(len(os.listdir(os.path.join(f'{pom}_{ll}'))) for ll in uyu )
    print(uyu)
    entry_names.extend(uyu)
    print(entry_names)
    lss=list(set(range(tott))-set(ls))
    ls=np.unique(ls+random.sample(lss, int(len(lss) * 0.7)))
    print('[[[[',ls,) 

if model_type.startswith(('vol',)):
    entry_names=['resize_5000',]
    if model_type.endswith(('AUG',)):
        entry_names=[]
        # uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),'name_all_vol.txt'),dtype=str) 
        # uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),'smooth_resize_150000_pca_scaling_1_angle_1_tanslate_0_jitter_0.txt'),dtype=str) 
        # uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),f'{get_aug_path(model_type)}.txt'),dtype=str) 
        # pom=dend_data['obj_org_path']
        # tott=sum(len(os.listdir(os.path.join(f'{pom}_{ll}'))) for ll in uyu )

        uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),f'{get_aug_path(model_type)}.txt'),dtype=str)  
        tott=(len(uyu)+1)*len(dend_data['dend_names'])
        print(uyu)
        entry_names.extend(uyu)
        print(entry_names)
        lss=list(set(range(tott))-set(ls))
        ls=np.unique(ls+random.sample(lss, int(len(lss) * 0.7)))
        print('[[[[',ls,)
else:
    if model_type.endswith(('AUG',)):
        # uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),'smooth_resize_150000_pca_scaling_1_angle_1_tanslate_0_jitter_0.txt'),dtype=str) 
        # uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),f'{get_aug_path(model_type)}.txt'),dtype=str) 
        # # uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),'name_all_0.txt'),dtype=str) 
        # pom=dend_data['obj_org_path']
        # ll=uyu[0]
        # print(uyu)
        # tott=sum(len(os.listdir(os.path.join(f'{pom}_{ll}'))) for ll in uyu  )+len(os.listdir(os.path.join(f'{pom}_{ll}')))

        uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),f'{get_aug_path(model_type)}.txt'),dtype=str)  
        tott=(len(uyu)+1)*len(dend_data['dend_names'])
        entry_names.extend(uyu)
        print(entry_names)
        lss=list(set(range(tott-2))-set(ls))
        ls=np.unique(ls+random.sample(lss, int(len(lss) * 0.7)))
        print('[[[[',ls,len(ls),tott)
        print(max(ls),max(set(range(tott))-set(ls)))  
        uiiu= range(tott)
#         print([ uiiu[ii] for ii in ls],len)  

'''
if model_type.endswith(('AUG',)):
    uyu=np.loadtxt(os.path.join(os.path.dirname(dend_data['obj_org_path']),f'{get_aug_path(model_type)}.txt'),dtype=str)  
    tott=(len(uyu)+1)*len(dend_data['dend_names'])
    entry_names.extend(uyu)
    print(entry_names)
    lss=list(set(range(tott-2))-set(ls))
    ls=np.unique(ls+random.sample(lss, int(len(lss) * 0.7)))
    print('[[[[',ls,len(ls),tott)
    print(max(ls),max(set(range(tott))-set(ls)))  

# print(len(ls),tott,len(dend_data['dend_names']),os.path.dirname(dend_data['obj_org_path']),f'{get_aug_path(model_type)}.txt')
# lp
# jkset(range(tott))-set(ls),
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
