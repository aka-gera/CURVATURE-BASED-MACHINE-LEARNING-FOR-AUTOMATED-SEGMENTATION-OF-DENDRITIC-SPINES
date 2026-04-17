 
 # 
import sys,os
DTYPE='float32'  
from dend_fun_0.main_0 import  app_run_param,algorithm,algorithm_param,get_data, get_data_all ,get_dict_param
from dend_fun_0.get_path import get_name,get_configs  
 

file_path_org=os.getcwd()  
file_path_data=os.path.dirname(file_path_org)    # Change this directory to the path dataset  
gdas=get_data(file_path_data)    # Dataset Path
dend_names_chld={}
  
 


nam_gen,action='neuropil_recon_000','resize' 
obj_list=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 16, 19, 20, 21, 22, 30, 31, 32, 59, 62, 63, 40, 113, 117] 
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]  

extraa=[f'{nam_gen}_{action}_{mm}' for mm in [5000,7000,15000]]
for nam in [nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_5000',f'{nam_gen}_resize_5000',f'{nam_gen}_smooth_resize_10000',f'{nam_gen}_smooth_resize_15000'  ]+extraa:
    obj_org_path= os.path.join(file_path_data,'neuropil_0', nam )
    dend_path_inits=[nam for _ in range(len(dend_names))]
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



nam_gen='meshes'  
obj_list=[1,2]
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]  

extraa=[f'{nam_gen}_{action}_{mm}' for mm in [1000,5000,7000,15000]]
for nam in [nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_5000',f'{nam_gen}_resize_5000',f'{nam_gen}_smooth_resize_10000'  ]+extraa:
    obj_org_path= os.path.join(file_path_data,nam, nam )
    dend_path_inits=[nam for _ in range(len(dend_names))]
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
for nam in [ nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_15000',f'{nam_gen}_smooth_resize_5000', ]+extraa:
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

# This dataset is located in

# This dataset is located in file_path_data/meshes/d001/data_org/
# Mesh file: file_path_data/meshes/d001/data_org/d001.obj

gdas=get_data_all(names_dic=dend_names_chld,dend_data=gdas.dend_data,file_path_data=file_path_data)

gnam = get_name()      
# Choose the name of the data to test here we choose meshes 
 

nam_gen,nam_loc,action,resize='p21-dendrites','p21-dendrites','smooth','resize_15000'
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000','smooth','resize_15000'
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000',None,'resize_15000'
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000',None,None
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000',None,'resize_5000'
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000','smooth',None  
 
 
 


 # Pick name for the model 
nnbb=5000 
model_type=f'vol_VGG16_FCN3D_{nnbb}_hpcc_crop' 
model_type=f'vol_3UNet3D3_{nnbb}_hpcc_crop'
model_type=f'vol_VoxNetSeg_{nnbb}_hpcc_crop'   
model_type='cml_cML'
model_type='gcn_UNet_SM10000_LOC' 
model_type='dnn_GINN_SM00000_LOC_AUG' 


# Pick feature type
dnn_mode = 'DNN-0'          
dnn_mode = 'DNN-1'          
dnn_mode = 'DNN-2'          
dnn_mode = 'DNN-3'          


   
# select weight 
weight=0.001
weight=0.5
weight=0.05 
weight=0.1
weight=13  
weight=16#  
weight=0.81  
weight=1   
weight=10   
weight=3    

if nam.startswith("p21"):
    size_threshold = 300
elif nam.startswith("cnn"):
    size_threshold = 10
else:
    size_threshold = 100

weight_pre=weight
path_dir_nam=f'dice'
path_dir=f'{path_dir_nam}_we_{weight}'
data_dir=f'{path_dir_nam}_we_3'
data_dir=path_dir
path_heads_show=[model_type,]

dict_param=get_dict_param(nam=nam,
                    n_step = 0,
                    weight=weight,
                    size_threshold=size_threshold,
                    path_heads_show=path_heads_show,
                    )
  
dict_param['model_type_data']=model_type 
param = algorithm_param(**dict_param)



dend_data = gdas.part(nam)  
mapp = app_run_param(param)
param=mapp.emerge_param()
param['dendrite_pred']['param']['size_threshold']=size_threshold
param['Resizing']['param']['target_number_of_triangles_faction']=150000
for val in ['get_pinn_features','get_skeleton','get_wrap','get_smooth']:
    param['dendrite_pred']['param']['param_dic']['tf_restart'][val]=False   #True   # 
param['dendrite_pred']['param']['param_dic']['tf_restart']['get_shaft_pred']=False   #True   #  
param['Smooth']['param']['n_step']=20


param['Smooth']['param']['dt']=1e-7
param['Smooth']['param']['method']='taubin' #'willmore'
param['Skeleton']['param']['path']="entry"     #'old'   #  
param['Skeleton']['param']['dict_mesh_to_skeleton_finder']['interval_target_number_of_triangles']=[200000]
param['annotations']['tf'] =False   #True   #  True/False: choose whether to generate annotations for training, accuracy, and recall
param['Spine-Shaft Segm']['tf'] =True    #False   #  True/False: choose whether to predict shafts/spines
param['skl_shaft_pred']['tf'] =False   #True  # 
param['Morphologic Param']['tf'] =False   #True   #    False   # True/False: choose whether to perform head/neck segmentation
param['iou']['tf'] =True       #False   # False   #  True/False: choose whether to compute IoU
param['roc']['tf'] =True       # False   # True/False: choose whether to compute IoU
param['graph_center']['tf'] = True   #False  # True   #False   #   True/False: compute central axis
param['cylinder_heatmap']['tf'] =True   # False   # False   # False   # True/False: generate cylindrical heatmap
param['dash_pages']['tf'] =True          #False   #  True/False: generate Dash pages
param['clean_path_dir']['tf'] = False          # True/False: delete computed data
param['intensity_rhs']['tf']=False #  True  #  
param['Resizing']['tf']= False # True  #  
param['Smooth']['tf'] =False   # True   # False   # True/False: choose whether to smooth the data
param['Skeleton']['tf']=True  #False #   
param['rhs']['tf']=  True  #  False #
param['model_shap']['tf']= True  #  False #
 
alg = algorithm(param)
alg.test(   
    dend_data = dend_data,  
    true_name = 'true_0', 
    dnn_mode = dnn_mode,
    model_type = model_type, 
    path_dir=path_dir,
    data_dir=data_dir,
    **dict_param 
)
