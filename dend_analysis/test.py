 
 # 
import sys,os
DTYPE='float32'  
from dend_fun_0.main_0 import  app_run_param,algorithm,algorithm_param,get_data, get_data_all 
from dend_fun_0.get_path import get_name,get_configs  
 

file_path_org=os.getcwd()  
file_path_data=os.path.dirname(file_path_org)    # Change this directory to the path dataset  
gdas=get_data(file_path_data)    # Dataset Path
dend_names_chld={}
  

gdas = get_data(file_path_data)               # Dataset path
dend_names_chld = {}

 
nam='meshes'  
obj_list=[1,2]
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 

obj_org_path= os.path.join(file_path_data,nam  )

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
                weights=[[6.,0.81]],
                )  







# This dataset is located in file_path_data/meshes/d001/data_org/
# Mesh file: file_path_data/meshes/d001/data_org/d001.obj

gdas=get_data_all(names_dic=dend_names_chld,dend_data=gdas.dend_data,file_path_data=file_path_data)

# Choose the name of the data to test here we choose meshes
nam = 'meshes'          

gnam = get_name()      
dend_data = gdas.part(nam)  

model_type = 'ML'  
model_type = 'cML' 
model_type = 'pinn'


dnn_mode = 'DNN-3'         # choose neural network type 
path_display = ['dest_shaft_path', ]
path_display = ['dest_shaft_path', 'dest_spine_path',]


n_step = 2       # smoothing step size
weight=3          # Decrease for less spine mesh/Increase for more spine mesh
size_threshold=200   # mesh size threshold above which a spine is consider 


param = algorithm_param(n_step = n_step,weight=weight,size_threshold=size_threshold, )
mapp = app_run_param(param)
param=mapp.emerge_param()
param['smooth']['tf'] = True   #False   # False   # True/False: choose whether to smooth the data
param['annotations']['tf'] = True   # True/False: choose whether to generate annotations for training, accuracy, and recall
param['shaft_pred']['tf'] = True    # True/False: choose whether to predict shafts/spines
param['head_neck_segss']['tf'] =  False   # True   # True/False: choose whether to perform head/neck segmentation
param['iou']['tf'] = True       # True/False: choose whether to compute IoU
param['graph_center']['tf'] = True   # True/False: compute central axis
param['cylinder_heatmap']['tf'] =True   # False   # True/False: generate cylindrical heatmap
param['dash_pages']['tf'] = True          # True/False: generate Dash pages
param['clean_path_dir']['tf'] = False          # True/False: delete computed data
param['resize']['tf']=False #   True  # 

alg = algorithm(param)
alg.text(   
    dend_data = dend_data,  
    true_name = 'true_0', 
    dnn_mode = dnn_mode,
    model_type = model_type,  
    path_display = path_display, 
)
