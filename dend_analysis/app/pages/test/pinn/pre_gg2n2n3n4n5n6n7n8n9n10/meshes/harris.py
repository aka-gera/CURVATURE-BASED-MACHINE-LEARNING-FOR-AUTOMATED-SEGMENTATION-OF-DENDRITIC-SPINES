 

import os, sys ,dash 
file_path_org=os.getcwd()
sys.path.append(file_path_org ) 
from  dend_fun_0.app_param_test import app_param
from dash import callback  


dend_names = ['harris']
dend_namess = ['d000_', 'd000', 'd000']
dend_path_inits =['meshes']
data_studied = 'test'  
model_sufix = 'pre_gg2n2n3n4n5n6n7n8n9n10' 
path_train= {'data_true_iou': 'true_save_save', 'pinn_save_save': 'pinn_save_save', 'data_shaft_path': 'pinn_pre_gmg2m2_save', 'dest_shaft_path': 'pinn_pre_gg2n2n3n4n5n6n7n8n9n10_save', 'data_spine_path': 'pinn_pre_gmg2m2_nfull_sp', 'dest_spine_path': 'pinn_pre_gg2n2n3n4n5n6n7n8n9n10_nfull_sp', 'data_spine_path_pre': 'pinn_pre_gmg2m2_nfull_sp_pre', 'dest_spine_path_pre': 'pinn_pre_gg2n2n3n4n5n6n7n8n9n10_nfull_sp_pre', 'data_shaft_vertices_center_path': 'pinn_pre_gmg2m2_save', 'dest_shaft_vertices_center_path': 'pinn_pre_gg2n2n3n4n5n6n7n8n9n10_save', 'dest_spine_path_pre_init': 'pinn_save_save', 'data_train_path': 'pinn_pre_gmg2m2_nfull_sp_pre', 'data_spine_path_center': 'pinn_pre_gmg2m2_nfull_sp', 'dest_spine_path_center': 'pinn_pre_gg2n2n3n4n5n6n7n8n9n10_nfull_sp'}
pinn_dir_data='nfull'
dend_data={'dend_namess': [['d000_', 'd000', 'd000']], 'dend_names': ['harris'], 'dend_last': ['d000_'], 'dend_first': ['d000'], 'dend_path_inits': ['meshes'], 'name_spine_id': 'p_sps', 'name_head_id': 'p_sphs', 'name_neck_id': 'p_spns', 'name_shaft_id': 'p_spsh', 'obj_org_path': '/Users/akag/Desktop/CURVATURE-BASED-MACHINE-LEARNING-FOR-AUTOMATED-SEGMENTATION-OF-DENDRITIC-SPINES/meshes', 'obj_org_path_dict': {'true_0': '/Users/akag/Desktop/CURVATURE-BASED-MACHINE-LEARNING-FOR-AUTOMATED-SEGMENTATION-OF-DENDRITIC-SPINES/meshes'}, 'weights': [[10.0, 0.81]]}
index=0
model_type='pinn'
obj_org_path_dict={'true_0': '/Users/akag/Desktop/CURVATURE-BASED-MACHINE-LEARNING-FOR-AUTOMATED-SEGMENTATION-OF-DENDRITIC-SPINES/meshes'}
model_sufix_dic={'pre_gmg2m2': 'DNN-1', 'opt_gmg2m2_shkl': 'DNN-2', 'pre_gg2n2n3n4n5n6n7n8n9n10': 'DNN-3', 'save': 'save'}
mapp = app_param(
    file_path_org=file_path_org,
    model_sufix=model_sufix,
    path_train=path_train, 
    pinn_dir_data=pinn_dir_data,
    index=index, 
    data_studied=data_studied,  
    dend_data=dend_data,
    model_type=model_type,
    obj_org_path_dict=obj_org_path_dict,
    model_sufix_dic=model_sufix_dic,
)

title_dend_name = f'harris'
dash.register_page(__name__, title=title_dend_name, name=title_dend_name, order=0) 

def layout():
    return mapp.app_layout

@callback(
    mapp.Output,
    mapp.Input, 
    prevent_initial_call=mapp.prevent_initial_call
)
def update_output( path_head,model_suf,path,mode,dendd, clusts,  intensity_type, width, height,  templ,nbin,index): 
    figure,txt = mapp.Get_output( path_head,model_suf,path,mode,dendd, clusts,   intensity_type, width, height, templ,nbin,index)
    return figure ,txt
