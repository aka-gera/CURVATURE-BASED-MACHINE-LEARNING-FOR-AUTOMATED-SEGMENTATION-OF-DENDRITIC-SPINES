import dash
from dash import html, dcc, Input, Output, State, callback 

import sys
import os
 
file_path_org=os.getcwd()  
sys.path.append(os.path.abspath(file_path_org))
  
# from app_run import app_run_param,algorithm 
from dend_fun_0.obj_get import parse_obj_upload
from  dend_fun_0.main_0 import app_run_param,algorithm,algorithm_param,get_data, get_data_all

name='pinn'
page_dir=f"/dsa-{name}"

page_name='DSA'
dash.register_page(
    __name__,
    title="DSA",
    name="DSA",
    path="/DSA-2",
    order=0
)
 
path_display = ['dest_shaft_path', ]
param = algorithm_param()  
mapp = app_run_param(param)
def layout():
    return mapp.app_layout
 
for gval in list(set(param['param_input']['param'])):
    @callback(
        Output(f"collapse-{gval}", "is_open"),
        Input(f"collapse-header-{gval}", "n_clicks"),
        State(f"collapse-{gval}", "is_open"),
        prevent_initial_call=True
    )
    def toggle_collapse(n_clicks, is_open, gval=gval):
        if n_clicks:
            return not is_open
        return is_open
 
 
@callback(
    mapp.Output,                             
    Input("run-button", "n_clicks"),           
    [State(idx, "value") for idx in mapp.Input_id] + [
        State("upload-data", "contents"),
        State("upload-data", "filename"),
        State("upload-data", "last_modified"),
        State("id-destination", "value") ,
    ],  
    prevent_initial_call=True
)



def update_output(n_clicks, *values):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    param=mapp.rebuild_param(values)

    *other_values, contents, filenames, last_modified, objs_path_org = values
    export_dir = os.path.dirname(objs_path_org)
    nam = os.path.basename(objs_path_org)
    print('[[[[[[[[[[[[- FILE NAMES -]]]]]]]]]]]]' ,filenames)
    dend_names_chld = {} 
    # objs_path_org = os.path.join(export_dir, nam)  # Data directory, e.g. file_path_data/nam/ 
    if isinstance(contents, list):
        dend_names=[]
        for content,filename in zip(contents,filenames): 
            filename, ext = os.path.splitext(filename)
            dend_path_original_new = os.path.join(objs_path_org,filename, 'data_org')  
            os.makedirs(dend_path_original_new, exist_ok=True)
            mesh=parse_obj_upload(content, filename, export_dir=dend_path_original_new )
            dend_names.append(filename) 
        dend_names_chld[nam] = dict(  
            dend_names=dend_names, 
            obj_org_path=objs_path_org,
            weights=[[10., 0.81]],   # ML shaft/spine parameter: decrease the first value to increase spine vertices
        )  
 
    else: 
        dend_names = [f for f in os.listdir(objs_path_org) if os.path.isdir(os.path.join(objs_path_org, f))]
        dend_names_chld[nam] = dict(  
            dend_names=dend_names, 
            obj_org_path=objs_path_org,
            weights=[[10., 0.81]],   # ML shaft/spine parameter: decrease the first value to increase spine vertices
        ) 

    gdas=get_data_all(names_dic=dend_names_chld,file_path_data=export_dir)
    dend_data = gdas.part(nam)  
    alg=algorithm(param=param,)
    result=alg.text(  
                dend_data = dend_data,   
                true_name='true_0', 
                dnn_mode=param['dnn_mode']['param'],
                model_type=param['path_head']['param'],
                path_display = path_display, 
                )

    return   



