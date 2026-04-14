# -*- coding: utf-8 -*-
# Import required libraries

import sys
import os
 
import pickle
# import dash
from dash import dcc, html, dash_table, Input, Output, State, callback 
import dash_bootstrap_components as dbc
import numpy as np 

import time
import sys,os
file_path_org=os.getcwd()   
from dend_fun_0.help_smooth import get_smooth   
DTYPE='float32'  
   
from dend_fun_0.help_dendrite_pred import dendrite_pred 
from dend_fun_0.get_path import get_name,get_configs,get_path_train,get_data_mode  
from dend_fun_0.side_bar import  dnn_page

file_path_parent=os.path.dirname(file_path_org)
file_path_parent=os.path.join(file_path_parent,'meshes','meshes')
os.makedirs(file_path_parent, exist_ok=True) 
import subprocess
# def restart_apps(process="gunicorn"):
#     try: 
#         subprocess.run(["pkill", "-f", process], check=False) 
#         subprocess.Popen([
#             "python3", "-m", "gunicorn.app.wsgiapp",
#             "-w", "4",
#             "-b", "0.0.0.0:8050",
#             "wsgi:server",
#             "-c", "gunicorn.conf.py"
#         ])
#         return "Restart triggered!"
#     except Exception as e:
#         return f"Error restarting: {e}"
import subprocess
import platform

def restart_apps(process="gunicorn"):
    try:
        system = platform.system()

        if system == "Windows":
            # Kill any running waitress processes
            subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/T"], check=False)

            # Start waitress server
            subprocess.Popen([
                "python", "-m", "waitress", "--listen=0.0.0.0:8050", "wsgi:server"
            ])
            return "Restart triggered using Waitress (Windows)."

        else:
            # Kill any running gunicorn processes
            subprocess.run(["pkill", "-f", process], check=False)

            # Start gunicorn server
            subprocess.Popen([
                "gunicorn",
                "-w", "4",
                "-b", "0.0.0.0:8050",
                "wsgi:server",
                "-c", "gunicorn.conf.py"
            ])
            return "Restart triggered using Gunicorn (Linux/Unix)."

    except Exception as e:
        return f"Error restarting: {e}"





def get_dict_param(nam='meshes',
                    n_step = 20,
                    weight=3.,
                    size_threshold=100,
                    kmean_n_run=50,
                    path_heads_show=None,
                    path_heads=None,
                    ):
    path_heads_show=path_heads_show if path_heads_show is not None else [ 
                f'cnn_3UNet3D3_5000_hpcc_crop',
                f'cnn_VGG16_FCN3D_5000_hpcc_crop',
                f'cnn_VoxNetSeg_5000_hpcc_crop',  
                'dnn_GINN_SM00000_AUG', 
                'dnn_GINN_SM00000_LOC_AUG',
                'gcn_UNet_SM10000_LOC',
                'cml_cML',
    ]
    path_heads=path_heads if path_heads is not None else [    
                'save',  
                ] 


    path_display = ['dest_shaft_path', 'dest_spine_path', ] 
    path_display = ['dest_shaft_path', ]
 


    # path_heads_show=path_heads_show if model_type in path_heads_show else path_heads_show +[model_type] 
    path_heads = list(set(path_heads+path_heads_show))
    dnn_modes=['DNN-0','DNN-1','DNN-2' , 'DNN-3',]
    # if model_type not in path_heads:'DNN-5',
    #     path_heads.append(model_type)
    # if dnn_mode not in dnn_modes:
    #     dnn_modes.append(dnn_mode)

    path_list=['shaft_path','spine_path',  ]   
    weig=sorted(list(set([0.001,0.05,0.1,0.5,0.81,1,2, 3,6,10,13,16,20]  )))
    path_dirs=[f'dice_we_{wei}' for wei in weig] 
    path_dirs_weig={key:val for key,val in zip(path_dirs,weig)}
    weigg=[ 3,10] 

    # for hh in ['iou', 'loss', 'auc']:
    #     for wei in weigg:
    #         key = f'{hh}_we_{wei}'
    #         path_dirs.append(key)
    #         path_dirs_weig[key] = wei

    # path_dirs=[f'pa_we_{wei}' for wei in  sorted(list(set([0.2,.5,0.81,1,3,4,6,10,16] +[weight])))]+[weight]

    path_display_dic={
                        'path':path_dirs,
                        'model_sufix':[],
                        'path_head':[],
                    } 
    param_dic={
        hh:{
            kk:True for kk in ['get_pinn_features','get_wrap','get_scale',
            'get_smooth','get_shaft_pred','get_dend_name','get_skeleton', ]
        }
        for hh in ['tf_restart','get_info','data']
    } 

    param_dic['data']['get_dend_name']=dict(
                        dict_dend_path='current',
                        drop_dic_name=None,)





    



    return dict(   
        # dend_data = dend_data,  
        # true_name = 'true_0', 
        # dnn_mode = dnn_mode,
        # model_type = model_type, 
        # path_dir=path_dir,
        # data_dir=data_dir, 
        nam=nam,
        n_step = n_step,
        weight=weight,
        size_threshold=size_threshold,
        path_display = path_display, 
        model_type_data=None,  
        path_display_dic=path_display_dic,
        path_heads_show=path_heads_show,
        # path_shaft_dir=path_shaft_dir,
        param_dic=param_dic,
        kmean_n_run=kmean_n_run,
        dnn_modes=dnn_modes,
        path_dirs=path_dirs,
        path_list=path_list,
        path_heads=path_heads,
        path_dirs_weig=path_dirs_weig,
    )

    





def algorithm_param(nam='meshes',
                    n_step = 200,
                    weight=3.,
                    size_threshold=100,
                    path_heads=None, 
                    dnn_modes=None,
                    path_dirs=None,
                    path_list=None,
                    path_display=None,
                    tf_restart=False,
                    kmean_n_run=50,
                    kmean_max_iter=600,
                    param_dic=None,
                    path_dir=None,
                    data_dir=None,  
                    model_type_data =None,
                    path_display_dic=None,
                    path_heads_show=None,
                    path_shaft_dir=None,
                    path_dirs_weig=None,
                    # path_display_dic=None,
                    ):
    if param_dic is None: 
        param_dic={
            hh:{
                kk:True for kk in ['get_pinn_features','get_wrap','get_scale','get_smooth','get_skeleton']
            }
            for hh in ['tf_restart','get_info']
        }
    if path_heads is None:
        path_heads=['pinn', ]
        path_heads=['pinn','rpinn','ML','pinn_new','unet3d']
    if dnn_modes is None:
        dnn_modes=['DNN-0','DNN-1','DNN-2' , 'DNN-3', 'DNN-5']
    if path_list is None:
        path_list=['shaft_path','spine_path',  ] 
    if path_dirs is None:
        path_dirs=['save',  ] 
    if path_display is None:
        path_display=['dest_shaft_path','dest_spine_path',]
        path_display=['dest_spine_path_pre','dest_spine_path','dest_shaft_path']
   
    # print('[[[[[[[[[[[[---------------------------000--------]]]]]]]]]]]]')
    # print(param_dic['data']['get_dend_name'])

  
    smooth_tf=True
    smooth_tf=False
    data_studied='train'  
    
    data_studied='test' 
    skip_first_n=1
    end_thre=15  
    subdivision_thre=2
    zoom_thre=50
    skip_end_n=14
    skip_mid_n=5 
    thre_target_number_of_triangles=10
    voxel_resolution=128
    tres=['myles','p21-dendrites',nam]


    line_num_points_shaft=300
    line_num_points_inter_shaft=400 if nam in tres else 80 

    line_num_points_shaft=200
    line_num_points_inter_shaft=150 if nam in tres else 80 

    spline_smooth_shaft=1
    subdivision_thre=0 if nam in tres else 2
    zoom_thre=150 if nam in tres else 100
    skip_end_n=20 if nam in tres else 200
    skip_end_n=80 if nam in tres else 200
    skip_end_n=250 if nam in tres else 2550
    skip_mid_n=10 if nam in tres else 60
    skip_mid_n=5 if nam in tres else 10000
    # size_threshold=400 if nam in tres else 60

    thre_target_number_of_triangles=10
    voxel_resolution = 521
    voxel_resolution = 256
    voxel_resolution = 128
    neck_lim=2


    zoom_thre=50
    skip_first_n=1
    skip_end_n=14
    subdivision_thre=2
 



    dict_mesh_to_skeleton_finder_mesh=dict( 
                interval_voxel_resolution=[100 ], 
                interval_target_number_of_triangles=[200000], 
                tf_largest=False,
                disp_infos=True, 
                min_voxel_resolution=200,
                min_target_number_of_triangles=1000,
                tf_division=True, 
                alpha_fraction=.8,
                offset_fraction=0.800,
                wrap_method='alpha_wrap',
                    )

    param_get_segments=dict(
                            thre_distance_min=1.75 ,
                            thre_distance_max=1.5,
                            thre_explained_variance_ratio=0.97,
                            size_threshold=size_threshold, 
                            tf_merge=False, 

    )
 
    tf_skl_shaft_distance=True 
    dict_mesh_to_skeleton_finder=dict( 
            interval_voxel_resolution=[50],
            interval_target_number_of_triangles=None,
            tf_largest=False,
            disp_infos=False, 
                )

    dict_wrap=dict(  
                number_of_points=5000,
                radius=0.1, 
                max_nn=30,
                )


    model_type='rpinn'
    model_type='ML' 
    model_type='pinn_old'
    model_type='pinn' 
    tf_smooth_data=True
    a,b,n=3,10,6 
    weights=[np.array([nn,.81]) for nn,mm in zip(np.arange(a,b,(b-a)/n),np.arange(a,b,(b-a)/n))] 
    weights=[[3,.81]] 
    
 
    param_clean=[ 'all', 'result']+path_heads_show

    param_dropdown=['path_heads', 'dnn_modes','path_dirs', ]
    param_dropdowni=['path_head', 'dnn_mode','path_dir']
    param_input=['Smooth','Resizing','Spine-Shaft Segm', 'Morphologic Param','model_shap','clean_path_dir', ]
    param_inputbv=[ 'intensity_rhs','intensity_all','iou','roc','graph_center','dash_pages', 'cylinder_heatmap' ,'annotations',  'dendrite_pred' ,'rhs','skl','Skeleton',  ]
    param_input_save=[ 'spines_segss_old','spines_segss_old_2', 'clean_path_dir','get_training','skl_shaft_pred']
    param_list=['param_dropdown','param_input','param_all', 'param_list','param_dropdowni','param_inputbv','path_list', 'path_dirs_weig']
    param_all=[]
    param_all.extend(param_dropdown)
    param_all.extend(param_input)
    param_all.extend(param_inputbv)
    param_all.extend(param_input_save)
    param_all.extend(param_list)
    param_all.extend(param_dropdowni)
    param_all.extend(param_clean) 

    param={key:
                {
                    'tf':{},
                    'param':{},
                    'param_fix':{},
                    'option':[],
                }
            for key in param_all
            }
    param['param_list']['param']=param_list
    param['path_heads']['param']=path_heads_show 
    param['path_dirs']['param']=path_dirs
    param['path_list']['param']=path_list
    param['param_dropdown']['param']=param_dropdown
    param['param_dropdowni']['param']=param_dropdowni
    param['param_input']['param']=param_input
    param['param_inputbv']['param']=param_inputbv
    param['param_all']['param']=param_all 
    param['dnn_modes']['param']=dnn_modes
    param['path_head']['param']=path_heads_show[0]
    param['dnn_mode']['param']=dnn_modes[0] 
    param['path_dir']['param']=path_dirs[0]
    param['path_dirs_weig']['param']=path_dirs_weig


    
    param['Smooth']['param']=dict(
                                get_data=True, 
                                n_step = n_step, 
                                method=['willmore','taubin',],
                                dt=1e-6, )
    param['Smooth']['param_fix']=dict(  
                                disp_time=500,  )
    param['Smooth']['tf']=tf_smooth=False#True 

    param['annotations']['tf']= False#True



    param['clean_path_dir']['param']=  dict(  
                                        path_clean=param_clean, 
                                        )

    param['intensity_all']['param']=dict(  
                                        thr_gauss=45,
                                        thr_mean=15,
                                        )
    param['intensity_all']['tf']= False


    param['intensity_rhs']['param_fix']=dict(   
						radius_threshold=None, 
                                        )
    param['intensity_rhs']['tf']= False

    param['model_shap']['param_fix']=dict(  
                                    dend_names_ls=[0,1],
                                    n_shap=25, 
                                    ) 


    rl_par= 0.7
    dnn_par= 1-rl_par
    mj=np.arange(1,1000,200)/1000
    weightt=np.array([ mj,mj[::-1]]).T
    param['get_training']['param']=dict( 
                                        full_dend='nfull', 
                                        get_training=True,

                                        line_num_points_shaft=line_num_points_shaft,
                                        line_num_points_inter_shaft=line_num_points_inter_shaft,  
                                        itime = 1500,
                                        itime_div=100, 
                                        ls=[0,1,2,3],
                                        dest_path='data_shaft_path',
                                        weight=weightt, 
                                        num_sub_nodes=None,
                                        rl_par= rl_par, 
                                        dnn_par= dnn_par,
                                        l1_values = [0,   1e-6,   1e-4, 1e-2],
                                        l2_values = [0,   1e-6,   1e-4, 1e-2], 
                                        )
    param['Spine-Shaft Segm']['param']=dict( 
                                      size_threshold=size_threshold,
                                    )   
    param['Skeleton']['param']=dict( weight=weight,
                                        size_threshold=size_threshold,
                                        path='entry',
                                        wrap_part='shaft_wrap' , 
                                        weights=weights, 
                                        shaft_thre=0.9, 
                                        smooth_tf=tf_smooth,
                                        neck_lim=neck_lim,
                                        dict_wrap=dict_wrap,
                                        dict_mesh_to_skeleton_finder=dict_mesh_to_skeleton_finder_mesh, 
                                        tf_skl_shaft_distance=tf_skl_shaft_distance,
                                        tf_restart=tf_restart,

                                    )   
    param['Spine-Shaft Segm']['param_fix']=dict(
                                        weight=weight,
                                        # train_spines=train_spines,
                                        # model_type=model_type,
                                        # # # 
                                        weights=weights, 
                                        shaft_thre=0.9, 
                                        smooth_tf=tf_smooth,
                                        neck_lim=neck_lim,
                                        dict_wrap=dict_wrap,
                                        dict_mesh_to_skeleton_finder=dict_mesh_to_skeleton_finder_mesh, 
                                        tf_skl_shaft_distance=tf_skl_shaft_distance,
                                    )   




    param['Resizing']['param']=dict(  
                                        # min_target_number_of_triangles_faction=600000,
                                        target_number_of_triangles_faction=200000,
                                                )
    param['Resizing']['param_fix']=dict( 
                                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                                        voxel_resolution=voxel_resolution,
                                        annotation_resized_train_tf=False, 
                                        nam=nam,
                                                )

    param['skl_shaft_pred']['tf']= True

    param['spines_segss_old']['param_fix']=dict( 
                                        seg_dend='nfull', 
                                        get_refine_=False,
                                        f=3.,
                                        zoom_thre=zoom_thre ,
                                        skip_first_n=skip_first_n,
                                        skip_mid_n=skip_mid_n,
                                        skip_end_n=skip_end_n,
                                        subdivision_thre=subdivision_thre,
                                        end_thre=end_thre,
                                        size_threshold=size_threshold, 
                                        spine_fraction=0.20,
                                        ctl_run_thre=0,
                                        spline_smooth_shaft=spline_smooth_shaft ,
                                        smooth_tf=smooth_tf, 
                                        )
    param['spines_segss_old']['tf']= True


    param['spines_segss_old_2']['param_fix']=dict( 
                                        seg_dend='nfull', 
                                        get_refine_=False,
                                        f=3.,
                                        zoom_thre=zoom_thre ,
                                        skip_first_n=skip_first_n,
                                        skip_mid_n=skip_mid_n,
                                        skip_end_n=skip_end_n,
                                        subdivision_thre=subdivision_thre,
                                        end_thre=end_thre,
                                        size_threshold=size_threshold, 
                                        spine_fraction=0.20,
                                        ctl_run_thre=0,
                                        spline_smooth_shaft=spline_smooth_shaft ,
                                        smooth_tf=smooth_tf, 
                                        param_get_segments=param_get_segments,
                                        )

    param['Morphologic Param']['param_fix']=dict(  
                                get_refine_=False,
                                f=3.,
                                zoom_thre=zoom_thre,
                                skip_first_n=skip_first_n,
                                skip_end_n=skip_end_n,
                                subdivision_thre=subdivision_thre,
                                end_thre=45,
                                size_threshold=size_threshold,
                                # head_neck_path='dest_spine_path_pre',
                                ctl_run_thre=0,
                                spline_smooth=spline_smooth_shaft , 
                                seg_dend='nfull',
                        ) 

    param['dendrite_pred']['param_fix']=dict(
                                        file_path_org=file_path_org,
                                        data_studied=data_studied, 
                                        radius_threshold=0.9,
                                        size_threshold=size_threshold, 
                                        disp_infos=True,
                                        spline_smooth_shaft=spline_smooth_shaft,
                                        line_num_points_shaft=line_num_points_shaft,
                                        line_num_points_inter_shaft=line_num_points_inter_shaft, 
                                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                                        voxel_resolution=voxel_resolution, 
                                        path_display=path_display,
                                        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                        kmean_n_run=kmean_n_run,
                                        kmean_max_iter=kmean_max_iter,
                                        param_dic=param_dic,
                            )
    param['dendrite_pred']['tf']=True 
    param['model_shap']['tf']=False

    param['Spine-Shaft Segm']['tf']=True 
    param['spines_segss_old']['tf']=False
    param['skl_shaft_pred']['tf']=False
    param['spines_segss_old_2']['tf']= False
    param['Morphologic Param']['tf']= True
    param['Resizing']['tf']=False
    param['skl']['tf']=True
    param['Skeleton']['tf']=True
    param['rhs']['tf']=True
    param['iou']['tf']=True
    param['roc']['tf']=True
    param['graph_center']['tf']=True
    param['cylinder_heatmap']['tf']=True
    param['dash_pages']['tf']=True 
    param['clean_path_dir']['tf']= False

    return param
 



styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
box_style = {
    'width': '100%',
    'padding': '3px',
    'font-size': '20px',
    'text-align-last': 'center',
    'margin': 'auto', 
    'background-color': 'black',
    'color': 'black'
} 
dropdown_style = {
    'width': '85%',
    'padding': '3px',
    'font-size': '20px',
    'text-align-last': 'center',
    # 'margin': 'auto', 
    'marginLeft': 'auto',
    # 'background-color': 'black',
    'color': 'black',
    # 'height': '10px',
} 
dropdown_options_style = {'color': 'white', 'background-color': 'gray'} 




class app_run_param:
    def __init__(self,param):
        self.param=param
        self.Input=[]
        self.Input_id=[]
        self.Input_id_coll_head=[]
        self.Input_id_coll=[]
        self.Output=[]
        self.graph=[]
        self.path_head=param['path_head']['param']
        self.dnn_mode=param['dnn_mode']['param']
        self.path_dir=param['path_dir']['param'] 
        self.path_heads=param['path_heads']['param'] 
        self.dnn_modes=param['dnn_modes']['param']
        self.path_dirs=param['path_dirs']['param'] 
        self.path_list=param['path_list']['param']
        self.Input_id.append('upload-data')

 
        for gval,gvali in zip(param['param_dropdown']['param'],param['param_dropdowni']['param']):
            # print(gval,param[gval]['param'] )
            dropdown_option = [
                {'label': val, 'value': val, 'style': dropdown_options_style}
                for   val in param[gval]['param'] 
            ]
            # idx=f'dropdown_{gvali}_param'
            # self.Input_id.append(idx)
            # param[gvali]['option'] = dcc.Dropdown(
            #     options=dropdown_option,
            #     id=idx,
            #     value=dropdown_option[0]['value'],   # default to first value
            #     placeholder=f'Select {gvali}',
            #     style=box_style
            # )
            idx=f'dropdown_{gvali}_param'
            self.Input_id.append(idx)
            param[gval]['option'] = dcc.Dropdown(
                options=dropdown_option,
                id=idx,
                value=dropdown_option[-1]['value'],   # default to first value
                placeholder=f'Select {gval}',
                style=box_style
            )
            # self.graph.append(param[gval]['option'])
            self.graph.append(
                dbc.Col(
                    param[gval]['option'],
                    # xs=12, sm=6, md=4, lg=3,  # full width on mobile, 2 per row on small, 3 per row on medium, 4 per row on large
                     xs=12, sm=6, md=4, lg=3, 
                    #width=3,
                    
                    style={    "width": "100%" }
                    # style={'borderRight': '1px solid #ccc', 'paddingRight': '15px', "width": "100%" }
                                )
            )
  



        for gval in param['param_input']['param'] : 
            self.Input_id.append(f"param_{gval}_tf")
            self.Input_id_coll_head.append(f"collapse-header-{gval}")
            self.Input_id_coll.append(f"collapse-{gval}")
            for k, v in param[gval]['param'].items(): 
                self.Input_id.append(f"param_{gval}_{k}") 
            param[gval]['option'] = dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.H5(f"{gval}",
                                    id=f"collapse-header-{gval}",
                                    className="card-title", 
                                    style={"cursor": "pointer"}), 
                            width=9,
                            style={'display': 'flex', 'alignItems': 'center'}
                        ),
                        dbc.Col(
                            dbc.Checkbox(
                                id=f"param_{gval}_tf",
                                value=param[gval]['tf'],
                                label="Enabled"
                            ),
                            width=3,
                            style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'center'}
                        )
                    ], style={'marginBottom': '10px'}),
 
                    html.Hr(style={
                        "borderTop": "1px solid #ccc",
                        "marginTop": "0.5rem",
                        "marginBottom": "1rem"
                    }), 
                    dbc.Collapse(
                        html.Div([
                            dbc.Row([
                                dbc.Col(html.Label(k, style={'textAlign': 'right'}), width=3),
                                dbc.Col(
                                    # dbc.Checkbox(
                                    #     id=f"param_{gval}_{k}",
                                    #     value=v,
                                    #     label="Enabled"
                                    # ) if isinstance(v, bool) else dcc.Input(
                                    #     id=f"param_{gval}_{k}",
                                    #     type="text",
                                    #     value=v,
                                    #     style={'width': '75px', 'marginLeft': 'auto'}  
                                    # ),
                                    # comp = (
                                        dbc.Checkbox(
                                                    id=f"param_{gval}_{k}", 
                                                    value=v, 
                                                    label="Enabled",
                                        ) if isinstance(v, bool) else
                                        dcc.Input(
                                            id=f"param_{gval}_{k}",
                                            type="text",
                                            value=v,
                                            style={'width': '75px', 'marginLeft': 'auto'}
                                        ) if isinstance(v, int) or isinstance(v, float) else
                                        dcc.Dropdown(
                                            options=[{'label': val, 'value': val, 'style': dropdown_options_style} for val in list(v)],
                                            id=f"param_{gval}_{k}",
                                            value=list(v)[0],
                                            placeholder=f"Select {gval} {k}",
                                            style=dropdown_style,

                                        ),
                                    # )

                                    width=9,
                                    className="text-end"  # right-align inside column
                                )
                            ], style={'marginBottom': '10px'})
                            for k, v in param[gval]['param'].items()
                        ]),
                        id=f"collapse-{gval}",
                        is_open=False
                    ),
                ]),
                style={"marginBottom": "0px", "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"}
            ) 
            self.graph.append(
                dbc.Col(
                    param[gval]['option'],
                    xs=12, sm=6, md=4, lg=3, 
                   #  width=3,
 
                    style={'marginBottom': '0px', "width": "100%" },
                )
            )  
            # self.graph.append( 
            #         param[gval]['option'], 
            # )  

    
        
        self.Input=[Input(idx, 'value') for idx in self.Input_id] 
        self.prevent_initial_call=True 

        self.app_layout = html.Div([
            dcc.Store(id="shared-data", storage_type="memory", clear_data=True, data={}),

            # Wrap the whole thing in a Card
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        # Left column
                        dbc.Col(
                            [
                                # Buttons row
                                dbc.Row([
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Run", 
                                                id="run-button", 
                                                n_clicks=0, 
                                                color="primary",
                                               #  color="success",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Reset", 
                                                id="reset-button", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Restart", 
                                                id="restart-button", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                            # style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Off", 
                                                id="shutdown-button", 
                                                n_clicks=0,
                                                color="primary",
                                               #   color="danger",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px" }
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                ], 
                                justify="center",   # centers the columns horizontally
                                align="center",     # centers vertically inside the row
                                # className="g-0",    # removes gutter spacing between columns
                                style={"marginTop": "20px"}

                                ),

                                # Destination input card 
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            dbc.Row([ 
                                                    dcc.Markdown('Add Destination', style={'textAlign': 'center', 'color': 'white'}),
                                                    html.Hr(),
                                                    dcc.Input(
                                                        id="id-destination",
                                                        type="text",
                                                        value=file_path_parent,
                                                        style={'width': '100%' }
                                                    )
                                            ], style={'marginBottom': '10px'}),
                        
                                            # html.Hr(style={
                                            #     "borderTop": "1px solid #ccc",
                                            #     "marginTop": "0.5rem",
                                            #     "marginBottom": "1rem"
                                            # }),  
                                        ]),
                                        style={"marginBottom": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"}
                                    ) ,
                                xs=12, sm=6, md=4, lg=3,  
                                style={'marginBottom': '0px', "width": "100%" },
                                ),
                                # Parameter cards
                                *self.graph
                            ],
        className="g-0"  ,
                            width=4,
                            style={"padding": "10px", "borderRight": "1px solid #ccc"}
                        ),

                        # Right column
                        dbc.Col(
                            [
                                dbc.Row([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('# Dendritic Spines Analysis',
                                                            style={'textAlign': 'center', 'color': 'white'})
                                            )
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    )
                                ], justify="start"),

                                # Upload card
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                dcc.Markdown('### Upload File', style={'textAlign': 'center', 'color': 'white'}),
                                                html.Hr(),
                                                dcc.Upload(
                                                    id='upload-data',
                                                    children=html.Div([
                                                        'Drag and Drop or ',
                                                        html.A('Select Files')
                                                    ]),
                                                    style={
                                                        'width': '100%',
                                                        'height': '60px',
                                                        'lineHeight': '60px',
                                                        'borderWidth': '1px',
                                                        'borderStyle': 'dashed',
                                                        'borderRadius': '5px',
                                                        'textAlign': 'center',
                                                        'margin': '10px',
                                                        'color': 'white'
                                                    },
                                                    multiple=True
                                                ),
                                                html.Div(id='output-file-upload', style={'color': 'white', 'marginTop': 20}),
                                                html.Hr(style={'borderTop': '2px dashed white', 'margin': '20px 0'}),
                                                dbc.Card([
                                                    html.H3("Press Run to Start",
                                                            id="run-status",
                                                            style={'color': 'lightgreen', 'textAlign': 'center'})
                                                ]),
                                            ]),
                                            style={
                                                "margin": "10px",
                                                "padding": "10px",
                                                "backgroundColor": "#2c2c2c",
                                                "width": "100%"
                                            }
                                        ),
                                        width='11'
                                    )
                                ], justify="center", align="center"),

                                # Parameters & Results cards
                                dbc.Row([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('### Parameters & Options',
                                                            style={'textAlign': 'center', 'color': 'white'}),
                                                id="toggle-text-starting",
                                                n_clicks=0,
                                                style={"cursor": "pointer"}
                                            ),
                                            html.Hr(),
                                            dbc.Collapse(
                                                dcc.Markdown(dnn_page()['starting']),
                                                id="collapse-starting",
                                                is_open=False
                                            ),
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    ),
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('### Checking Results After Segmentation',
                                                            style={'textAlign': 'center', 'color': 'white'}),
                                                id="toggle-text",
                                                n_clicks=0,
                                                style={"cursor": "pointer"}
                                            ),
                                            html.Hr(),
                                            dbc.Collapse(
                                                html.Div([
                                                    dcc.Markdown(dnn_page()['restart']),
                                                    dcc.Markdown(dnn_page()['results']),
                                                ]),
                                                id="collapse-result",
                                                is_open=False
                                            ),
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    ),
                                ], justify="start"),

                                # GitHub link
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Markdown(
                                            "[View the full repository on GitHub](https://github.com/aka-gera/CURVATURE-BASED-MACHINE-LEARNING-FOR-AUTOMATED-SEGMENTATION-OF-DENDRITIC-SPINES)",
                                            style={"textAlign": "center", "color": "white"}
                                        )
                                    )
                                ], justify="center")
                            ],
                            width=8,
                            style={"padding": "10px"}
                        )
                    ])
                ]),
                style={
                    "margin": "20px",
                    "padding": "20px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.2)",
                    "backgroundColor": "#1e1e1e",
                    "maxWidth": "95%",   # scale control
                    "marginLeft": "auto",
                    "marginRight": "auto"
                }
            )
        ])

    def rebuild_param(self,*values):
        param=self.param 
        updated = dict(zip(self.Input_id, values[0])) 
        new_param={key:
                    {
                        'tf':{},
                        'param':{},
                        'option':[],
                    }
                for key in param['param_all']['param']
                } 
         

        for gval in param['param_input']['param']:  
            new_param[gval]['tf'] = updated[f"param_{gval}_tf"] 
            for k in param[gval]['param_fix'].keys():
                new_param[gval]['param'][k] = param[gval]['param_fix'][k] 

            for k in param[gval]['param'].keys():
                val = updated[f"param_{gval}_{k}"]   

                if isinstance(param[gval]['param'][k], bool):
                    new_param[gval]['param'][k] = bool(val)
                else:
                    try: 
                        new_param[gval]['param'][k] = float(val)
                    except (ValueError, TypeError):
                        try:
                            new_param[gval]['param'][k] = float(val)
                        except (ValueError, TypeError):
                            new_param[gval]['param'][k] = val  
        for gval in param['param_dropdowni']['param']: 
            new_param[gval]['param'] = updated[f"dropdown_{gval}_param"] 

        for gval in param['param_dropdown']['param']:  
            new_param[gval]['param'] = param[gval]['param']

        for gval in param['param_list']['param']:  
            new_param[gval]['param'] = param[gval]['param']

        for gval in param['param_inputbv']['param']:  
            new_param[gval]['tf'] = param[gval]['tf']
            for k in param[gval]['param_fix'].keys():
                new_param[gval]['param'][k] = param[gval]['param_fix'][k] 

            for k in param[gval]['param'].keys():
                val = param[gval]['param'][k]  

                if isinstance(param[gval]['param'][k], bool):
                    new_param[gval]['param'][k] = bool(val)
                else:
                    try: 
                        new_param[gval]['param'][k] = float(val)
                    except (ValueError, TypeError):
                        try:
                            new_param[gval]['param'][k] = float(val)
                        except (ValueError, TypeError):
                            new_param[gval]['param'][k] = val  
                            
  
        return new_param



    def emerge_param(self, ):
        param=self.param  
        new_param={key:
                    {
                        'tf':{},
                        'param':{},
                        'option':[],
                    }
                for key in param['param_all']['param']
                } 
         

        for gval in param['param_input']['param']:  
            new_param[gval]['tf'] = param[gval]['tf']
            for k in param[gval]['param_fix'].keys():
                new_param[gval]['param'][k] = param[gval]['param_fix'][k] 

            for k in param[gval]['param'].keys():
                val = param[gval]['param'][k]  

                if isinstance(param[gval]['param'][k], bool):
                    new_param[gval]['param'][k] = bool(val)
                else:
                    try: 
                        new_param[gval]['param'][k] = float(val)
                    except (ValueError, TypeError):
                        try:
                            new_param[gval]['param'][k] = float(val)
                        except (ValueError, TypeError):
                            new_param[gval]['param'][k] = val  

        for gval in param['param_inputbv']['param']:  
            new_param[gval]['tf'] = param[gval]['tf']
            for k in param[gval]['param_fix'].keys():
                new_param[gval]['param'][k] = param[gval]['param_fix'][k] 

            for k in param[gval]['param'].keys():
                val = param[gval]['param'][k]  

                if isinstance(param[gval]['param'][k], bool):
                    new_param[gval]['param'][k] = bool(val)
                else:
                    try: 
                        new_param[gval]['param'][k] = float(val)
                    except (ValueError, TypeError):
                        try:
                            new_param[gval]['param'][k] = float(val)
                        except (ValueError, TypeError):
                            new_param[gval]['param'][k] = val  

        for gval in param['param_dropdowni']['param']: 
            new_param[gval]['param'] = param[gval]['param'] 

        for gval in param['param_dropdown']['param']:  
            new_param[gval]['param'] = param[gval]['param']

        for gval in param['param_list']['param']:  
            new_param[gval]['param'] = param[gval]['param']
 
  
        return new_param






class algorithm:
    def __init__(self, 
                param,  
                true_name='true_0',
                pre_portion='spine', 
                path_heads_true=['true',]
                ): 
        self.path_heads=param['path_heads']['param'] 
        self.path_dirs=param['path_dirs']['param'] 
        self.dnn_modes=param['dnn_modes']['param']
        self.path_list=param['path_list']['param'] 
        self.param=param  
        self.true_keys=[true_name]
        self.pre_portion=pre_portion
        pre_portion=self.pre_portion
        path_list=self.path_list 
        self.path_heads_true=path_heads_true









    def pre_test_train( self, 
            path_heads_show,
            model_sufix_show, 
            path_dirs_show,
            dend_data=None,   
            train_test='test',
            true_name='true_0', 
            dnn_mode='mode0', 
            model_type='pinn' , 
            path_dir='save',
            path_display=None, 
            size_threshold=None,
            path_display_dic=None, 
            path_shaft_dir=None,
            model_type_data=None,
            data_dir=None,

            ): 
        pre_portion=self.pre_portion
        path_list=self.path_list
        model_type_data=model_type_data if model_type_data is not None else model_type
        path_heads_show_data=[model_type_data]
        modes={
                val:{  
                        name:{
                                dirs:{} 
                                for dirs in path_dirs_show
                            } 
                            for name in model_sufix_show+['mode0']+["DNN-4"] 
                        }  
                    for val in path_heads_show
                }
        configs=get_configs() 

        dmode = get_data_mode(pre_portion=pre_portion,path_list=path_list) 
        for val in path_heads_show:   
                ids,name,dirs=0,'mode0',path_dirs_show[0]
                data_dir_tmp=data_dir if data_dir is not None else dirs
                data_mode=dmode.test_pre(pre_portion=pre_portion,
                                            data_head=model_type_data,
                                            dest_head=val,
                                            seg_dend=dirs,
                                            data_dir=data_dir_tmp,
                                            dest_dir=dirs,
                                            )
                modes[val][name][dirs]=[dmode.mode_id] 
                for dirs in path_dirs_show:
                    data_dir_tmp=data_dir if data_dir is not None else dirs
                    for ids, name in enumerate(model_sufix_show):
                        # print(f"Finished {name}, dnn_modes={self.dnn_modes}")
                        cfg=configs[name]
                        cfg["data_sufix"]=cfg["dest_sufix"]=None
                        data_mode = dmode.test_opt( 
                                                pre_portion=pre_portion, 
                                                train_test=train_test,
                                                data_head=model_type_data,
                                                dest_head=val,
                                                seg_dend=dirs,
                                                data_dir=data_dir_tmp,
                                                dest_dir=dirs,
                                                **cfg
                                            ) 
                        modes[val][name][dirs]=[dmode.mode_id]  
        model_sufix_dic={mm:{} for mm in ['model_sufix_dic','path','path_heads_show','path_heads_dic','model_sufix_dic_inverse','model_sufix_show','path_dirs_show','drop_dic']}
        model_sufix_dic['model_sufix_show']=[data_mode[modes[model_type][dnn_mode][path_dir][0]]['model_sufix'][0] for dnn_mode in  model_sufix_show]
        model_sufix_dic['model_sufix_dic'] ={data_mode[modes[model_type][dnn_mode][path_dir][0]]['model_sufix'][0] :dnn_mode for dnn_mode in  model_sufix_show}
        model_sufix_dic['model_sufix_dic']['save']='save'
        model_sufix_dic['model_sufix_dic_inverse'] = {v: k for k, v in model_sufix_dic['model_sufix_dic'].items()} 
        # print('[[[[[[[[[[lllllloooooolllll]]]]]]]]]]',model_sufix_show,model_sufix_dic['model_sufix_dic'],model_sufix_dic['model_sufix_show'])
 
        configs=get_configs()
        fcgs={}
        for ids, name in enumerate(model_sufix_show):
            # print(f"Finished {name}, dnn_modes={model_sufix_show}")
            cfg=configs[name]
            # print('[[[]]]',name,cfg)
            cfg["data_sufix"],cfg["dest_sufix"]=model_sufix_dic['model_sufix_dic_inverse'][cfg["data_sufix"]],model_sufix_dic['model_sufix_dic_inverse'][cfg["dest_sufix"]]
            cfg["data_dir"]=path_dir #if path_shaft_dir is None else path_shaft_dir.get('data_dir',path_dir)
            cfg["dest_dir"]=path_dir #if path_shaft_dir is None else path_shaft_dir.get('data_dir',path_dir)
            fcgs[name]=cfg
 

        dmode = get_data_mode(pre_portion=pre_portion,path_list=path_list) 
        for val in path_heads_show:
            # for valdata in path_heads_show_data: 
                ids,name,dirs=0,'mode0',path_dirs_show[0]
                data_dir_tmp=data_dir if data_dir is not None else dirs
                data_mode=dmode.test_pre(pre_portion=pre_portion,
                                            data_head=model_type_data,
                                            dest_head=val,
                                            seg_dend=dirs, 
                                            data_dir=data_dir_tmp,)
                modes[val][name][dirs]=[dmode.mode_id] 
                # print(f"Finished {name}, mode_id={dmode.mode_id}") 

                for dirs in path_dirs_show: 
                    for ids, name in enumerate(model_sufix_show):
                        fcgs[name]['data_dir']=data_dir if data_dir is not None else dirs 
                        # cfg=configs[name]
                        data_mode = dmode.test_opt(
                                                data_mode=data_mode,
                                                pre_portion=pre_portion,
                                                train_test='test',
                                                data_head=model_type_data,
                                                dest_head=val, 
                                                seg_dend=dirs, 
                                                **fcgs[name]
                                            )
                        modes[val][name][dirs]=[dmode.mode_id] 
                        # modes[val][dmode.mode_id]=name
                        # print(f"Finished {name}, mode_id={dmode.mode_id}")
        self.modes,self.data_mode,self.dmode=modes,data_mode,dmode 
        path_heads=self.path_heads 
        param=self.param 
        # mode_ids=modes[model_type][dnn_mode][path_dir] 
        # true_keys=self.true_keys 
        # print(f"heeeds===== {model_type}, mo---de_id={dnn_mode}")
        # dend_data= gdas.part(nam)
        path_display = path_display if path_display is not None else param['dendrite_pred']['param']['path_display']
 

        # model_sufix_dic={data_mode[modes[model_type][dnn_mode]]['model_sufix'][0]:dnn_mode for dnn_mode in  model_sufix_show}
        model_sufix_dic['path_dirs_dic']={ke:va for ke,va in zip(path_dirs_show,path_dirs_show)}  
        model_sufix_dic['path_dir']=path_dir
        result =[part if len(part.split('_'))<=1 else '_'.join(part.split('_')[1:]) for part in path_heads]
        path_headss=path_heads+['save']
        result=result+['default'] 
        model_sufix_dic['path_heads_show']=path_heads_show  
        model_sufix_dic['path_dirs_show']=path_dirs_show

        model_sufix_dic['path_heads_dic']={ke:va for ke,va in zip(path_headss,result)}  
        model_sufix_dic['head_navbar']={
                                        "DSA": "dsa",
                                        "DNN": "dnn",
                                        # "DNN v.0": "pinn",
                                        "3D CNN": "cnn",
                                        "GCN": 'gcn',
                                        "class ML":'cml',
                                    }
        self.model_sufix_dic=model_sufix_dic
        self.obj_org_path_dict = dend_data['obj_org_path_dict']



 

 





    def test( self,   
            dend_data=None,  
            true_name='true_0', 
            dnn_mode='mode0', 
            model_type='pinn' , 
            model_type_data=None,
            path_dir='save',
            path_display=None, 
            size_threshold=None,
            path_display_dic=None,
            path_shaft_dir=None, 
            path_heads_show=None, 
            model_sufix_show=None, 
            path_dirs_show=None,
            train_spines=False,
            data_dir=None,
            drop_dic=None,
            n_step = None,
            weight=None,
            dnn_modes=None,
            path_dirs=None, 
            param_dic=None,
            kmean_n_run=None,
            path_list=None,
            path_heads=None,
            nam=None,
            path_dirs_weig=None,
            ): 
        param=self.param
        path_heads_show=path_heads_show if path_heads_show is not None else self.path_heads
        model_sufix_show=model_sufix_show if model_sufix_show is not None else self.dnn_modes 
        path_dirs_show=path_dirs_show if path_dirs_show is not None else self.path_dirs
        model_type_data=model_type_data if model_type_data is not None else model_type
        self.data(   
                exit_name=None,
                entry_name=None, 
                train_test='test',
                dend_data=dend_data,  
                true_name=true_name, 
                dnn_mode=dnn_mode, 
                model_type=model_type, 
                model_type_data=model_type_data, 
                path_dir=path_dir,
                path_display=path_display, 
                size_threshold=size_threshold,
                path_display_dic=path_display_dic, 
                path_shaft_dir=path_shaft_dir, 
                path_heads_show=path_heads_show,
                model_sufix_show=model_sufix_show, 
                path_dirs_show=path_dirs_show, 
                data_dir=data_dir,
                drop_dic=drop_dic, 
                ) 
        dend_data_tmp=self.dend_data_tmp
        pre_portion=self.pre_portion
        path_list=self.path_list
        data_mode=self.data_mode
        modes,data_mode,dmode=self.modes,self.data_mode,self.dmode
        self.model_sufix_dic['drop_dic']=drop_dic
        model_sufix_dic=self.model_sufix_dic
        mode_ids=modes[model_type][dnn_mode][path_dir] 
        true_keys=self.true_keys 
        path_heads=self.path_heads  
        obj_org_path_dict =self.obj_org_path_dict  
    
        time_start = time.time()
 

        model_type11=model_type #if path_shaft_dir is None else path_shaft_dir.get('model_type',model_type)
        path_dir11=path_dir  #if path_shaft_dir is None else   path_shaft_dir.get('path_dir',path_dir)
        path_head_clean_=[False for item in range(len(mode_ids))]
        path_head_clean_=[True if item <1 else False for item in range(len(mode_ids))]
        for ixi,mode_id in enumerate(mode_ids):#data_mode.keys(): #
            nhh=data_mode[mode_id]['model_sufix'] 
            model_sufix=data_mode[mode_id]['model_sufix'][0] 


            mode_dnn1 =data_mode[mode_id]['model_init']
            mode_dnn1=mode_dnn1 # if mode_dnn1 is not None else (mode_dnn1 if path_shaft_dir is None else path_shaft_dir.get('model_sufix',mode_dnn1))
            mode_dnn11=model_sufix_dic['model_sufix_dic_inverse'].get(mode_dnn1,None)
            path_shaft_dir=None if mode_dnn11 is None else f'{model_type11}_{mode_dnn11}_{path_dir11}'


            dend_cla=dendrite_pred(   
                                dend_data=dend_data_tmp, 
                                model_sufix=model_sufix,
                                data_mode=data_mode[mode_id],   
                                pinn_dir_data_all=dmode.pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                                path_heads= path_heads,
                                model_type=model_type, 
                                obj_org_path_dict=obj_org_path_dict ,
                                true_keys=true_keys,
                                model_sufix_dic=model_sufix_dic,
                                path_display_dic=path_display_dic,
                                # path_display=path_display,
                                **param['dendrite_pred']['param']
                                ) 
 
            dend_cla.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)  
            
 
            if param['clean_path_dir']['tf']:
                path_head_clean=param['clean_path_dir']['param']['path_clean'] 
                print('[[[[[[path_head_clean]]]]]]',path_head_clean) 
    
            size_threshold=param['dendrite_pred']['param']['size_threshold'] 

            if param['rhs']['tf']: 
                dend_cla.get_rhs(
                                        param_dend_name=param['dendrite_pred']['param']['param_dic']['data']['get_dend_name'],
                                        **param['Skeleton']['param'])
         

            if param['Spine-Shaft Segm']['tf']:   
                if (dnn_mode== 'DNN-2'):
                        dend_cla.get_skl_shaft_pred(
                                                # weights=weights,
                                                path_shaft_dir=path_shaft_dir,
                                                train_spines=train_spines,
                                                model_type=model_type,
                                            **param['Spine-Shaft Segm']['param']
                                            ) 
                if model_type.startswith(('vol','cnn')):
                    dend_cla.get_shaft_pred_cnn(
                                            # weights=weights,
                                            train_spines=train_spines,
                                            model_type=model_type,
                                        **param['Spine-Shaft Segm']['param']
                                        )   
                elif model_type.startswith(('gcn',)): 
                    dend_cla.get_shaft_pred_gcn(
                                            # weights=weights, 
                                            train_spines=train_spines,
                                            model_type=model_type,
                                        **param['Spine-Shaft Segm']['param']
                                        )   
                elif model_type.startswith(('dnn',)): 
                    dend_cla.get_shaft_pred_dnn(
                                            # weights=weights,
                                            train_spines=train_spines,
                                            model_type=model_type,
                                        **param['Spine-Shaft Segm']['param']
                                        )   
                elif model_type.startswith(('pnet',)): 
                    dend_cla.get_shaft_pred_pnet(
                                            # weights=weights,
                                            train_spines=train_spines,
                                            model_type=model_type,
                                        **param['Spine-Shaft Segm']['param']
                                        )   
                elif model_type.startswith(('cml','cML', 'ML')): 
                    dend_cla.get_shaft_pred_cml(
                                            # weights=weights,
                                            train_spines=train_spines,
                                            model_type=model_type,
                                        **param['Spine-Shaft Segm']['param']
                                        )   
                elif model_type.startswith(('pinn',)): 
                    dend_cla.get_shaft_pred_PINN(
                                            # weights=weights,
                                            train_spines=train_spines,
                                            model_type=model_type,
                                        **param['Spine-Shaft Segm']['param']
                                        )   
                else:
                    dend_cla.get_shaft_pred(
                                            # weights=weights,
                                            train_spines=train_spines,
                                            model_type=model_type,
                                            **param['Spine-Shaft Segm']['param']
                                        )    
                dend_cla.get_shaft_process(
                                        # weights=weights,
                                        train_spines=train_spines,
                                        model_type=model_type,
                                        **param['Spine-Shaft Segm']['param']
                                    ) 

            if param['skl_shaft_pred']['tf']: 
                dend_cla.get_skl_shaft_pred(
                                        # weights=weights,
                                        path_shaft_dir=path_shaft_dir,
                                        train_spines=train_spines,
                                        model_type=model_type,
                                        **param['Spine-Shaft Segm']['param']
                                    ) 
            
            if param['model_shap']['tf']:
                dend_cla.model_shap( 
                                        train_spines=train_spines,
                                        model_type=model_type,
                                    **param['Spine-Shaft Segm']['param'],
                                    # **param['model_shap']['param']
                                    ) 
            dom='sp_nfull' 
            if param['spines_segss_old']['tf']:
                dend_cla.get_spines_segss_old(  
                                            dest_path=data_mode[mode_id][dom]['path'],
                                            **param['spines_segss_old']['param']
                                            )
     
         
            if param['Morphologic Param']['tf']: 
                for pdisplay in path_display:
                    dend_cla.get_head_neck_segss(  
                                        head_neck_path=pdisplay,
                                        **param['Morphologic Param']['param']
                                )  

            if param['iou']['tf']:
                dend_cla.get_iou()  

            if param['roc']['tf']:
                dend_cla.get_roc()  




            if param['graph_center']['tf']:
                dend_cla.get_graph_center() 

            if param['cylinder_heatmap']['tf']:
                # True
                dend_cla.get_cylinder_heatmap()

            if param['dash_pages']['tf']:
                dend_cla.get_dash_pages()





            mytime0 = time.time() - time_start 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60) 
            print(f'Prediction completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s')  

        return dend_cla
                
 




    def train( self,  
            dend_data=None,  
            true_name='true_0', 
            path_dir='save',
            dnn_mode='mode0',
            model_type='pinn' , 
            entry_names=[],
            path_display=None,  
            path_list=None,
            path_display_dic=None,
            path_shaft_dir=None, 
            path_heads_show=None,
            model_sufix_show=None, 
            path_dirs_show=None,
            size_threshold=None,
            model_type_data=None, 
            data_dir=None,
            ): 
        param=self.param
        path_heads_show=path_heads_show if path_heads_show is not None else self.path_heads_true
        model_sufix_show=model_sufix_show if model_sufix_show is not None else self.dnn_modes 
        path_dirs_show=path_dirs_show if path_dirs_show is not None else self.path_dirs
        model_type_data=model_type_data if model_type_data is not None else model_type
        # model_type='true'
        self.pre_test_train(   
            train_test='train',
            dend_data=dend_data,  
            true_name=true_name, 
            path_heads_show=path_heads_show,
            model_sufix_show=model_sufix_show, 
            path_dirs_show=path_dirs_show,
            dnn_mode=dnn_mode, 
            model_type=model_type, 
            model_type_data=model_type_data, 
            path_dir=path_dir,
            path_display=path_display, 
            size_threshold=size_threshold,
            path_display_dic=path_display_dic, 
            path_shaft_dir=path_shaft_dir, 
            data_dir=data_dir,
            )
        pre_portion=self.pre_portion
        path_list=self.path_list
        param=self.param
        data_mode=self.data_mode
        modes,data_mode,dmode=self.modes,self.data_mode,self.dmode
        model_sufix_dic=self.model_sufix_dic
        mode_ids=modes[model_type][dnn_mode][path_dir]
        true_keys=self.true_keys 
        path_heads=self.path_heads  
        obj_org_path_dict =self.obj_org_path_dict

 
        time_start = time.time()
  
        for ixi,mode_id in enumerate(mode_ids):#data_mode.keys(): #
            nhh=data_mode[mode_id]['model_sufix'] 
            model_sufix=data_mode[mode_id]['model_sufix'][0] 
            dend_cla=dendrite_pred(   
                                dend_data=dend_data, 
                                model_sufix=model_sufix,
                                data_mode=data_mode[mode_id],   
                                pinn_dir_data_all=dmode.pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                                path_heads= path_heads,
                                model_type=model_type,
                                obj_org_path_dict=obj_org_path_dict ,
                                true_keys=true_keys,
                                model_sufix_dic=model_sufix_dic,
                                **param['dendrite_pred']['param']
                                ) 
 
            dend_cla.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)   

   
            size_threshold=param['dendrite_pred']['param']['size_threshold'] 
            train_spines=data_mode[mode_id]['train_spines']
            if param['get_training']['tf']:
                if model_type.startswith(('cml','ML','cML')): 
                    dend_cla.train_model_spine_cml(     
                                            model_sufix=model_sufix, 
                                            train_spines=train_spines, 
                                            model_type=model_type,  
                                            num_sub_nodes=20000,
                                            entry_names=entry_names,
                                            **param['get_training']['param']
                                            )
                elif model_type.startswith(('gcn',)):
                    dend_cla.train_model_spine_gcn(
                                            train_spines=train_spines, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            )  
                elif model_type.startswith(('dnn',)):
                    dend_cla.train_model_spine_dnn(
                                            train_spines=train_spines, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            ) 
                elif model_type.startswith(('pinn',)):
                    dend_cla.train_model_spine_PINN(
                                            train_spines=train_spines, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            )  
                elif model_type.startswith(('pnet',)): 
                    dend_cla.train_model_spine_pnet(
                                            train_spines=train_spines, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            )  
                elif model_type.startswith(('vol',"fastfcn3d","unet3d",'cnn')):
                    dend_cla.train_model_spine_cnn(
                                            train_spines=train_spines, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            )  
                else:
                    dend_cla.train_model_spine(
                                            train_spines=train_spines, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            ) 

            mytime0 = time.time() - time_start 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60) 
            print(f'Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s')  

        return [f'j ']
                





    def data( self,  
            dend_data=None,  
            true_name='true_0', 
            path_dir='save',
            dnn_mode='mode0',
            model_type='pinn' , 
            entry_names=[],
            path_display=None,  
            path_list=None,
            path_display_dic=None,
            path_shaft_dir=None, 
            path_heads_show=None,
            model_sufix_show=None, 
            path_dirs_show=None,
            size_threshold=None,
            exit_name=None,
            entry_name=None,
            train_test='test',
            model_type_data=None,
            data_dir=None,
            drop_dic=None,
            ): 
        param=self.param
        path_heads_show=path_heads_show if path_heads_show is not None else self.path_heads_true
        model_sufix_show=model_sufix_show if model_sufix_show is not None else self.dnn_modes 
        path_dirs_show=path_dirs_show if path_dirs_show is not None else self.path_dirs
        # model_type='true'
        self.pre_test_train(   
            train_test=train_test,
            dend_data=dend_data,  
            true_name=true_name, 
            path_heads_show=path_heads_show,
            model_sufix_show=model_sufix_show, 
            path_dirs_show=path_dirs_show,
            dnn_mode=dnn_mode, 
            model_type=model_type, 
            path_dir=path_dir,
            path_display=path_display, 
            size_threshold=size_threshold,
            path_display_dic=path_display_dic, 
            path_shaft_dir=path_shaft_dir,  
            data_dir=data_dir,
            )
        pre_portion=self.pre_portion
        path_list=self.path_list
        data_mode=self.data_mode
        modes,data_mode,dmode=self.modes,self.data_mode,self.dmode
        self.model_sufix_dic['drop_dic']=drop_dic
        model_sufix_dic=self.model_sufix_dic
        # print('[[[[[[[[[[[[[[-----]]]]]]]]]]]]]]',dnn_mode,modes[model_type].keys())
        mode_ids=modes[model_type][dnn_mode][path_dir] 
        true_keys=self.true_keys 
        path_heads=self.path_heads  
        obj_org_path_dict =self.obj_org_path_dict
        from copy import deepcopy


        path_head_clean_=[False for item in range(len(mode_ids))]
        path_head_clean_=[True if item <1 else False for item in range(len(mode_ids))]
        for ixi,mode_id in enumerate(mode_ids):#data_mode.keys(): # 
            model_sufix=data_mode[mode_id]['model_sufix'][0] 
            dend_cla=dendrite_pred(   
                                dend_data=dend_data, 
                                model_sufix=model_sufix,
                                data_mode=data_mode[mode_id],   
                                pinn_dir_data_all=dmode.pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                                path_heads= path_heads,
                                model_type=model_type,
                                obj_org_path_dict=obj_org_path_dict ,
                                true_keys=true_keys,
                                model_sufix_dic=model_sufix_dic,
                                **param['dendrite_pred']['param']
                                ) 
 
            dend_cla.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)   


            time_start = time.time()
            # if param['rhs']['tf']: 
            #     dend_cla.get_rhs( 
            #                     entry_name=exit_name,
            #                     exit_name =exit_name, 
            #                     **param['Skeleton']['param'],
            #         )

            if param['Resizing']['tf']: 
                parr=dict(
                        get_data=True,
                        n_error = 1,
                        n_step = 0,
                        dt=1e-6,
                        disp_time=500, )
            else:
                parr=param['Smooth']['param']
            save_entry_exit={nn:[] for nn in ['entry','exit','old_path']}
            save_entry_exit['entry'].append(entry_name)
            save_entry_exit['exit'].append(exit_name)
            save_entry_exit['old_path'].append(entry_name)
            if param['Resizing']['tf']: 
                sze=param['Resizing']['param']['target_number_of_triangles_faction']
                exit_name=f'resize_{sze}' if entry_name is None else f'{entry_name}_resize_{sze}'
                dend_cla.get_resize(entry_names=[entry_name],
                                    exit_names=[exit_name],
                                    target_number_of_triangles_faction=sze,  
                                    )
                entry_name=exit_name
                save_entry_exit['entry'].append(entry_name)
                save_entry_exit['exit'].append(exit_name)
                save_entry_exit['old_path'].append(entry_name)

            if param['Smooth']['tf']:
                exit_name='smooth'  if entry_name is None else f'{entry_name}_smooth'
                dend_cla.get_smooth( 
                                entry_name=entry_name,
                                exit_name=exit_name,
                                # dt=1e-6, 
                                # n_step = 100,
                                **param['Smooth']['param'] 
                                )
                entry_name=exit_name
                save_entry_exit['entry'].append(entry_name)
                save_entry_exit['exit'].append(exit_name)
                save_entry_exit['old_path'].append(entry_name,)
            for nn in ['entry','exit']:
                if len(save_entry_exit[nn])==0:
                    save_entry_exit[nn].append(None) 
                    param['dendrite_pred']['param']['param_dic']['data']['get_dend_name']['old_path']='current'
                save_entry_exit['old_path'].append('entry')



            if param['Resizing']['tf']: 
                # param['dendrite_pred']['param']['param_dic']['data']['get_dend_name']['dict_dend_path']=dict(dict_dend_path='old',
                #                                 drop_dic_name=save_entry_exit['exit'][-1])
                param['dendrite_pred']['param']['param_dic']['data']['get_dend_name']['dict_dend_path']= 'old'
                param['dendrite_pred']['param']['param_dic']['data']['get_dend_name']['drop_dic_name']= save_entry_exit['exit'][-1]
                param['dendrite_pred']['param']['param_dic']['data']['get_dend_name']['old_path']='current'
            mmnn=param['Skeleton']['param']['path']
            pathhh=f'{mmnn}'
            for entry_name,exit_name,old_path in zip(save_entry_exit['entry'],save_entry_exit['exit'],save_entry_exit['old_path']):
                for paath in ['entry','old']:
                    param['Skeleton']['param']['path']=paath
                    # if entry_name == 'smooth':'old',
                    if param['Skeleton']['tf']: 
                        # print('[[[[[[[--------------skl------------]]]]]]]',paath,entry_name,exit_name )
                        # print('[[[[[[[--------------skl------------]]]]]]]',param['dendrite_pred']['param']['param_dic']['data']['get_dend_name'] )
                        param_dend_name=deepcopy(param['dendrite_pred']['param']['param_dic']['data']['get_dend_name'])
                        param_dend_name['drop_dic_name']=None
                        param_dend_name['nam_gen']=None
                        # print('[[[[[[[--------------skl------------]]]]]]]',param_dend_name )

                        dict_wrap=param['Skeleton']['param']['dict_wrap']
                        dend_cla.get_wrap(wrap_part='shaft', 
                                        alpha_fraction=1,
                                        offset_fraction=.7,
                                        entry_name=entry_name,
                                        exit_name=exit_name,
                                        dict_wrap=dict_wrap,
                                        old_path=old_path,
                                        param_dend_name=param_dend_name,
                                        ) 
                        param['Skeleton']['param']['wrap_part']=None
                        param['Skeleton']['param']['tf_restart']=True
                        dend_cla.get_skeleton(
                                        entry_name=entry_name,
                                        exit_name =exit_name,
                                        old_path=old_path,
                                        # wrap_part='shaft_wrap' , 
                                        **param['Skeleton']['param'], 
                        )
                        param['Skeleton']['param']['wrap_part']='shaft_wrap'
                        param['Skeleton']['param']['tf_restart']=True
                        dend_cla.get_skeleton(
                                        entry_name=entry_name,
                                        exit_name =exit_name,
                                        old_path=old_path,
                                        # wrap_part='shaft_wrap' , 
                                        **param['Skeleton']['param'], 
                        )
                    if param['rhs']['tf']: 
                        dend_cla.get_rhs( 
                                        entry_name=exit_name,
                                        exit_name =exit_name,
                                        old_path=old_path,
                                        param_dend_name=param_dend_name,
                                        **param['Skeleton']['param'],
                            )
            param['Skeleton']['param']['path']=pathhh

 
                
            dend_data_tmp=deepcopy(dend_data)     
            mmm=dend_data['obj_org_path']
            dend_data_tmp['obj_org_path']=dend_data['obj_org_path'] if exit_name is None else f'{mmm}_{exit_name}'  
            dend_data_tmp['dend_path_inits']=dend_data['dend_path_inits'] if exit_name is None else [f'{nam}_{exit_name}'  for nam in dend_data['dend_path_inits']] 


            mytime0 = time.time() - time_start 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60) 
            print(f'Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s')  
        self.dend_data_tmp=dend_data_tmp
    
 

class get_data:
    def __init__(self, file_path_data): 
        dend_data={}   
        nam='p21-dendrites'

        dend_names = [
            'BTLNG_d004_RECON',
            'BTLNG_d005_RECON',
            'BTLNG_d010_RECON',
            'CNJVY_d001_RECON',
            'CNJVY_d003_RECON',
            'CNJVY_d005_RECON',
        ]   
        dend_last = [
            'BTLNG', 
            'BTLNG', 
            'BTLNG', 
            'CNJVY', 
            'CNJVY', 
            'CNJVY', 
        ] 
        dend_first = [
            'd004',
            'd005',
            'd010',
            'd001',
            'd003',
            'd005',
        ] 
        dend_namess = [[f'{dn}_',df,f'{df}sp'] for dn,df in zip(dend_names,dend_first)]
        for nam in [ 'p21-dendrites', ]:
            dend_data[nam]=dict( 
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

        
        nam='neuropil_recon_0'  
        obj_list=[ls for ls in range(150) if ls not in [129]]  
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

        obj_org_path= os.path.join(file_path_data,'neuropil_0', nam )

        dend_data[nam]= dict(  
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
        
        
        nam='neuropil_recon_1_resized' 
        obj_list=[ls for ls in range(150) if ls not in [129]] 
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

        obj_org_path= os.path.join(file_path_data,'neuropil_0', nam )

        dend_data[nam]= dict(  
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


        self.dend_data=dend_data 
 
    
    def part(self,nam, istart=0, ifin=None): 
        dd = self.dend_data[nam]
        return {
            'dend_namess': dd['dend_namess'][istart:ifin],
            'dend_names': dd['dend_names'][istart:ifin],
            'dend_last': dd['dend_last'][istart:ifin],
            'dend_first': dd['dend_first'][istart:ifin],
            'dend_path_inits': dd['dend_path_inits'][istart:ifin],
            'name_spine_id':dd['name_spine_id'],
            'name_head_id':dd['name_head_id'],
            'name_neck_id':dd['name_neck_id'],
            'name_shaft_id':dd['name_shaft_id'],
            'objs_path_org':dd['objs_path_org'],
            'weights':dd['weights'],
        } if ifin is not None else  {
            'dend_namess': dd['dend_namess'][istart:],
            'dend_names': dd['dend_names'][istart:],
            'dend_last': dd['dend_last'][istart:],
            'dend_first': dd['dend_first'][istart:],
            'dend_path_inits': dd['dend_path_inits'][istart:],
            'name_spine_id':dd['name_spine_id'],
            'name_head_id':dd['name_head_id'],
            'name_neck_id':dd['name_neck_id'],
            'name_shaft_id':dd['name_shaft_id'],
            'objs_path_org':dd['objs_path_org'],
            'weights':dd['weights'],
        } 
    




class get_data_all:
    def __init__(self, dend_data=None,names_dic={},obj_org_path=None,file_path_data=None,true_keys=[f'true_0',] ):  
        self.dend_data=dend_data   
            
        if self.dend_data is None:
            self.dend_data={}
        else: 
            for nam,names in dend_data.items() :   
                dend_names = names.get('dend_names', [])
                dend_first = names.get('dend_first', ['d000'] * len(dend_names))
                dend_path_inits = names.get('dend_path_inits',[nam  for _ in range(len(dend_names))])
                dend_namess = names.get('dend_namess', [[f'{de}_', de, f'{de}'] for de in dend_first])
                dend_last = names.get('dend_last', [f'{de}_' for de in dend_first])
                name_spine_id=names.get('name_spine_id','p_sps')
                name_head_id=names.get('name_head_id','p_sphs')
                name_neck_id=names.get('name_neck_id','p_spns')
                name_shaft_id=names.get('name_shaft_id','p_spsh')
                objs_path_org_tmp=  os.path.join(file_path_data,nam )  
                obj_org_path=names.get('obj_org_path',objs_path_org_tmp) 
                weights=names.get('weights',[[6.,0.81]])
                obj_org_path_dict=names.get('obj_org_path_dict',{val: obj_org_path for val in true_keys}) 
                for na in [nam,f'{nam}_resized']:
                    self.dend_data[nam]=dict( 
                                            dend_namess=dend_namess ,
                                            dend_names=dend_names ,
                                            dend_last=dend_last ,
                                            dend_first=dend_first ,
                                            dend_path_inits=dend_path_inits,
                                            name_spine_id=name_spine_id,
                                            name_head_id=name_head_id,
                                            name_neck_id=name_neck_id,
                                            name_shaft_id=name_shaft_id,
                                            obj_org_path=obj_org_path,
                                            obj_org_path_dict=obj_org_path_dict,
                                            weights=weights,
                                            ) 

        for nam,names in names_dic.items() :   
            dend_names = names.get('dend_names', [])
            dend_first = names.get('dend_first', ['d000'] * len(dend_names))
            dend_path_inits = names.get('dend_path_inits',[nam  for _ in range(len(dend_names))])
            dend_namess = names.get('dend_namess', [[f'{de}_', de, f'{de}'] for de in dend_first])
            dend_last = names.get('dend_last', [f'{de}_' for de in dend_first])
            name_spine_id=names.get('name_spine_id','p_sps')
            name_head_id=names.get('name_head_id','p_sphs')
            name_neck_id=names.get('name_neck_id','p_spns')
            name_shaft_id=names.get('name_shaft_id','p_spsh')
            objs_path_org_tmp=  os.path.join(file_path_data,nam )  
            obj_org_path=names.get('obj_org_path',objs_path_org_tmp) 
            obj_org_path_dict=names.get('obj_org_path_dict',{f'true_0': obj_org_path for val in dend_names})  
            weights=names.get('weights',[[6.,0.81]]) 
            for na in [nam,f'{nam}_resized']:
                self.dend_data[na]=dict( 
                                        dend_namess=dend_namess ,
                                        dend_names=dend_names ,
                                        dend_last=dend_last ,
                                        dend_first=dend_first ,
                                        dend_path_inits=dend_path_inits,
                                        name_spine_id=name_spine_id,
                                        name_head_id=name_head_id,
                                        name_neck_id=name_neck_id,
                                        name_shaft_id=name_shaft_id,
                                        obj_org_path=obj_org_path,
                                        obj_org_path_dict=obj_org_path_dict,
                                            weights=weights,
                                        )

    def part(self,nam, istart=0, ifin=None): 
        dd = self.dend_data[nam]
        return {
            'dend_namess': dd['dend_namess'][istart:ifin],
            'dend_names': dd['dend_names'][istart:ifin],
            'dend_last': dd['dend_last'][istart:ifin],
            'dend_first': dd['dend_first'][istart:ifin],
            'dend_path_inits': dd['dend_path_inits'][istart:ifin],
            'name_spine_id':dd['name_spine_id'],
            'name_head_id':dd['name_head_id'],
            'name_neck_id':dd['name_neck_id'],
            'name_shaft_id':dd['name_shaft_id'],
            'obj_org_path':dd['obj_org_path'],
            'obj_org_path_dict':dd['obj_org_path_dict'],
            'weights':dd['weights'],
        } if ifin is not None else  {
            'dend_namess': dd['dend_namess'][istart:],
            'dend_names': dd['dend_names'][istart:],
            'dend_last': dd['dend_last'][istart:],
            'dend_first': dd['dend_first'][istart:],
            'dend_path_inits': dd['dend_path_inits'][istart:],
            'name_spine_id':dd['name_spine_id'],
            'name_head_id':dd['name_head_id'],
            'name_neck_id':dd['name_neck_id'],
            'name_shaft_id':dd['name_shaft_id'],
            'obj_org_path':dd['obj_org_path'],
            'obj_org_path_dict':dd['obj_org_path_dict'],
            'weights':dd['weights'],
        } 
    



