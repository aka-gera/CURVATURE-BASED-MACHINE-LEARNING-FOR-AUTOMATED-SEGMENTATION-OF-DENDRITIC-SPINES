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
file_path_parent=os.path.join(file_path_parent,'meshes')

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


def algorithm_param(nam='meshes',n_step = 200,weight=3,size_threshold=100, ):
 
    smooth_tf=True
    smooth_tf=False
    data_studied='train'
    data_studied='test' 
    
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

    spline_smooth_shaft=.03
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
 
    path_heads=['pinn','rpinn','ML','pinn_old']
    path_heads=['pinn', ]
    dnn_modes=['DNN-1','DNN-2' , 'DNN-3']




    path_display=['dest_spine_path_pre','dest_spine_path','dest_shaft_path']
    path_display=['dest_shaft_path','dest_spine_path',]
    dict_mesh_to_skeleton_finder_mesh=dict( 
                interval_voxel_resolution=[100 ], 
                interval_target_number_of_triangles=[1], 
                tf_largest=False,
                disp_infos=True, 
                min_voxel_resolution=200,
                min_target_number_of_triangles=1000,
                tf_division=True,
                    )

    param_get_segments=dict(
                            thre_distance_min=1.75 ,
                            thre_distance_max=1.5,
                            thre_explained_variance_ratio=0.97,
                            size_threshold=100, 
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
    
 
    param_clean=[ 'all', 'result']+path_heads

    param_dropdown=['path_heads', 'dnn_modes']
    param_dropdowni=['path_head', 'dnn_mode']
    param_input=['Smooth','Resizing','Spine–Shaft Segm', 'Morphologic Param','model_shap','clean_path_dir',  ]
    param_inputbv=[ 'intensity_all','iou','graph_center','dash_pages', 'cylinder_heatmap' ,'annotations',  'dendrite_pred',  ]
    param_input_save=[ 'spines_segss_old','spines_segss_old_2', 'clean_path_dir','get_training', ]
    param_list=['param_dropdown','param_input','param_all', 'param_list','param_dropdowni','param_inputbv' ]
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
    param['path_heads']['param']=path_heads
    param['param_dropdown']['param']=param_dropdown
    param['param_dropdowni']['param']=param_dropdowni
    param['param_input']['param']=param_input
    param['param_inputbv']['param']=param_inputbv
    param['param_all']['param']=param_all 
    param['dnn_modes']['param']=dnn_modes
    param['path_head']['param']=path_heads[0]
    param['dnn_mode']['param']=dnn_modes[0] 


    
    param['Smooth']['param']=dict(
                                get_data=True,
                                n_error = 1,
                                n_step = n_step,
                                dt=1e-6,
                                disp_time=500, )
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
    param['Spine–Shaft Segm']['param']=dict( weight=weight,
                                      size_threshold=size_threshold,
                                    )   
    param['Spine–Shaft Segm']['param_fix']=dict(
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
                                        min_target_number_of_triangles_faction=600000,
                                        target_number_of_triangles_faction=10000,
                                                )
    param['Resizing']['param_fix']=dict( 
                                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                                        voxel_resolution=voxel_resolution,
                                        annotation_resized_train_tf=False, 
                                                )


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
                            )
    param['dendrite_pred']['tf']=True 
    param['model_shap']['tf']=False

    param['Spine–Shaft Segm']['tf']=True 
    param['spines_segss_old']['tf']=False
    param['spines_segss_old_2']['tf']= False
    param['Morphologic Param']['tf']= True
    param['Resizing']['tf']=False
    param['iou']['tf']=True
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
        self.path_heads=param['path_heads']['param']
        self.dnn_modes=param['dnn_modes']['param']
        self.Input_id.append('upload-data')

 
        for gval,gvali in zip(param['param_dropdown']['param'],param['param_dropdowni']['param']):
            print(gval,param[gval]['param'] )
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
                                    dbc.Checkbox(
                                        id=f"param_{gval}_{k}",
                                        value=v,
                                        label="Enabled"
                                    ) if isinstance(v, bool) else dcc.Input(
                                        id=f"param_{gval}_{k}",
                                        type="text",
                                        value=v,
                                        style={'width': '75px', 'marginLeft': 'auto'}  # smaller + pushed right
                                    ),
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
                                        dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Run", 
                                                id="run-button", 
                                                n_clicks=0, 
                                                color="primary",
                                               #  color="success",
                                                className="w-1",)
                                            ]),
                                            style={"margin": "0px", "padding": "0px"}
                                        ),
                                        width="4", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Restart", 
                                                id="restart-button", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                            style={"margin": "0px", "padding": "0px"}
                                        ),
                                        width="4", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Off", 
                                                id="shutdown-button", 
                                                n_clicks=0,
                                                color="primary",
                                               #   color="danger",
                                                className="w-1",)
                                            ]),
                                            style={"margin": "0px", "padding": "0px" }
                                        ),
                                        width="4", 
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
                                                dcc.Markdown(dnn_page()['results']),
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
                        new_param[gval]['param'][k] = int(val)
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
                        new_param[gval]['param'][k] = int(val)
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
                        new_param[gval]['param'][k] = int(val)
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
                        new_param[gval]['param'][k] = int(val)
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
                ): 
        self.dnn_modes=param['dnn_modes']['param']
        self.path_heads=param['path_heads']['param'] 
        self.param=param  
        self.true_keys=[true_name]
        self.pre_portion=pre_portion

    def text( self,   
            dend_data=None,  
            true_name='true_0', 
            dnn_mode='mode0',
            model_type='pinn' , 
            path_display=None, 
            ): 
        pre_portion=self.pre_portion
        modes={val:{} for val in self.path_heads}
        configs=get_configs()
        dmode = get_data_mode(pre_portion=pre_portion) 
        for val in self.path_heads:
            ids,name=0,'mode0'  
            data_mode=dmode.test_pre(pre_portion=pre_portion,
                                        data_head=val,
                                        dest_head=val, )
            modes[val][name]=[dmode.mode_id] 
            print(f"Finished {name}, mode_id={dmode.mode_id}") 

            for ids, name in enumerate(self.dnn_modes):
                print(f"Finished {name}, mode_id={self.dnn_modes}")
                cfg=configs[name]
                data_mode = dmode.test_opt(
                                        data_mode=data_mode,
                                        pre_portion=pre_portion,
                                        train_test='test',
                                        data_head=val,
                                        dest_head=val, 
                                        **cfg
                                    )
                modes[val][name]=[dmode.mode_id] 
                print(f"Finished {name}, mode_id={dmode.mode_id}")
        self.modes,self.data_mode,self.dmode=modes,data_mode,dmode

        path_heads=self.path_heads 
        param=self.param 
        print(f"heeeds===== {model_type}, mo---de_id={dnn_mode}")
        mode_ids=modes[model_type][dnn_mode]
        # dend_data= gdas.part(nam)      
        self.obj_org_path_dict =obj_org_path_dict=dend_data['obj_org_path_dict']
        true_keys=self.true_keys 
        path_display = path_display if path_display is not None else param['dendrite_pred']['param']['path_display']

        dend_path_inits=dend_data['dend_path_inits']
        dend_names=dend_data['dend_names'] 
        weights=dend_data['weights']

        # model_sufix_dic={data_mode[modes[model_type][dnn_mode]]['model_sufix'][0]:dnn_mode for dnn_mode in  self.dnn_modes}
 
        model_sufix_dic={data_mode[modes[model_type][dnn_mode][0]]['model_sufix'][0] :dnn_mode for dnn_mode in  self.dnn_modes}
        model_sufix_dic['save']='save'
            # model_sufix_dic={data_mode[modes[model_type][dnn_mode][0]]['model_sufix'][0] :dnn_mode for dnn_mode in  self.dnn_modes}
        print(f"heeeds===== model_sufix_dic, mo---de_id={model_sufix_dic}")
        # weightss=[np.array([nn,.81]) for nn,mm in zip(np.arange(a,b,(b-a)/n),np.arange(a,b,(b-a)/n))] 

        # if model_type == 'pinn_old':
        #     weights=[np.array([nn,.81]) for nn,mm in zip(np.arange(a,b,(b-a)/n),np.arange(a,b,(b-a)/n))]  
        #     path_dend_fun='dend_fun_3' 
        # else:
        #     weights=None
    
        time_start = time.time()

        if param['Resizing']['tf']: 
            parr=dict(
                    get_data=True,
                    n_error = 1,
                    n_step = 0,
                    dt=1e-6,
                    disp_time=500, )
        else:
            parr=param['Smooth']['param']

        get_smh=get_smooth(  
                            dend_data=dend_data, 
                            file_path_org=file_path_org,
                            obj_org_path=obj_org_path_dict[true_name],
                            true_keys=true_keys,
                            obj_org_path_dict=obj_org_path_dict,
                                model_sufix_dic=model_sufix_dic, 
                            **parr )

        if param['Smooth']['tf']:
            get_smh.get_smooth_all() 
        else:
            get_smh.get_load_smooth_all()

         
        path_head_clean_=[False for item in range(len(mode_ids))]
        path_head_clean_=[True if item <1 else False for item in range(len(mode_ids))]
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
            train_spines=False if mode_id in ['pre_non_full','pre_non_full_he'] else True
            get_process_=True if mode_id in ['pre_non_full','pre_non_full_he'] else False 

 
            if param['clean_path_dir']['tf']:
                path_head_clean=param['clean_path_dir']['param']['path_clean'] 
                if path_head_clean_[ixi]:
                    dend_cla.clean_path_dir(path_head_clean=path_head_clean)
                get_smh.get_load_smooth_all()


            if param['Resizing']['tf']: 
                dend_cla.get_annotation_resized(
                                                **param['Resizing']['param']
                                                )  
                dend_data['dend_path_inits']=[f'{nam}_resized'  for nam in dend_data['dend_path_inits']]
                pathd= dend_data['obj_org_path']
                dend_data['obj_org_path']=f'{pathd}_resized' 
                dend_data['obj_org_path_dict']['true_0']= dend_data['obj_org_path']  
                get_smh=get_smooth(  
                                    dend_data=dend_data, 
                                    file_path_org=file_path_org,
                                    obj_org_path=obj_org_path_dict[true_name],
                                    true_keys=true_keys,
                                    obj_org_path_dict=obj_org_path_dict, 
                                    model_sufix_dic=model_sufix_dic,
                                    **parr )

                if param['Smooth']['tf']:
                    get_smh.get_smooth_all() 
                else:
                    get_smh.get_load_smooth_all()


                dend_cla=dendrite_pred(   
                                    dend_data=dend_data, 
                                    model_sufix=model_sufix,
                                    data_mode=data_mode[mode_id],   
                                    pinn_dir_data_all=dmode.pinn_dir_data_all,
                                    model_sufix_all=dmode.model_sufix_all, 
                                    path_heads= path_heads,
                                    model_type=model_type,
                                    obj_org_path_dict=obj_org_path_dict ,
                                    model_sufix_dic=model_sufix_dic,
                                    true_keys=true_keys,
                                    **param['dendrite_pred']['param']
                                    ) 
                dend_cla.get_model_opt_name(model_sufix=model_sufix,
                                            model_type=model_type)  

                if param['clean_path_dir']['tf']:
                    path_head_clean=param['clean_path_dir']['param']['path_clean']
                    print('=======[[[[[[[[========]]]]]]]]',path_head_clean)
                    if path_head_clean_[ixi]:
                        dend_cla.clean_path_dir(path_head_clean=path_head_clean)
                    get_smh.get_load_smooth_all()

            if param['annotations']['tf']:
                dend_cla.get_annotations()
                
            if param['intensity_all']['tf']:
                dend_cla.get_intensity_all(**param['intensity_all']['param'])
    
         
            if param['model_shap']['tf']:
                dend_cla.model_shap( 
                                    train_spines=train_spines,
                                    **param['model_shap']['param']
                                    ) 

            
            if param['Spine–Shaft Segm']['tf']:  
                dend_cla.get_shaft_pred(
                                        # weights=weights,
                                        train_spines=train_spines,
                                        model_type=model_type,
                                    **param['Spine–Shaft Segm']['param']
                                    )   


            dom='sp_nfull' 
            if param['spines_segss_old']['tf']:
                dend_cla.get_spines_segss_old(  
                                            dest_path=data_mode[mode_id][dom]['path'],
                                            **param['spines_segss_old']['param']
                                            )
    
            dom='sp_nfull' 
            if param['spines_segss_old_2']['tf']:
                dend_cla.get_spines_segss_old_2(  
                                            dest_path=data_mode[mode_id][dom]['path'],
                                            **param['spines_segss_old_2']['param']
                                            )
    
        

            dom='hn_nfull'
            if param['Morphologic Param']['tf']: 
                for pdisplay in path_display:
                    dend_cla.get_head_neck_segss(  
                                        head_neck_path=pdisplay,
                                        **param['Morphologic Param']['param']
                                )  

            if param['iou']['tf']:
                dend_cla.get_iou()  

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

        return ''
                
 




    def train( self,  
            dend_data=None,  
            true_name='true_0', 
            dnn_mode='mode0',
            model_type='pinn' , 
            path_display=None, 
            ): 
        pre_portion=self.pre_portion  

        configs=get_configs()
        dmode = get_data_mode(pre_portion=pre_portion)
        data_mode = dmode.train_pre(pre_portion=pre_portion)
        modes={}
        ids,name=0,'mode0'  
        data_mode=dmode.train_pre(pre_portion=pre_portion,  )
        modes[name]=[dmode.mode_id ]
        print(f"Finished {name}, mode_id={dmode.mode_id}") 
        for ids in range(1,len(list(configs.items()))):
            name=f'mode{ids}'
            cfg=configs[name]
            data_mode = dmode.train_opt(
                                    data_mode=data_mode,
                                    pre_portion=pre_portion,
                                    train_test='train',
                                    **cfg
                                )
            modes[name]=[dmode.mode_id]
            print(f"Finished {name}, mode_id={dmode.mode_id}" )

 
        self.modes,self.data_mode,self.dmode=modes,data_mode,dmode

        path_heads=self.path_heads 
        param=self.param  
        mode_ids=modes[dnn_mode]
       
        self.obj_org_path_dict =obj_org_path_dict=dend_data['obj_org_path_dict']
        true_keys=self.true_keys 
        path_display = path_display if path_display is not None else param['dendrite_pred']['param']['path_display']
 
         
  
    
        time_start = time.time()

        get_smh=get_smooth(  
                            dend_data=dend_data, 
                            file_path_org=file_path_org,
                            obj_org_path=obj_org_path_dict[true_name],
                            true_keys=true_keys,
                            obj_org_path_dict=obj_org_path_dict, 
                            **param['Smooth']['param'] )

        if param['Smooth']['tf']:
            get_smh.get_smooth_all() 
        else:
            get_smh.get_load_smooth_all()


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
                                **param['dendrite_pred']['param']
                                ) 
            dend_cla.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)  
            if param['clean_path_dir']['tf']:
                path_head_clean_=[True if item <1 else False for item in range(len(mode_ids))] 
                path_head_clean=param['clean_path_dir']['param']['path_head_clean'] 
                if path_head_clean_[ixi]:
                    dend_cla.clean_path_dir(path_head_clean=path_head_clean)

            if param['annotations']['tf']:
                dend_cla.get_annotations()
                
            if param['intensity_all']['tf']:
                dend_cla.get_intensity_all(**param['intensity_all']['param'])
    

            dend_cla.get_intensity_rhs()
 
 
            train_spines=data_mode[mode_id]['train_spines']
            if param['get_training']['tf']:
                if (model_type in ['cML']) or model_type.startswith('ML'): 
                    dend_cla.train_model_spine_ML(     
                                            model_sufix=model_sufix, 
                                            train_spines=train_spines, 
                                            model_type=model_type,  
                                            num_sub_nodes=20000,
                                            **param['get_training']['param']
                                            )
                elif model_type in ['pinn_adv','rpinn_adv','pinn_resized_adv']:
                    dend_cla.train_model_spine_adv(    
                                            train_spines=train_spines, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            **param['get_training']['param'],
                                            ) 
                else:
                    dend_cla.train_model_spine(
                                            train_spines=train_spines, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            **param['get_training']['param'],
                                            ) 

            mytime0 = time.time() - time_start 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60) 
            print(f'Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s')  

        return [f'j ']
                




 

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
    



