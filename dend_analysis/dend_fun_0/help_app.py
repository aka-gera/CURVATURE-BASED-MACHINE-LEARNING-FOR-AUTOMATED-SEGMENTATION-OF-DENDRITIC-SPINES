import dash
from dash import html, dcc, Input, Output, State, callback , ctx, ALL 
import sys
import os
 
file_path_org=os.getcwd()  
sys.path.append(os.path.abspath(file_path_org))
  
# from app_run import app_run_param,algorithm 
from dend_fun_0.obj_get import parse_obj_upload
from  dend_fun_0.main_0 import app_run_param,algorithm,algorithm_param,get_data, get_data_all ,get_dict_param,get_navs_bar
from dend_fun_0.help_funn import remove_directory
import threading
import subprocess 
import signal 
import platform

 
 

 

def restart_appsx():
    try:
        # Kill existing gunicorn/app processes
        subprocess.run(["pkill", "-f", "python"], check=False)

        # Relaunch the app script directly
        subprocess.Popen([
            "/usr/bin/python3", "app/app.py"
        ])
        return "Restart triggered!"
    except Exception as e:
        return f"Error restarting: {e}"

 


def free_port(port=8050):
    system = platform.system()

    if system in ["Darwin", "Linux"]: 
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        pids = result.stdout.strip().splitlines()
        print("PIDs using port:", pids)

        if not pids:
            print(f"Port {port} is free.")
            return
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Killed process {pid} using port {port}")
            except Exception as e:
                print(f"Error killing {pid}: {e}")

    elif system == "Windows": 
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True
        )
        lines = result.stdout.splitlines()
        pids = []
        for line in lines:
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                pids.append(pid)

        if not pids:
            print(f"Port {port} is free.")
            return

        for pid in pids:
            try:
                subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
                print(f"Killed process {pid} using port {port}")
            except Exception as e:
                print(f"Error killing {pid}: {e}")




def start_server(): 
    system = platform.system()
    print('[[[[[[[[[[]]]]]]]]]]', system)

    if system == "Windows": 
        venv_python = os.path.abspath(os.path.join("..", "dsa_venv", "Scripts", "python.exe")) 
        proc = subprocess.Popen([
            venv_python,
            "run_app_win.py"
        ])
        msg = f"Waitress started (Windows) with PID {proc.pid}"

    else:
        # Use the SAME Python interpreter running this script
        # python_exec = sys.executable

        # proc = subprocess.Popen([
        #     python_exec, "-m", "gunicorn.app.wsgiapp",
        #     "-w", "4",
        #     "-b", "0.0.0.0:8050",
        #     "wsgi:server",
        #     "-c", "gunicorn.conf.py"
        # ])
        # msg = f"Gunicorn started (Linux/Unix/macOS) with PID {proc.pid}" 
        subprocess.run(["pkill", "-f", "python"], check=False)
        subprocess.Popen([
            "/usr/bin/python3", "app/app.py"
        ])
        msg="Restart triggered!"
    return msg

import subprocess
import platform
import os

def start_server():
    system = platform.system()
    print('[[[[[[[[[[]]]]]]]]]]', system)

    if system == "Windows":
        venv_python = os.path.abspath(os.path.join("..", "dsa_venv", "Scripts", "python.exe"))
        proc = subprocess.Popen([venv_python, "run_app_win.py"])
        return f"Waitress started (Windows) with PID {proc.pid}"

    else:
        # The exact command you told me to run 
        '''
        subprocess.run(["pkill", "-f", "python"], check=False)
        subprocess.Popen([
            "/usr/bin/python3", "app/app.py"
        ])
        command = (
            '/usr/bin/python3 -m gunicorn -w 4 -b 0.0.0.0:8050 ',
            'wsgi:server --timeout 1200 -c gunicorn.conf.py'
        )

        subprocess.Popen([
            command
        ])
'''
        # # Escape quotes for AppleScript
        # safe = command.replace('"', '\\"')

        # # Open a NEW Terminal window and run the command
        # script = f'tell application "Terminal" to do script "{safe}"'

        # subprocess.Popen([
        #     "osascript",
        #     "-e",
        #     script
        # ])

        # venv_python = os.path.abspath(os.path.join("..", "dsa_venv", "Scripts", "/usr/bin/python3"))
        # proc = subprocess.Popen([venv_python, '-m gunicorn -w 4 -b 0.0.0.0:8050 ', 'wsgi:server --timeout 1200 -c gunicorn.conf.py']) 
        subprocess.run(["pkill", "-f", "python"], check=False)

        venv_python = os.path.abspath(os.path.join("..", "dsa_venv", "bin", "python3"))
# 
        proc = subprocess.Popen([venv_python, "app/app.py"]) 
        proc = subprocess.Popen([
            venv_python,
            "-m", "gunicorn",
            "-w", "4",
            "-b", "0.0.0.0:8050",
            "wsgi:server",
            "--timeout", "1200",
            "-c", "gunicorn.conf.py"
        ])



        return "Restart triggered!"





def async_restart():
    # start_server()
    # free_port()
   # start_server()
    #  free_port(8050)   # kills old Gunicorn workers
    start_server()    # launches new Gunicorn in a detached session
     # open_terminal_and_run()


def async_shutdown():
    free_port()

    
def restart_appsx():
    try:
        # Kill existing gunicorn/app processes
        subprocess.run(["pkill", "-f", "python"], check=False) 
        subprocess.Popen(["/usr/bin/python3", "app/app.py"])
        return "Restart triggered!"
    except Exception as e:
        return f"Error restarting: {e}"

 
import os
import dash
from dash import html, Input, Output, State, callback, ctx, ALL
from dend_fun_0.obj_get import parse_obj_upload
from dend_fun_0.main_0 import (
    app_run_param, algorithm, algorithm_param,
    get_data_all, get_dict_param, get_navs_bar
)
from dend_fun_0.help_funn import remove_directory
import numpy as np



def align_head(path_heads_show):
    headd = [
        'dnn_GINN_SM00000_LOC_AUG',
        'cnn_3UNet3D3_5000_hpcc_crop',
        'cnn_VGG16_FCN3D_5000_hpcc_crop',
        'cnn_VoxNetSeg_5000_hpcc_crop',
        'gcn_UNet_SM10000_LOC',
        'cml_cML',
    ]

    # Add new heads at the end
    for nn in path_heads_show:
        if nn not in headd:
            headd.append(nn)

    # Map head → priority index
    ghhg = {name: idx for idx, name in enumerate(headd)}

    # Sort path_heads_show by priority
    order = np.argsort([ghhg[name] for name in path_heads_show])

    return [path_heads_show[i] for i in order]




class DSAPage:

    def __init__(self,path_heads_show=None,categories=None,path_display=None):
        self.name = "dnn"
        self.page_path = "/DSA-2"
        self.categories = ["dsa", "dnn"]  if categories is None else categories
        self.path_display = ["dest_shaft_path"] if path_display is None else path_display
 

        # Load parameters
        path_heads_show = [
            "dnn_GINN_SM00000_LOC_AUG", 
        ] if path_heads_show is None else path_heads_show
        head_navbar = get_navs_bar(path_heads_show)

        dict_param = get_dict_param(
            n_step=0,
            head_navbar=head_navbar,
            path_heads_show=path_heads_show,
        )

        self.param = algorithm_param(**dict_param)
        self.mapp = app_run_param(self.param)
 
    def layout(self):
        return self.mapp.app_layout
 






    def param_toggle_all(self):
        outputs = [Output(f"collapse-{p}", "is_open") for p in self.categories]

        inputs = (
            [Input(f"btn-{p}", "n_clicks") for p in self.categories] +
            [Input({"type": "nav", "prefix": ALL, "path": ALL}, "n_clicks")]
        )

        states = [State(f"collapse-{p}", "is_open") for p in self.categories]

        return outputs, inputs, states, False

    def param_toggle_single(self, gval):
        return (
            Output(f"collapse-{gval}", "is_open"),
            Input(f"collapse-header-{gval}", "n_clicks"),
            State(f"collapse-{gval}", "is_open"),
            False
        )

    def param_upload(self):
        outputs = [
            Output("output-file-upload", "children"),
            Output("shared-data", "data")
        ]

        inputs = [
            Input("upload-data", "filename"),
            Input("reset-button", "n_clicks"),
            Input("restart-button", "n_clicks"),
            Input("shutdown-button", "n_clicks"),
        ] + [Input(idx, "value") for idx in self.mapp.Input_id]

        states = [
            State("upload-data", "contents"),
            State("upload-data", "filename"),
            State("upload-data", "last_modified"),
            State("id-destination", "value"),
        ]

        return outputs, inputs, states, False

    def param_run_algorithm(self):
        outputs = Output("run-status", "children")
        inputs = Input("run-button", "n_clicks")
        states = State("shared-data", "data")
        return outputs, inputs, states, True
 

    def toggle_all(self, args):
        total = len(self.categories)
        nav_clicks = args[total]
        states = args[-total:]
        triggered = ctx.triggered_id

        if isinstance(triggered, dict) and triggered.get("type") == "nav":
            return [False] * total

        if isinstance(triggered, str) and triggered.startswith("btn-"):
            clicked = triggered.replace("btn-", "")
            return [
                not state if prefix == clicked else False
                for prefix, state in zip(self.categories, states)
            ]

        return states

    def toggle_single(self, n_clicks, is_open):
        return not is_open if n_clicks else is_open

    def upload(self, args):
        filenames, reset_clicks, restart_clicks, shutdown_clicks, *rest = args
        values = rest
        param = self.mapp.rebuild_param(values)

        *other_values, contents, filenames, last_modified, objs_path_org = values

        os.makedirs(objs_path_org, exist_ok=True)
        export_dir = os.path.dirname(objs_path_org)
        nam = os.path.basename(objs_path_org)


        if shutdown_clicks and shutdown_clicks > 0:
            threading.Thread(target=async_shutdown).start()
            return html.Div("Shut Down requested"), {}

        if reset_clicks and reset_clicks > 0:
            # threading.Thread(target=async_restart).start() 
            remove_directory(export_dir)
            remove_directory(os.path.join(file_path_org,'data',nam))
            objs_path_app=os.path.join(file_path_org,'app','pages','test')
            for mm in [f for f in os.listdir(objs_path_app) if os.path.isdir(os.path.join(objs_path_app, f))]:
                pmm=os.path.join(objs_path_app,mm)
                for nn in [f for f in os.listdir(pmm) if os.path.isdir(os.path.join(pmm, f))]:
                    pmmm=os.path.join(pmm,nn)
                    pfmmm=os.listdir(pmmm)
                    for f in pfmmm: 
                        if f.endswith('.py'):  
                            if nam in f.split('_'):
                                mnnm=os.path.join(pmmm, f)
                                os.remove(os.path.join(pmmm, f))
                    for f in pfmmm:
                        nnn=os.path.join(pmmm, f)
                        if os.path.isdir(nnn): 
                            if f==nam:
                                remove_directory(nnn)
                                break
                    

            # remove_directory(os.path.join(file_path_org,'app','pages','test'))
            return html.Div(f"Reset requested"), {}


        if restart_clicks and restart_clicks > 0:
            threading.Thread(target=async_restart).start()
            return html.Div("Restart requested"), {}

        # if n_clicks == 0 and contents is None:
        #     raise dash.exceptions.PreventUpdate 

        # Prepare directories
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), "data", nam), exist_ok=True)

        nh = param["path_dir"]["param"]
        nhh = param["path_dirs_weig"]["param"][nh]
        param["Spine-Shaft Segm"]["param"]["weight"] = nhh

        dend_names_chld = {}

        # Upload case
        if isinstance(contents, list):
            dend_names = []
            for content, filename in zip(contents, filenames):
                filename, ext = os.path.splitext(filename)
                dend_path = os.path.join(objs_path_org, filename, "data")
                os.makedirs(dend_path, exist_ok=True)
                parse_obj_upload(content, filename, export_dir=dend_path)
                dend_names.append(filename)

            dend_names_chld[nam] = dict(
                dend_names=dend_names,
                obj_org_path=objs_path_org,
                weights=[[10., 0.81]],
            )
 
        else:
            dend_names_chld[nam] = dict(
                dend_names=[
                    f for f in os.listdir(objs_path_org)
                    if os.path.isdir(os.path.join(objs_path_org, f))
                ],
                obj_org_path=objs_path_org,
                weights=[[10., 0.81]],
            )
        objs_path_model=os.path.join(file_path_org,'model')
        path_heads_show=[f for f in os.listdir(objs_path_model) if os.path.isdir(os.path.join(objs_path_model, f))]
        # headd=[ 
        #                 'dnn_GINN_SM00000_LOC_AUG',
        #                 f'cnn_3UNet3D3_5000_hpcc_crop',
        #                 f'cnn_VGG16_FCN3D_5000_hpcc_crop',
        #                 f'cnn_VoxNetSeg_5000_hpcc_crop', 
        #                 'gcn_UNet_SM10000_LOC',
        #                 'cml_cML',
        #     ]
        # for nn in path_heads_show:
        #     if nn not in headd:
        #         headd.append(nn)
        # ghhg={va:ke for ke,va in enumerate(headd)} 
        # path_heads_show=[headd[ii] for ii in np.argmin(np.array([ghhg[ii] for ii in path_heads_show]))]
        path_heads_show=align_head(path_heads_show)
        print('[[[[[[[[[[[------------------]]]]]]]]]]]',path_heads_show)
        store_data = dict(
            param=param,
            nam=nam,
            export_dir=export_dir,
            dend_names_chld=dend_names_chld,
            path_heads_show=path_heads_show,
        )

        # Display
        if contents is None:
            display = html.Div([
                html.H5("No files uploaded yet.", style={'color': 'white'}),
                html.H5("Ready to execute files in Directory:", style={'color': 'white'}),
                html.Ul([
                    html.Li(name, style={'color': 'lightgray'})
                    for name in os.listdir(objs_path_org)
                    if os.path.isdir(os.path.join(objs_path_org, name))
                ])
            ])
        else:
            display = html.Div([
                html.H5("Uploaded Files", style={'color': 'white'}),
                html.Ul([html.Li(name) for name in filenames], style={'color': 'white'}),
            ])

        return display, store_data

    def run_algorithm(self, store_data):
        param = store_data["param"]
        dend_names_chld = store_data["dend_names_chld"]
        export_dir = store_data["export_dir"]
        nam = store_data["nam"]
        path_heads_show=store_data["path_heads_show"]

        nh = param["path_dir"]["param"]
        nhh = param["path_dirs_weig"]["param"][nh]
        param["Spine-Shaft Segm"]["param"]["weight"] = nhh

        gdas = get_data_all(names_dic=dend_names_chld, file_path_data=export_dir)
        dend_data = gdas.part(nam)

        head_navbar =get_navs_bar(path_heads_show)
 
        # param = algorithm_param(**dict_param)
        # mapp = app_run_param(param)
        # param=mapp.emerge_param()
        param['head_navbar']['param']=head_navbar
        alg = algorithm(param=param)
        alg.test(
            dend_data=dend_data,
            true_name="true_0",
            dnn_mode=param["dnn_mode"]["param"],
            model_type=param["path_head"]["param"],
            path_dir=param["path_dir"]["param"],
            data_dir=param["path_dir"]["param"], 
            path_display=self.path_display,
        )

        return html.H3("Completed!", style={'color': 'lightgreen', 'textAlign': 'center'})
