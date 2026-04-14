import dash
from dash import html, dcc, Input, Output, State, callback 

import sys
import os
 
file_path_org=os.getcwd()  
sys.path.append(os.path.abspath(file_path_org))
  
# from app_run import app_run_param,algorithm 
from dend_fun_0.obj_get import parse_obj_upload
from  dend_fun_0.main_0 import app_run_param,algorithm,algorithm_param,get_data, get_data_all ,get_dict_param
from dend_fun_0.help_funn import remove_directory
import threading
import subprocess


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

 

import os
import signal
import subprocess
import platform

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




import sys
import subprocess
import platform
import os
import signal
import subprocess
 

 

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

        venv_python = os.path.abspath(os.path.join("..", "dsa_venv", "Scripts", "/usr/bin/python3"))
        proc = subprocess.Popen([venv_python, "app/app.py"]) 
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

     

name='pinn'
page_dir=f"/dsa-{name}"

page_name='DSA'
dash.register_page(
    __name__,
    title="DSA",
    name="Prediction",
    path="/DSA-2",
    order=0
)

dict_param=get_dict_param(
                    n_step = 0,
                    # nam=nam,
                    # weight=weight,
                    # size_threshold=size_threshold,
                    )


path_display = ['dest_shaft_path', ]
# param = algorithm_param()  
param = algorithm_param(**dict_param)
mapp = app_run_param(param)
def layout():
    return mapp.app_layout
 
 







from dash import callback, Output, Input, State, ctx, ALL

categories = ["dsa", "dnn",  "cnn", "gcn", "cml",]

@callback(
    [Output(f"collapse-{p}", "is_open") for p in categories],

    # Inputs: all buttons + all nav links
    [Input(f"btn-{p}", "n_clicks") for p in categories] +
    [Input({"type": "nav", "prefix": ALL, "path": ALL}, "n_clicks")],

    [State(f"collapse-{p}", "is_open") for p in categories],
)
def toggle_all(*args):
    total = len(categories)

    # First N inputs = button clicks
    button_clicks = args[:total]

    # Next input = list of nav clicks (pattern-matching)
    nav_clicks = args[total]

    # Last N inputs = collapse states
    states = args[-total:]

    triggered = ctx.triggered_id

    # If a nav link was clicked → close everything
    if isinstance(triggered, dict) and triggered.get("type") == "nav":
        return [False] * total

    # If a button was clicked
    if isinstance(triggered, str) and triggered.startswith("btn-"):
        clicked_prefix = triggered.replace("btn-", "")
        new_states = []
        for prefix, state in zip(categories, states):
            if prefix == clicked_prefix:
                new_states.append(not state)  # toggle clicked
            else:
                new_states.append(False)      # close others
        return new_states

    return states













for gval in list(set(param['param_input']['param'])):
    @callback(
        Output(f"collapse-{gval}", "is_open"),
        Input(f"collapse-header-{gval}", "n_clicks"),
        State(f"collapse-{gval}", "is_open"),
        prevent_initial_call=False
    )
    def toggle_collapse(n_clicks, is_open, gval=gval):
        if n_clicks:
            return not is_open
        return is_open

@callback(
    Output("collapse-starting", "is_open"),
    Input("toggle-text-starting", "n_clicks"),
    State("collapse-starting", "is_open"),
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@callback(
    Output("collapse-result", "is_open"),
    Input("toggle-text", "n_clicks"),
    State("collapse-result", "is_open"),
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# @callback(
#     Output("output-file-upload", "children"),
#     Output("shared-data", "data"),   # save to store
#    [Input("upload-data", "filename"), 
#      Input("restart-button", "n_clicks"),
#     Input("run-button", "n_clicks"),],
#      # # # Input("upload-data", "filename"), 
#     [State(idx, "value") for idx in mapp.Input_id] + [
#         State("upload-data", "contents"),
#         State("upload-data", "filename"),
#         State("upload-data", "last_modified"),
#         State("id-destination", "value"),
#     ],
#     prevent_initial_call=True
# )
@callback(
    [Output("output-file-upload", "children"),
    # Output("run-status", "children"),
     Output("shared-data", "data")],
    [Input("upload-data", "filename"),
     Input("reset-button", "n_clicks"),
     Input("restart-button", "n_clicks"),
     Input("shutdown-button", "n_clicks")],
    #  Input("run-button", "n_clicks") +
    [Input(idx, "value") for idx in mapp.Input_id],  
    [State("upload-data", "contents"),
     State("upload-data", "filename"),
     State("upload-data", "last_modified"),
     State("id-destination", "value")],
    prevent_initial_call=False
)
def display_filenames(filenames,reset_clicks,restart_clicks,shutdown_clicks,  *values):
    param=mapp.rebuild_param(values)

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


    os.makedirs(export_dir, exist_ok=True) 
    os.makedirs(os.path.join(file_path_org,'data',nam), exist_ok=True)  
    hhg=[f for f in os.listdir(objs_path_org) if os.path.isdir(os.path.join(objs_path_org, f))] 
    nh=param['path_dir']['param']
    nhh=param['path_dirs_weig']['param'][nh]
    param['Spine-Shaft Segm']['param']['weight']=nhh 
    dend_names_chld = {} 
    # objs_path_org = os.path.join(export_dir, nam)  # Data directory, e.g. file_path_data/nam/ filename,
    if isinstance(contents, list):
        dend_names=[]
        for content,filename in zip(contents,filenames): 
            filename, ext = os.path.splitext(filename)
            dend_path_original_new = os.path.join(objs_path_org,filename, 'data')  
            os.makedirs(dend_path_original_new, exist_ok=True)
            mesh=parse_obj_upload(content, filename, export_dir=dend_path_original_new )
            dend_names.append(filename) 
        dend_names_chld[nam] = dict(  
            dend_names=dend_names, 
            obj_org_path=objs_path_org,
            weights=[[10., 0.81]],   # ML shaft/spine parameter: decrease the first value to increase spine vertices
        )  
 
    else:  
        dend_names_chld[nam] = dict(  
            dend_names=[f for f in os.listdir(objs_path_org) if os.path.isdir(os.path.join(objs_path_org, f))], 
            obj_org_path=objs_path_org,
            weights=[[10., 0.81]],   # ML shaft/spine parameter: decrease the first value to increase spine vertices
        )  
 
    store_data = {
        "param": param, 
        "nam": nam, 
        "export_dir": export_dir,
        'dend_names_chld':dend_names_chld,
    }
 
    if contents is None:
        display = html.Div([
            html.H5("No files uploaded yet.", style={'color': 'white'}),
            html.H5("Ready to execute file in Directory:", style={'color': 'white'}),
            html.Ul([
                html.Li(name, style={'color': 'lightgray'})
                for name in os.listdir(objs_path_org)
                if os.path.isdir(os.path.join(objs_path_org, name))
            ])
        ], style={'marginTop': 20})
    else:
        display = html.Div([
            html.H5("Uploaded Files", style={'color': 'white'}),
            html.Ul([html.Li(name) for name in filenames], style={'color': 'white'}),
            # html.H5(f"--------{param['Smooth']}"),
            # html.H4(f"--------{[param[gval]['param']    for gval in param['param_dropdown']['param']]}"),
            
        ]) 

    return  display, store_data


@callback(    
    # [  # status message
    #  mapp.Output],  
    Output("run-status", "children"), 
    Input("run-button", "n_clicks"),
    State("shared-data", "data"),
    prevent_initial_call=True
)
def update_output(n_clicks, store_datas):
    if n_clicks == 0 or not store_datas:
        raise dash.exceptions.PreventUpdate
    status = "Running..."
    store_data=store_datas.copy()
    store_datas={}
    param = store_data["param"] 
    dend_names_chld = store_data["dend_names_chld"] 
    export_dir = store_data["export_dir"] 
    nam = store_data["nam"]     
    nh=param['path_dir']['param']
    nhh=param['path_dirs_weig']['param'][nh]
    param['Spine-Shaft Segm']['param']['weight']=nhh
    # dict_param=get_dict_param(nam=nam,
    #                     n_step = 0,
    #                     # weight=weight,
    #                     # size_threshold=size_threshold,
    #                     )
    gdas=get_data_all(names_dic=dend_names_chld,file_path_data=export_dir)
    dend_data = gdas.part(nam)  
    alg=algorithm(param=param,)  
    _ = alg.test(
        dend_data=dend_data,
        true_name='true_0',
        dnn_mode=param['dnn_mode']['param'],
        model_type=param['path_head']['param'],
        path_dir=param['path_dir']['param'],
        path_display=path_display,
    # path_dir=path_dir,
    # data_dir=data_dir,
    # **dict_param
    )
     
    status = html.H3("Completed!", style={'color': 'lightgreen', 'textAlign': 'center'})

    return status





