import dash 
import dash_bootstrap_components as dbc 
from dash import html, dcc 
style={
        'color': 'black',
        'backgroundColor': 'grey',
        'height': '100vh'  # Full viewport height
    },
style_equation={'textAlign': 'center', 'color': 'white' }

   

def sidebar(page_dir):
    nav_links = []
    
    for page in dash.page_registry.values():
        if page["path"].startswith(page_dir):
        # if page["path"].find(page_dir) == 0: 
            display_name = page["name"]
        elif page["path"] == "/projects":
            display_name = "App 1"
        else:
            continue  # Skip unrelated pages
        
        nav_links.append(
            dbc.NavLink(
                html.Div(display_name, className="text-center"),
                href=page["path"],
                active="exact",
            )
        )

    return dbc.Nav(
        children=nav_links,
        vertical=True,
        pills=True,
        className="bg-dark p-2",
    )

  
 

 
def layout_1(page_dir):
    return html.Div([
    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar(page_dir=page_dir)
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
 
        html.Br(),
        html.H1('Dendritic Spines Analysis',
                style={'textAlign': 'center', 'color': 'white', 'font-size': 40}),
    html.Hr(),
        html.Br(),
        html.Br(),

  
              
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        
        ]
    )
])
           

 


def get_dnn(page_dir,page_container):
    return html.Div([
    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar(page_dir=page_dir)
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
 
        html.Br(),
        html.H1('Dendritic Spines Analysis',
                style={'textAlign': 'center', 'color': 'white', 'font-size': 40}),
    html.Hr(),
        html.Br(),
        html.Br(),
        dcc.Markdown(
            page_container,
        )
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        
        ]
    )
])



def get_text_dash_dnn(page_name,page_dir_txt,dash_pages_path,
                        disp_infos=False, 
                        path_train=None, 
                        path_file=None, 
                        pinn_dir_data=None,
                        dend_data=None,
                        index=None,
                        page_view=None,
                        ): 
    code_content = f"""

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from dend_fun_0.side_bar import sidebar ,get_dnn,dnn_page 


page_dir= '/{page_dir_txt}' 
page_name='{page_name}'  
page_name_view=dnn_page()['{page_view}']
dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return get_dnn(page_dir,page_name_view)
""" 
    with open(dash_pages_path, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")

 

def dnn_page():
    page={} 




    page['starting']= """
    
 
 ## Getting Started with DSA Segmentation 

- Click on **DSA** in the top‑left corner.  
- Drag one or multiple `.obj` meshes into the segmentation box.  
  - If you don’t provide meshes, two demo meshes included in the repository will be used.  
  - The parent directory of meshes is shown in the **“Add Destination”** box. Update this path if needed.  

--- 
 
## Parameters & Options

- **Model Architecture**  
  Use the dropdowns to select the segmentation model.  
  - `DNN-3` generally performs better.  

- **Smoothing (Smooth box)**  
  - Enable/disable smoothing for new meshes.  
  - Check the **“Smooth”** box to activate.  
  - Clicking the label opens parameter options (e.g., step size, total steps).  

- **Resizing (Resize box)**  
  - Enable/disable resizing for large meshes (>600,000 vertices).  
  - Clicking the label opens parameter options for resizing.  

- **Spine–Shaft Segm**  
  - Click **“spine-shaft segm”** to open a dropdown.  
  - Adjust weights or thresholds to classify spines.  
  - Enable the checkbox to run segmentation.  

- **Morphologic Parameters (Morphologic Param)**  
  - Enable/disable to perform head–neck segmentation.  
  - Computes head/neck diameter and length.  

- **Clean Path Directory**  
  - Enable the **“clean_path_dir”** box to clear previous output directories before running new segmentation.  

- **Run Analysis**  
  -Double Click to run the segmentation after all parameters are set up.
  
- **Restart**  
  -Click the **Restart** button to kill and restart the process.
 
--- 
 

    """















    page['instructions']= """
    

---

# Instructions
---


- Click on **DSA** in the top‑left corner.  
- Drag one or multiple `.obj` meshes into the segmentation box.  
  - If you don’t provide meshes, two demo meshes included in the repository will be used.  
  - The parent directory of meshes is shown in the **“Add Destination”** box. Update this path if needed.  

---

## Parameters & Options

- **Model Architecture**  
  Use the dropdowns to select the segmentation model.  
  - `DNN-3` generally performs better.  

- **Smoothing (Smooth box)**  
  - Enable/disable smoothing for new meshes.  
  - Check the **“Smooth”** box to activate.  
  - Clicking the label opens parameter options (e.g., step size, total steps).  

- **Resizing (Resize box)**  
  - Enable/disable resizing for large meshes (>600,000 vertices).  
  - Clicking the label opens parameter options for resizing.  

- **Spine–Shaft Segm**  
  - Click **“spine-shaft segm”** to open a dropdown.  
  - Adjust weights or thresholds to classify spines.  
  - Enable the checkbox to run segmentation.  

- **Morphologic Parameters (Morphologic Param)**  
  - Enable/disable to perform head–neck segmentation.  
  - Computes head/neck diameter and length.  

- **Clean Path Directory**  
  - Enable the **“clean_path_dir”** box to clear previous output directories before running new segmentation.  

- **Run Analysis**  
  -Double Click to run the segmentation after all parameters are set up.
  
- **Restart**  
  -Click the **Restart** button to kill and restart the process.
 
 
    
    
    """



    page['restart']= """
    
    
 

Once segmentation has terminated, you can check the results by following these steps:  

---

## Restart the Application  

1. Kill the current running code in the terminal using **`Ctrl + C`**.  
2. Reopen the application by running:  

```bash
python -m  gunicorn -w 4 -b 0.0.0.0:8050 wsgi:server -c gunicorn.conf.py
```
3. Alternatively, use the **Restart button** in the interface and then refresh your browser page.
 
    """


    page['results']= """
     

---
 
## Navigating the Interface

- On the right of **DSA** (top corner), click on **“PINN”** if the **PINN** method was used.  
- Select the **architecture (DNN-nth)** that was used.  
- Click to choose the **file parent name**.  
- Click on the **mesh name**.  
- A new page will appear with visualization options.  
 
    """

    page['visualization']= """


## Visualization Options

Use the dropdown buttons to explore different features:

- **Mesh Selection**  
  - Switch between multiple meshes if more than one was dropped.  

- **Method Selection**  
  - Choose the method used (e.g., **PNN**, **cML**) if available.  

- **Architecture Selection**  
  - Select the **DNN-nth** architecture.  

- **Skeleton Visualization**  
  - Choose skeleton view.  
  - **model_sp_iou** → visualize Jaccard index.  
  - **model_sp_loss** → visualize training loss.  
  
- **SHAP Coefficients**  
  - **model_shap** → visualize SHAP coefficients using [Shapley values](https://en.wikipedia.org/wiki/Shapley_value).

- **Morphologic Parameters**  
  - Plot morphologic parameters against each other.  
  - Visualize **heatmap** and **cylindrical heatmap** showing spine distribution on dendrite segments.  

---

## Feature Visualization

Features used in training/testing can be visualized:

- Distance shaft skeleton vertices  
- Regionalization:  
  - With smoothing (**kmean_n**)  
  - Without smoothing (**kmean_mean_n**)  
- Gaussian curvature and mean curvature (and their squares):  
  - With smoothing (**mean**, **gauss**)  
  - Without smoothing (**imean**, **igauss**)  

---

## Additional Dropdowns

- Switch view between **dendrite segment**, **shaft**, and **spines segmented**.  
- Under **image/skeleton**, visualize dendrite segment parts.  
- Toggle between **Smoothed** and **Initial (non-smoothed)** meshes.  
- Change page theme: **seaborn**, **plotly_dark**, etc.  

---

## Graph Controls

- Adjust histogram bin count using the **radio bar**.  
- Modify **graph width** and **height** for better visualization.  

    
    """

    return page




def get_text_dash_train(user_input,file_path_org,dend_path_inits,   dend_name, dend_namess, data_studied, model_sufix,dash_pages_name,disp_infos=False,path=None):
    # Generate the Python script content
    code_content = f"""
import dash
from dash import callback  

import sys
import os

file_path_org = os.getcwd() 
     
sys.path.append(os.path.abspath(os.path.join(file_path_org,'dend_fun')))

from app_param_4 import app_param 


dend_names = ['{dend_name}']
dend_namess = {dend_namess}
dend_path_inits =['{dend_path_inits}']
data_studied = '{data_studied}'  
model_sufix = '{model_sufix}' 

mapp = app_param(
    file_path_org=file_path_org,
    model_sufix=model_sufix, 
    data_studied=data_studied,
    dend_names=dend_names,
    dend_namess=dend_namess, 
    dend_path_inits=dend_path_inits,  
    dropdow_path={path},
)

title_dend_name = f'{dend_name}'
dash.register_page(__name__, title=title_dend_name, name=title_dend_name, order={user_input}) 

def layout():
    return mapp.app_layout

@callback(
    mapp.Output,
    mapp.Input, 
    prevent_initial_call=mapp.prevent_initial_call
)
def update_output(mode, dendd, clusts, metric, intensity_type, width, height, iou_per, templ,index): 
    figure, txt = mapp.Get_output(mode, dendd, clusts, metric, intensity_type, width, height, iou_per, templ,index)
    return figure, txt
"""
 
    with open(dash_pages_name, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_name}")

 
def get_text_dash_test(user_input,file_path_org,dend_path_inits,   dend_name, dend_namess, data_studied, model_sufix,dash_pages_name,
                       disp_infos=False,
                       path_train=None, 
                            path_file=None, 
                    pinn_dir_data=None,
                    dend_data=None,
                    index=None,
                    model_type=None,
                    obj_org_path_dict=None,
                    model_sufix_dic=None,):
    # Generate the Python script content
    code_content = f""" 

import os, sys ,dash 
file_path_org=os.getcwd()
sys.path.append(file_path_org ) 
from  dend_fun_0.app_param_test import app_param
from dash import callback  


dend_names = ['{dend_name}']
dend_namess = {dend_namess}
dend_path_inits =['{dend_path_inits}']
data_studied = '{data_studied}'  
model_sufix = '{model_sufix}' 
path_train= {path_train}
pinn_dir_data='{pinn_dir_data}'
dend_data={dend_data}
index={index}
model_type='{model_type}'
obj_org_path_dict={obj_org_path_dict}
model_sufix_dic={model_sufix_dic}
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

title_dend_name = f'{dend_name}'
dash.register_page(__name__, title=title_dend_name, name=title_dend_name, order={user_input}) 

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
""" 
    with open(dash_pages_name, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_name}")

 

 


def get_text_dash_all(page_name,page_dir_txt,dash_pages_path,
                    disp_infos=False,
                    path_train=None, 
                    path_file=None, 
                    pinn_dir_data=None,
                    dend_data=None,
                    index=None,
                    ): 
    code_content = f"""

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from  dend_fun_0.side_bar import layout_1


page_dir= '/{page_dir_txt}' 
page_name='{page_name}'
dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return layout_1(page_dir)
""" 
    with open(dash_pages_path, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")



 