 

import dash_bootstrap_components as dbc
from dash import html, dcc
import dash,os
 

style={
        'color': 'black',
        'backgroundColor': 'grey',
        'height': '100vh'  # Full viewport height
    },
style_equation={'textAlign': 'center', 'color': 'white' }
 




'''




def sidebar_grouped_dropdown():
    # Define categories and their prefixes
    categories = {
        "DNN": "dnn",
        "Volume": "vol",
        "Panel": "pnel",
    }

    # Prepare container for grouped links
    grouped_links = {cat: [] for cat in categories}

    # Loop through Dash pages
    for page in dash.page_registry.values():
        path = page["path"].lstrip("/")  # remove leading slash
        name = page["name"]

        # Assign page to its category
        for cat, prefix in categories.items():
            if path.startswith(prefix):
                grouped_links[cat].append(
                    dbc.NavLink(
                        name,
                        href=page["path"],
                        active="exact",
                        className="ms-3 mb-1"
                    )
                )

    # Build accordion items
    accordion_items = []
    for cat, links in grouped_links.items():
        accordion_items.append(
            dbc.AccordionItem(
                children=links if links else html.Div("No pages", className="text-muted ms-3"),
                title=cat
            )
        )

    return html.Div(
        [
            dbc.Accordion(
                accordion_items,
                start_collapsed=True,
                flush=True,
                always_open=False,
            )
        ],
        className="bg-dark p-3",
        style={"height": "100vh", "overflow-y": "auto"}
    )









def sidebar(page_dir):
    nav_links = []
    
    for page in dash.page_registry.values():
        # if page["path"].startswith(page_dir):
        if page["path"]==page_dir:
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
'''
def sidebar(page_dir,forbidden=None):
    nav_links = []

    for page in dash.page_registry.values():
        path = page["path"] 
        if not path.startswith(page_dir):
            continue
        if forbidden is not None:
            if not path.endswith(forbidden):
                continue
        display_name = page["name"] 
        diff = path[len(page_dir):] 

        if '/' in diff: 
            if not diff.startswith('/'):
                continue
 
             
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



 
def layout_1(page_dir,forbidden=None,):
    return html.Div([
    dbc.Row(
        [
            # dbc.Col(
            #     [
            #         sidebar(page_dir=page_dir,forbidden=forbidden,) 
            #     ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),
            dbc.Col(
                [
                    sidebar(page_dir=page_dir, forbidden=forbidden,)
                ],
                xs=12, sm=6, md=4, lg=4, xl=4, xxl=4
            ),

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
           

 


def get_dnn(page_dir,page_container,forbidden=None,):
    return html.Div([
    dbc.Row(
        [
            # dbc.Col(
            #     [
            #         sidebar(page_dir=page_dir)
            #     ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2)
            dbc.Col(
                [
                    sidebar(page_dir=page_dir,forbidden=forbidden,)
                ],
                xs=4, sm=4, md=2, lg=2, xl=2, xxl=2,
                style={
                    "minWidth": "220px",
                    "maxWidth": "260px",
                    "whiteSpace": "normal",
                    "wordWrap": "break-word",
                    "flexShrink": 0
                }
            )
            ,

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
                        forbidden_endswith=None,
                        ): 
    code_content = f"""

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from dend_fun_0.side_bar import sidebar ,get_dnn,dnn_page 


page_dir= '/{page_dir_txt}' 
page_name='{page_name}'  
page_name_view=dnn_page()['{page_view}']
forbidden_endswith={forbidden_endswith} 
forbidden_endswith = None if forbidden_endswith in (None, 'None') else forbidden_endswith


dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return get_dnn(page_dir,page_name_view,forbidden_endswith,)
""" 
    with open(dash_pages_path, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")


 

def get_text_dash_app(dash_pages_path,
                      disp_infos=False, 
                      head_navbar=None,
                      forbidden_page = "",
                      forbidden_endswith=None, ):

    code_content = f"""
import os, sys, dash
import dash_bootstrap_components as dbc
from dash import Dash, html
import webbrowser, threading
 
forbidden_page={forbidden_page}
forbidden_endswith='{forbidden_endswith}'
forbidden_endswith = None if forbidden_endswith in (None, 'None') else forbidden_endswith

 
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

def header_navbar(groups=None,forbidden_endswith=None,):
    if groups is None:
        groups = {{
            "DSA": "dsa",
            "DNN": "dnn", 
            "3D CNN": "cnn",
            "GCN": 'gcn',
            "class ML":'cml',
        }}

    # Group pages by prefix
    grouped = {{prefix: [] for prefix in groups.values()}}

    for page in dash.page_registry.values():
        if page["path"].startswith(forbidden_page):
            continue
        raw_path = page["path"].lower().rstrip("/")
        nlast = raw_path.split("/")[-1].split("-") 
        last,lasti=nlast[0],nlast[-1]

        name = page["name"] 
        for label, prefix in groups.items():
            if last.startswith(prefix) and lasti.endswith(('data','test','train','dsa','2',)):
                grouped[prefix].append(
                    dbc.NavLink(
                        name,
                        href=page["path"],
                        className="ms-2",
                        active="exact"
                    )
                )

    horizontal_groups = []

    for label, prefix in groups.items():
        horizontal_groups.append(
            html.Div(
                [
                    # BUTTON stays fixed in place
                    dbc.Button(
                        label,
                        id=f"btn-{{prefix}}",
                        color="secondary",
                        className="px-2 py-1",
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Nav(
                                    grouped[prefix],
                                    vertical=True,
                                    pills=True,
                                )
                            ),
                            className="mt-1"
                        ),
                        id=f"collapse-{{prefix}}",
                        is_open=False,
                    ),
                ], className="d-flex flex-column", #",#
                # style={{"margin-right": "1cm"}} ,
            )
        )

    return dbc.Navbar(
        dbc.Container(
                    dbc.Stack(
            horizontal_groups,
            gap=0, 
            direction="horizontal",
        ),
 
            fluid=True,
            className="d-flex flex-row align-items-start p-0 m-0",
            style={{"margin-right": "0.1cm"}} ,
        ),
        dark=True,
        color="dark",
        className="p-1"
    )




head_navbar={head_navbar}
header=header_navbar(head_navbar,forbidden_endswith)



def open_browser():
    webbrowser.open('http://127.0.0.1:8050/DSA-2')

app.layout = dbc.Container([header, dash.page_container], fluid=False)

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=False)
"""

    with open(dash_pages_path, "w") as file:
        file.write(code_content)

    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")





def dnn_page():
    page={} 




    page['starting']= """
    
 
 ## Getting Started with DSA Segmentation 

- Click on **DSA** in the top-left corner.  
- Drag one or multiple `.obj` meshes into the segmentation box.  
  - If you don't provide meshes, two demo meshes included in the repository will be used.  
  - The parent directory of meshes is shown in the **“Add Destination”** box. Update this path if needed.  

--- 

## Parameters & Options

- **Model Architecture**  
  Use the first two dropdowns to select the segmentation model.
  - The first dropdown selects the **type of model**:  
    - **GINN** — *Geometric-Informed Neural Network*  
    - **CNN** — *Convolutional Neural Network*  
    - **GCN** — *Graph Convolutional Neural Network*  
    - **cML** — *Classic Machine Learning*  
  - The second dropdown selects the **feature class**.  
    Selecting **GINN** with **DNN3** generally performs better.
  - The third dropdown selects the **weights** used to classify spines.


- **Smoothing (Smooth box)**  
  - Enable/disable smoothing for new meshes.  
  - Check the **“Smooth”** box to activate.  
  - Clicking the label opens parameter options (e.g., step size, total steps, method).  

- **Resizing (Resize box)**  
  - Enable/disable resizing for large meshes (>600,000 vertices).  
  - Clicking the label opens parameter options for resizing.  

- **Spine-Shaft Segm**  
  - Click **“spine-shaft segm”** to open a dropdown.  
  - Enable the checkbox to run segmentation.  
  - Adjust thresholds to classify spines.  

- **Morphologic Parameters (Morphologic Param)**  
  - Enable/disable to perform head-neck segmentation.  
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


- Click on **DSA** in the top-left corner.  
- Drag one or multiple `.obj` meshes into the segmentation box.  
  - If you don't provide meshes, two demo meshes included in the repository will be used.  
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

- **Spine-Shaft Segm**  
  - Click **“spine-shaft segm”** to open a dropdown.  
  - Adjust weights or thresholds to classify spines.  
  - Enable the checkbox to run segmentation.  

- **Morphologic Parameters (Morphologic Param)**  
  - Enable/disable to perform head-neck segmentation.  
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

- On the right side of **DSA** (top corner), click on one of the following to visualize the results:
    - **GINN** — *Geometric-Informed Neural Network*  
    - **CNN** — *Convolutional Neural Network*  
    - **GCN** — *Graph Convolutional Neural Network*  
    - **cML** — *Classic Machine Learning*  
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
  - Choose the method used (e.g., **GINN**, **cML**) if available.  

- **Weight Selection**  
  - Choose the weight used during the preditction.

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
                    path_file_dir=None,
                    path_file_sub=None, 
                    pinn_dir_data=None,
                    dend_data=None,
                    index=None,
                    model_type=None,
                    obj_org_path_dict=None,
                    model_sufix_dic=None,
                    path_display=None,
                    path_display_dic=None,):
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
path_file= {path_file}
path_file_sub={path_file_sub}
path_file_dir='{path_file_dir}'
pinn_dir_data='{pinn_dir_data}'
dend_data={dend_data}
index={index}
model_type='{model_type}'
obj_org_path_dict={obj_org_path_dict}
model_sufix_dic={model_sufix_dic}
path_display={path_display}
path_display_dic={path_display_dic}
mapp = app_param(
    file_path_org=file_path_org,
    model_sufix=model_sufix,
    path_train=path_train, 
    path_file=path_file,
    path_file_sub=path_file_sub,
    path_file_dir=path_file_dir,
    pinn_dir_data=pinn_dir_data,
    index=index, 
    data_studied=data_studied,  
    dend_data=dend_data,
    model_type=model_type,
    obj_org_path_dict=obj_org_path_dict,
    model_sufix_dic=model_sufix_dic,
    path_display=path_display,
    path_display_dic=path_display_dic,
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



 