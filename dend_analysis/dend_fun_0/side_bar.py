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

   ####################################################### 1  ##########################################################  
        dbc.Row([
            # Column 1: Bending Energy Theory
dbc.Col([ 
    dcc.Markdown('### Smoothing Curvature', style={'textAlign': 'center', 'color': 'white' }), 

    html.Hr(),
    dcc.Markdown(
        '$$\\mathbf{W}_{bend}(\\mathbf{X}) = \\frac{k_{bend}}{2} \\int_{\\mathcal{A}} H^2(\\mathbf{X}) \, d\\mathcal{A}$$', 
        style=style_equation,
        mathjax=True
    ),


    # Combine the text into one string for each Markdown component
    dcc.Markdown(
        '''
      where   $k_{bend}$ is  the bending coefficient over a surface $\\mathcal{A}$.
    ''', mathjax=True
    ),

    dcc.Markdown(
    '''
    $H(\mathbf{X})$ is the mean curvature.
    ''', mathjax=True
    ),

    dcc.Markdown('The bending force is the gradient of the bending energy:'),

    dcc.Markdown(
        '$$\\mathbf{F}_{bend}(\\mathbf{X}) = -{\\nabla}_{\\mathbf{X}} \\mathbf{W}_{bend}(\\mathbf{X})$$', 
        style=style_equation,
        mathjax=True
    ),

    dcc.Markdown('Smoothing the curvature using Willmore flow minimization:'),

    dcc.Markdown(
        '$$\\frac{\\partial}{\\partial t} \\mathbf{X} = \\mathbf{F}_{bend}(\\mathbf{X})$$', 
        style=style_equation,
        mathjax=True
    )
], width=3),
#################################################### 2 ##########################################
dbc.Col([ 
    dcc.Markdown('### Clusterization', style={'textAlign': 'center', 'color': 'white' }), 

    html.Hr(),
    # Combine the text into one string for each Markdown component
    
    dcc.Markdown(
        '''
      Sigmoid function to enhance Gaussian curvature  
    ''', mathjax=True
    ),
    dcc.Markdown(
        '$$\\zeta(x)=\\frac{1}{1+\\exp{(-x)}}$$', 
        style=style_equation,
        mathjax=True
    ),

    dcc.Markdown(
        '$$\\bar{K}(\mathbf{X})=\zeta(a  K(\mathbf{X})+b)$$', 
        style=style_equation,
        mathjax=True
    ),

    dcc.Markdown(
    '''
    $K(\mathbf{X})$   the Gaussian curvature
    ''', mathjax=True
    ),
    dcc.Markdown(
    '''
    $a$ and $b$  are empirical parameters 
    ''', mathjax=True
    ),
    
    html.Hr(),


    dcc.Markdown('Use K-mean to classify the curvature values'),
    dcc.Markdown(
        '$$\\hat{K}(\mathbf{X})=K-mean(\\bar{K}(\mathbf{X}))$$', 
        style=style_equation,
        mathjax=True
    ),

    dcc.Markdown(
        '$$\\hat{H}(\mathbf{X})=K-mean({H(\mathbf{X})})$$', 
        style=style_equation,
        mathjax=True
    ),
    
 


 
], width=3),


######################################################### 3 #################################################################
dbc.Col([ 
    dcc.Markdown('### Segmentation', style={'textAlign': 'center', 'color': 'white'}), 

    html.Hr(),
    
    dcc.Markdown(
        '''
        1. Construction of concentric spheres with gradually increasing radii.
        2. Centroid of the spheres is set at the unique faces of each cluster.
        3. Remove the faces of the cluster that are within the inner sphere (i.e., remove branch faces).
        4. Repeat the process for further segmentation.
        ''' 
    ),
  
], width=3),

 
html.Iframe(src='/assets/nnk.pdf', style={'width': '100%', 'height': '600px'}),

      
        ],justify='center'
        ),

 
              
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        
        ]
    )
])
           

 


def get_dnn(page_dir):
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

   ####################################################### 1  ##########################################################  
        dbc.Row([
            # Column 1: Bending Energy Theory
dbc.Col([ 
    dcc.Markdown('### Smoothing Curvature', style={'textAlign': 'center', 'color': 'white' }), 

    html.Hr(),
    dcc.Markdown(
        '$$\\mathbf{W}_{bend}(\\mathbf{X}) = \\frac{k_{bend}}{2} \\int_{\\mathcal{A}} H^2(\\mathbf{X}) \, d\\mathcal{A}$$', 
        style=style_equation,
        mathjax=True
    ),


    # Combine the text into one string for each Markdown component
    dcc.Markdown(
        '''
      where   $k_{bend}$ is  the bending coefficient over a surface $\\mathcal{A}$.
    ''', mathjax=True
    ),

    dcc.Markdown(
    '''
    $H(\mathbf{X})$ is the mean curvature.
    ''', mathjax=True
    ),

    dcc.Markdown('The bending force is the gradient of the bending energy:'),

    dcc.Markdown(
        '$$\\mathbf{F}_{bend}(\\mathbf{X}) = -{\\nabla}_{\\mathbf{X}} \\mathbf{W}_{bend}(\\mathbf{X})$$', 
        style=style_equation,
        mathjax=True
    ),

    dcc.Markdown('Smoothing the curvature using Willmore flow minimization:'),

    dcc.Markdown(
        '$$\\frac{\\partial}{\\partial t} \\mathbf{X} = \\mathbf{F}_{bend}(\\mathbf{X})$$', 
        style=style_equation,
        mathjax=True
    )
], width=3),
#################################################### 2 ##########################################
dbc.Col([ 
    dcc.Markdown('### Clusterization', style={'textAlign': 'center', 'color': 'white' }), 

    html.Hr(),
    # Combine the text into one string for each Markdown component
    
    dcc.Markdown(
        '''
      Sigmoid function to enhance Gaussian curvature  
    ''', mathjax=True
    ),
    dcc.Markdown(
        '$$\\zeta(x)=\\frac{1}{1+\\exp{(-x)}}$$', 
        style=style_equation,
        mathjax=True
    ),

    dcc.Markdown(
        '$$\\bar{K}(\mathbf{X})=\zeta(a  K(\mathbf{X})+b)$$', 
        style=style_equation,
        mathjax=True
    ),

    dcc.Markdown(
    '''
    $K(\mathbf{X})$   the Gaussian curvature
    ''', mathjax=True
    ),
    dcc.Markdown(
    '''
    $a$ and $b$  are empirical parameters 
    ''', mathjax=True
    ),
    
    html.Hr(),


    dcc.Markdown('Use K-mean to classify the curvature values'),
    dcc.Markdown(
        '$$\\hat{K}(\mathbf{X})=K-mean(\\bar{K}(\mathbf{X}))$$', 
        style=style_equation,
        mathjax=True
    ),

    dcc.Markdown(
        '$$\\hat{H}(\mathbf{X})=K-mean({H(\mathbf{X})})$$', 
        style=style_equation,
        mathjax=True
    ),
    
 


 
], width=3),


######################################################### 3 #################################################################
dbc.Col([ 
    dcc.Markdown('### Segmentation', style={'textAlign': 'center', 'color': 'white'}), 

    html.Hr(),
    
    dcc.Markdown(
        '''
        1. Construction of concentric spheres with gradually increasing radii.
        2. Centroid of the spheres is set at the unique faces of each cluster.
        3. Remove the faces of the cluster that are within the inner sphere (i.e., remove branch faces).
        4. Repeat the process for further segmentation.
        ''' 
    ),
  
], width=3),

 
html.Iframe(src='/assets/nnk.pdf', style={'width': '100%', 'height': '600px'}),

      
        ],justify='center'
        ),

 
              
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        
        ]
    )
])
           



