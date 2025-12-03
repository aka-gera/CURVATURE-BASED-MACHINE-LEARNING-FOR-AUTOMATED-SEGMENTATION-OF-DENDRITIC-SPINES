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
 
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        
        ]
    )
])
           



