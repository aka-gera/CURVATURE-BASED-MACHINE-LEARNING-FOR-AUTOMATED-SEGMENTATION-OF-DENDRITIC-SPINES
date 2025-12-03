import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
import webbrowser, threading
import sys, os

# Helper for PyInstaller
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)



app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row([
                dbc.NavbarToggler(id="navbar-toggler"),


                    dbc.Nav([
                        dbc.NavLink(page["name"], href=page["path"])
                        for page in dash.page_registry.values()
                        if not page["path"].startswith(( 
                                                        '/test/ml',
                                                        '/test/cml',
                                                        '/test/pinn/',
                                                        '/test/rpinn',
                                                        '/test/gcn',
                                                        '/test/opt', 
                                                        '/train/pre', 
                                                        '/dsa', 
                                                        '/dsa/test',
                                                        ))
                    ])
            ])
        ],
        fluid=True, 
    ),
    dark=True,
    color='dark'
)

def open_browser():
    webbrowser.open('http://127.0.0.1:8050')

app.layout = dbc.Container([header, dash.page_container], fluid=False)
 
# Run the app
if __name__ == '__main__': 
    threading.Timer(1.25,open_browser).start()
    app.run(   debug=False)#, dev_tools_ui=False, dev_tools_props_check=False)
