 
from dash import callback
import sys,os,dash
 
sys.path.append(os.path.abspath(os.getcwd()))

from dend_fun_0.help_app import DSAPage

dash.register_page(
    __name__,
    title="DSA",
    name="Prediction",
    path="/DSA-2",
    order=0
)

path_heads_show= ['dnn_GINN_SM00000_LOC_AUG']
categories= ['dsa', 'dnn', ]
path_display= ['dest_shaft_path']  

# Instantiate page
dsa_page = DSAPage(
    path_heads_show=path_heads_show,
    categories=categories,
    path_display=path_display
)

layout = dsa_page.layout
 
out, inp, st, prevent = dsa_page.param_toggle_all()

@callback(
    *out,
    *inp,
    *st,
    prevent_initial_call=prevent
)
def toggle_all(*args):
    return dsa_page.toggle_all(args)
 
for gval in list(set(dsa_page.param["param_input"]["param"])):
    out, inp, st, prevent = dsa_page.param_toggle_single(gval)

    @callback(
        out,
        inp,
        st,
        prevent_initial_call=prevent
    )
    def toggle_single(n_clicks, is_open, gval=gval):
        return dsa_page.toggle_single(n_clicks, is_open)
 
out, inp, st, prevent = dsa_page.param_upload()

@callback(
    *out,
    *inp,
    *st,
    prevent_initial_call=prevent
)
def callback_upload(*args):
    return dsa_page.upload(args)
 
out, inp, st, prevent = dsa_page.param_run_algorithm()

@callback(
    out,
    inp,
    st,
    prevent_initial_call=prevent
)
def callback_run_algorithm(n_clicks, store_data):
    if not n_clicks or not store_data:
        raise dash.exceptions.PreventUpdate
    return dsa_page.run_algorithm(store_data)
