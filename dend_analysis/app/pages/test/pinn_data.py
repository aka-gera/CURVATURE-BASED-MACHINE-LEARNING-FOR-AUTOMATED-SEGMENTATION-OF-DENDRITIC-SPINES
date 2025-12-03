

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from dend_fun_0.side_bar import sidebar ,get_dnn


page_dir= '/test/pinn/pinn-data-' 
page_name='PINN'
dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return get_dnn(page_dir)
