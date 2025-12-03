

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from dend_fun_0.side_bar import sidebar ,get_dnn


page_dir= '/test/pinn/pre-gg2n2n3n4n5n6n7n8n9n10/pinn-data-pre-gg2n2n3n4n5n6n7n8n9n10' 
page_name='DNN-3'
dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return get_dnn(page_dir)
