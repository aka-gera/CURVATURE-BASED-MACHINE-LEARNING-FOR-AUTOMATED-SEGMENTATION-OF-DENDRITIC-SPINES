 
 # 
import sys,os
DTYPE='float32'  
from dend_fun_0.main_0 import  app_run_param,algorithm,algorithm_param,get_data, get_data_all ,get_dict_param
from dend_fun_0.get_path import get_name,get_configs  
 

file_path_org=os.getcwd()  
file_path_data=os.path.dirname(file_path_org)    # Change this directory to the path dataset  
gdas=get_data(file_path_data)    # Dataset Path
dend_names_chld={}
  

gdas = get_data(file_path_data)               # Dataset path
dend_names_chld = {}
'''
nam='neuropil_recon_0'  
obj_list=[ls for ls in range(150) if ls not in [129]] #[7,8,9,10,11,14,16]#[2,3,4]#,[30,31,32,63]#[20,22]#
# obj_list=[0,1,2,3,4,7,8,9,10,11,14,16,20,22,30,31,32,59,63] 
obj_list1=[13,18,23,24,25,34,36,39,42,49,50,
           53,56,57,60,61,65,66,68,71,75,
           79,80,84,85,88,89,91,92,93,97,
           100,107,108,116,120,121,123,127,129,133,
           135,136,138,141,142,144,145,146,148,150]
obj_list2=[5,6,15,17,26,27,28,29,33,35,41,43,44,45,46,47,48,51,52,54,55,58,64,67,69,70,72,73,74,76,77,78,81,82,83,86,87,90,94,95,96,98,99,101,102,103,105,106,110,111,112,115,118 ,122,124,125,132,139,140,143,147,]
obj_list3=[37,38,114,119,126,128,130,131,134,137,149]
obj_list4=[117,113,62,63,40] # My additional selection
obj_list=list(set(obj_list)-set(obj_list1)) 
obj_list=list(set(obj_list)-set(obj_list2))
obj_list=list(set(obj_list)-set(obj_list3))
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 

obj_org_path= os.path.join(file_path_data,'neuropil_0', nam )

dend_names_chld[nam]= dict(  
                dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits,
                name_spine_id='p_sps',
                name_head_id='p_sphs',
                name_neck_id='p_spns',
                name_shaft_id='p_spsh',
                weights=[[3.,0.81]],
                ) 
 '''
 
 
 
 

nam='diane'  
obj_list=[0,3,4]
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 

obj_org_path= os.path.join(file_path_data,nam  )

dend_names_chld[nam]= dict(  
                dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits,
                name_spine_id='p_sps',
                name_head_id='p_sphs',
                name_neck_id='p_spns',
                name_shaft_id='p_spsh',
                weights=[[3.,0.81]],
                )  



nam='princeton' 
obj_list=list(set(range(1,401))-set(range(261,281)))
obj_list=obj_list[:15]
dend_names=[f'{nn}' for nn in obj_list ] 
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 
                 
objs_path_org= os.path.join(file_path_data,nam )
obj_org_path= os.path.join(file_path_data, nam )

dend_names_chld[nam]= dict(  
                dend_names=dend_names , 
                objs_path_org=objs_path_org,
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits,
                name_spine_id='p_sps',
                name_head_id='p_sphs',
                name_neck_id='p_spns',
                name_shaft_id='p_spsh',
                weights=[[2.,0.81]],
                ) 





nam='diane_shrink'  
obj_list=[0,]
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 
0
obj_org_path= os.path.join(file_path_data,nam  )

dend_names_chld[nam]= dict(  
                dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits,
                name_spine_id='p_sps',
                name_head_id='p_sphs',
                name_neck_id='p_spns',
                name_shaft_id='p_spsh',
                weights=[[16.,0.81]],
                )  


nam='d000'
'''
nam_gen='diane_shrink'  
nam='d003'
obj_list=[nn for nn in range(79)]
dend_names = [f'{nam}_{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 
0
obj_org_path= os.path.join(file_path_data,nam_gen,nam  )

dend_names_chld[nam]= dict(  
                dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits,
                name_spine_id='p_sps',
                name_head_id='p_sphs',
                name_neck_id='p_spns',
                name_shaft_id='p_spsh',
                weights=[[3.,0.81]],
                )  
'''



nam='diane_raw_0_wrap' 
obj_org_path = os.path.join(file_path_data,nam)
obj_org_path = os.path.join(file_path_data,'diane',nam)
dend_names=[f for f in os.listdir(obj_org_path)  if f.startswith('d')] 

dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]  

dend_names_chld[nam]= dict(  
                dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits,
                name_spine_id='p_sps',
                name_head_id='p_sphs',
                name_neck_id='p_spns',
                name_shaft_id='p_spsh',
                weights=[[6.,0.81]],
                )  

  
nam_gen='diane_raw_0_wrap'   
nam='d003'
obj_list=[nn for nn in range(94)]
dend_names = [f'{nam}_{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]

obj_org_path = os.path.join(file_path_data,'diane',nam_gen,nam)

dend_names_chld[nam]= dict(  
                dend_namess=dend_namess ,
                dend_names=dend_names , 
                obj_org_path=obj_org_path,
                dend_last=dend_last ,
                dend_first=dend_first ,
                dend_path_inits=dend_path_inits,
                name_spine_id='p_sps',
                name_head_id='p_sphs',
                name_neck_id='p_spns',
                name_shaft_id='p_spsh',
                weights=[[16.,0.81]],
                )  

'''
nam='p21-dendrites'

dend_names = ['BTLNG_d004_RECON','BTLNG_d005_RECON','BTLNG_d010_RECON','CNJVY_d001_RECON','CNJVY_d003_RECON','CNJVY_d005_RECON',]   
dend_last = ['BTLNG', 'BTLNG', 'BTLNG', 'CNJVY', 'CNJVY', 'CNJVY', ] 
dend_first = ['d004', 'd005',  'd010',  'd001', 'd003',  'd005', ] 

dend_namess = [[f'{dn}_',df,f'{df}sp'] for dn,df in zip(dend_names,dend_first)]


dend_namess = [[f'{dn}_',df,f'{df}sp'] for dn,df in zip(dend_names,dend_first)]
extraa=[f'p21-dendrites_resized_{mm}' for mm in [5000,7000,15000,30000,60000,]]
for nam in [ 'p21-dendrites','p21-dendrites_resized', ]+extraa:
    dend_names_chld[nam]=dict( 
        dend_namess=dend_namess ,
        dend_names=dend_names ,
        dend_last=dend_last ,
        dend_first=dend_first ,
        dend_path_inits=[nam  for _ in range(len(dend_names))],
        name_spine_id='sp',
        name_head_id='hsp',
        name_neck_id='nsp',
        name_shaft_id='shsp',
        objs_path_org=None,
        weights=[[3.,0.81]],
        )'''




nam_gen,action='neuropil_recon_0','resized'
obj_list=[ls for ls in range(151) if ls not in [129]] #[7,8,9,10,11,14,16]#[2,3,4]#,[30,31,32,63]#[20,22]#
# obj_list=[0,1,2,3,4,7,8,9,10,11,14,16,20,22,30,31,32,59,63] 
obj_list1=[13,18,23,24,25,34,36,39,42,49,50,
           53,56,57,60,61,65,66,68,71,75,
           79,80,84,85,88,89,91,92,93,97,
           100,107,108,116,120,121,123,127,129,133,
           135,136,138,141,142,144,145,146,148,150]
obj_list2=[5,6,15,17,26,27,28,29,33,35,41,43,44,45,46,47,48,51,52,54,55,58,64,67,69,70,72,73,74,76,77,78,81,82,83,86,87,90,94,95,96,98,99,101,102,103,105,106,110,111,112,115,118 ,122,124,125,132,139,140,143,147,]
obj_list3=[37,38,114,119,126,128,130,131,134,137,149]
obj_list4=[117,113,62,63,40] # My additional selection
obj_list=list(set(obj_list)-set(obj_list1)) 
obj_list=list(set(obj_list)-set(obj_list2))
obj_list=list(set(obj_list)-set(obj_list3))
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam_gen for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 

extraa=[f'{nam_gen}_{action}_{mm}' for mm in [5000,7000,15000]]
for nam in [nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_5000',f'{nam_gen}_resize_5000',f'{nam_gen}_smooth_resize_10000'  ]+extraa:
    obj_org_path= os.path.join(file_path_data,'neuropil_0', nam )
    dend_names_chld[nam]= dict(  
                    dend_namess=dend_namess ,
                    dend_names=dend_names , 
                    obj_org_path=obj_org_path,
                    dend_last=dend_last ,
                    dend_first=dend_first ,
                    dend_path_inits=dend_path_inits,
                    name_spine_id='p_sps',
                    name_head_id='p_sphs',
                    name_neck_id='p_spns',
                    name_shaft_id='p_spsh',
                    ) 




nam_gen,action='neuropil_recon_00','resize'
obj_list=[ls for ls in range(151) if ls not in [129]] #[7,8,9,10,11,14,16]#[2,3,4]#,[30,31,32,63]#[20,22]#
# obj_list=[0,1,2,3,4,7,8,9,10,11,14,16,20,22,30,31,32,59,63] 
obj_list1=[13,18,23,24,25,34,36,39,42,49,50,
           53,56,57,60,61,65,66,68,71,75,
           79,80,84,85,88,89,91,92,93,97,
           100,107,108,116,120,121,123,127,129,133,
           135,136,138,141,142,144,145,146,148,150]
obj_list2=[5,6,15,17,26,27,28,29,33,35,41,43,44,45,46,47,48,51,52,54,55,58,64,67,69,70,72,73,74,76,77,78,81,82,83,86,87,90,94,95,96,98,99,101,102,103,105,106,110,111,112,115,118 ,122,124,125,132,139,140,143,147,]
obj_list3=[37,38,114,119,126,128,130,131,134,137,149]
obj_list4=[117,113,62,63,40] # My additional selection
obj_list=list(set(obj_list)-set(obj_list1)) 
obj_list=list(set(obj_list)-set(obj_list2))
obj_list=list(set(obj_list)-set(obj_list3))
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_path_inits=[nam_gen for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names] 

extraa=[f'{nam_gen}_{action}_{mm}' for mm in [5000,7000,15000]]
for nam in [nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_5000',f'{nam_gen}_resize_5000',f'{nam_gen}_smooth_resize_10000'  ]+extraa:
    obj_org_path= os.path.join(file_path_data,'neuropil_0', nam )
    dend_names_chld[nam]= dict(  
                    dend_namess=dend_namess ,
                    dend_names=dend_names , 
                    obj_org_path=obj_org_path,
                    dend_last=dend_last ,
                    dend_first=dend_first ,
                    dend_path_inits=dend_path_inits,
                    name_spine_id='p_sps',
                    name_head_id='p_sphs',
                    name_neck_id='p_spns',
                    name_shaft_id='p_spsh',
                    ) 



nam_gen,action='neuropil_recon_000','resize'
obj_list=[ls for ls in range(5) if ls not in [129]] #[7,8,9,10,11,14,16]#[2,3,4]#,[30,31,32,63]#[20,22]#
obj_list=[ls for ls in range(151) if ls not in [129]] #[7,8,9,10,11,14,16]#[2,3,4]#,[30,31,32,63]#[20,22]#
# 0,0,obj_list=[0,1,2,3,4,7,8,9,10,11,14,16,20,22,30,31,32,59,63] 
obj_list1=[13,18,23,24,25,34,36,39,42,49,50,
           53,56,57,60,61,65,66,68,71,75,
           79,80,84,85,88,89,91,92,93,97,
           100,107,108,116,120,121,123,127,129,133,
           135,136,138,141,142,144,145,146,148,150]
obj_list2=[5,6,15,17,26,27,28,29,33,35,41,43,44,45,46,47,48,51,52,54,55,58,64,67,69,70,72,73,74,76,77,78,81,82,83,86,87,90,94,95,96,98,99,101,102,103,105,106,110,111,112,115,118 ,122,124,125,132,139,140,143,147,]
obj_list3=[37,38,114,119,126,128,130,131,134,137,149]
obj_list4=[117,113,62,63,40] # My additional selection
obj_list=list(set(obj_list)-set(obj_list1)) 
obj_list=list(set(obj_list)-set(obj_list2))
obj_list=list(set(obj_list)-set(obj_list3))
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list]
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]  

extraa=[f'{nam_gen}_{action}_{mm}' for mm in [5000,7000,15000]]
for nam in [nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_5000',f'{nam_gen}_resize_5000',f'{nam_gen}_smooth_resize_10000',f'{nam_gen}_smooth_resize_15000'  ]+extraa:
    obj_org_path= os.path.join(file_path_data,'neuropil_0', nam )
    dend_path_inits=[nam for _ in range(len(dend_names))]
    dend_names_chld[nam]= dict(  
                    dend_namess=dend_namess ,
                    dend_names=dend_names , 
                    obj_org_path=obj_org_path,
                    dend_last=dend_last ,
                    dend_first=dend_first ,
                    dend_path_inits=dend_path_inits,
                    name_spine_id='p_sps',
                    name_head_id='p_sphs',
                    name_neck_id='p_spns',
                    name_shaft_id='p_spsh',
                    ) 



nam_gen='meshes'  
obj_list=[1,2]
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_namess = [[f'{de}_',de,f'{de}'] for de in dend_names]
dend_last = [f'{de}_'  for de in dend_names]
dend_first= [ f'{de}' for de in dend_names]  

extraa=[f'{nam_gen}_{action}_{mm}' for mm in [1000,5000,7000,15000]]
for nam in [nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_5000',f'{nam_gen}_resize_5000',f'{nam_gen}_smooth_resize_10000'  ]+extraa:
    obj_org_path= os.path.join(file_path_data,nam, nam )
    dend_path_inits=[nam for _ in range(len(dend_names))]
    dend_names_chld[nam]= dict(  
                    dend_namess=dend_namess ,
                    dend_names=dend_names , 
                    obj_org_path=obj_org_path,
                    dend_last=dend_last ,
                    dend_first=dend_first ,
                    dend_path_inits=dend_path_inits,
                    name_spine_id='p_sps',
                    name_head_id='p_sphs',
                    name_neck_id='p_spns',
                    name_shaft_id='p_spsh',
                    ) 


action='resize'
nam_gen='Kasthuri15'
dend_names=[
         #  ### ##    # 'Kasthuri15__0016_Objects',
            'Kasthuri15__0016_Objects',
            'Kasthuri15__0017_Objects', 
            'Kasthuri15__0018_Objects', 
           #  'Kasthuri15__0017_Objects_resized',
           #  'Kasthuri15__0018_Objects_resized',
            # 'Kasthuri15__0016_Objects_resized_2',
            # 'Kasthuri15__0017_Objects_resized_2',
            # 'Kasthuri15__0018_Objects_resized_2',
            ]
extraa=[f'{nam_gen}_{action}_{mm}' for mm in [450000,5000,7000,15000]]
extraaa=[f'{nam_gen}_{action}_{mm}_smooth' for mm in [450000,5000,7000,15000]]
for nam in [nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_{action}_smooth',f'{nam_gen}_{action}_450000',f'{nam_gen}_{action}_450000_smooth',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_5000',f'{nam_gen}_resize_5000',f'{nam_gen}_smooth_resize_10000'  ]+extraa:
    obj_org_path= os.path.join(file_path_data,nam_gen, nam ) 

    dend_names_chld[nam]= dict(  
                    dend_names=dend_names , 
                    obj_org_path=obj_org_path,
                    weights=[[16.,0.81]],
                    ) 




nam_gen,action='p21-dendrites','resize'


dend_names = ['BTLNG_d004_RECON','BTLNG_d005_RECON','BTLNG_d010_RECON','CNJVY_d001_RECON','CNJVY_d003_RECON','CNJVY_d005_RECON',]   
dend_last = ['BTLNG', 'BTLNG', 'BTLNG', 'CNJVY', 'CNJVY', 'CNJVY', ] 
dend_first = ['d004', 'd005',  'd010',  'd001', 'd003',  'd005', ]  
dend_namess = [[f'{dn}_',df,f'{df}sp'] for dn,df in zip(dend_names,dend_first)]
extraa=[f'{nam_gen}_{action}_{mm}' for mm in [1000,5000,7000,15000]]
for nam in [ nam_gen,f'{nam_gen}_{action}',f'{nam_gen}_smooth',f'{nam_gen}_smooth_resize_15000',f'{nam_gen}_smooth_resize_5000', ]+extraa:
    obj_org_path= os.path.join(file_path_data,nam_gen, nam )
    dend_names_chld[nam]=dict( 
        dend_namess=dend_namess ,
        dend_names=dend_names ,
        dend_last=dend_last ,
        dend_first=dend_first ,
        dend_path_inits=[nam  for _ in range(len(dend_names))],
        name_spine_id='sp',
        name_head_id='hsp',
        name_neck_id='nsp',
        name_shaft_id='shsp',
        objs_path_org=None,
        weights=[[3.,0.81]],
        obj_org_path=obj_org_path,
        )

# This dataset is located in

# This dataset is located in file_path_data/meshes/d001/data_org/
# Mesh file: file_path_data/meshes/d001/data_org/d001.obj

gdas=get_data_all(names_dic=dend_names_chld,dend_data=gdas.dend_data,file_path_data=file_path_data)

gnam = get_name()      
# Choose the name of the data to test here we choose meshes
nam='princeton' 
 
nam='Kasthuri15'    
nam='diane'

nam='d003'
nam = 'meshes'      
nam='d000'  
nam='d003'  
nam='diane_raw_0_wrap' 
nam='diane_shrink' 

nam_gen,action,nnbb='p21-dendrites','resized', 7000 
nam_gen,action,nnbb='neuropil_recon_0','resized', 30000 
nam_gen,action,nnbb='neuropil_recon_0','resized', 7000 
nam_gen,action,nnbb='neuropil_recon_0','resized', 15000 
nam=f'{nam_gen}_{action}_{nnbb}'
nam='p21-dendrites_resize_5000'
nam='neuropil_recon_0_smooth_resize_5000'
nam='neuropil_recon_0_smooth_resize_10000'

nam='neuropil_recon_0_smooth' 
nam='neuropil_recon_00_smooth'  
nam='neuropil_recon_00'  
nam='neuropil_recon_0' 

nam = 'p21-dendrites' 
nam='Kasthuri15_resize_smooth'
nam='Kasthuri15_resize'
nam='Kasthuri15_resize_450000_smooth'
nam = 'p21-dendrites_smooth'
nam='neuropil_recon_000_smooth'
nam='neuropil_recon_000_resize_5000'
nam='meshes'  
nam='Kasthuri15_resize_450000'
nam='neuropil_recon_000'

nam = 'p21-dendrites_smooth_resize_15000' 

nam_gen,nam_loc,action,resize='p21-dendrites','p21-dendrites','smooth','resize_15000'
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000','smooth','resize_15000'
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000',None,'resize_15000'
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000',None,None
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000',None,'resize_5000'
nam_gen,nam_loc,action,resize='neuropil_0','neuropil_recon_000','smooth',None
# nam = f'{nam_loc}_{resize}' if action is None else  f'{nam_loc}_{action}_{resize}'
nam = "_".join(x for x in [nam_loc, action, resize] if x is not None)
drop_dic_name='_'.join(x for x in [nam_loc,action] if x is not None)
# f'{nam_loc}_{action}' =
 

model_type='vol_3UNet3D3_5000_hpcc' 

nnbb=5000
model_type='vol_VGG16_FCN3D_5000_hpcc'
model_type='dnn_GINN_1'
model_type = 'pinn'
model_type = 'pinn_NN'
model_type=f'dnn_GINN_0'
model_type='dnn_GINN_1_AUG'
model_type='dnn_1GINN_SM_SMOOTH'
model_type='dnn_GINN_DICE_AUG'
model_type='dnn_GINN_AUG'
model_type='vol_UNet3D_DICE_AUG'
model_type='dnn_GINN_SMOOTH_AUG'
model_type='dnn_1GINN_SM'
model_type='dnn_GINN_SMOOTH'
model_type='dnn_GINN_SMOOTH'
model_type='dnn_GINN_DICE_SMOOTH_AUG'
model_type='dnn_GINN_DICE_SMOOTH'
model_type='vol_VoxNetSeg_DICE'
model_type='vol_VGG16_FCN3D_DICE'
model_type='vol_UNet3D_DICE'
model_type='vol_3UNet3D3_5000_hpcc'

model_type='dnn_1GINN_SM'
model_type='vol_FastFCN3D_SIMP_AUG'
model_type='dnn_1GINN_SM_ORG_AUG'

model_type='dnn_GINN_ORG1_AUG'
model_type='dnn_1GINN_SM_ORG_AUG'
model_type='pinn_GINN_ORG'
model_type='dnn_GINN_ORG1_AUG'
model_type='dnn_GINN_ORG0_AUG'
model_type_data='pinn_GINN_ORG'
model_type='vol_UNet3D_SM07000_AUG'
model_type='pinn_GINN_ORG'
model_type='pinn_GINN_ORG'
model_type='dnn_GINN_SM00000_AUG'
model_type=f'cnn_VoxNetSeg_{nnbb}_hpcc_crop'
model_type=f'cnn_VGG16_FCN3D_{nnbb}_hpcc_crop'
model_type='gcn_UNet_SM10000_LOC' 
model_type=f'cnn_3UNet3D3_{nnbb}_hpcc_crop'
model_type='dnn_GINN_SM00000_LOC_AUG'
model_type='cml_cML' 

dnn_mode = 'DNN-1'         # choose neural network type
dnn_mode = 'DNN-2'         # choose neural network type 
dnn_mode = 'DNN-0'         # choose neural network type 
dnn_mode = 'DNN-3'         # choose neural network type

   
# size_threshold=300 if nam.startswith('p21') else 10 if nam.start
weight=0.001
weight=0.5
weight=0.05 
weight=0.1
weight=13  
weight=16# smoothing step size
weight=0.81  
weight=1   
weight=10   
weight=3    

if nam.startswith("p21"):
    size_threshold = 300
elif nam.startswith("cnn"):
    size_threshold = 10
else:
    size_threshold = 100

weight_pre=weight
path_dir_nam=f'dice'
path_dir=f'{path_dir_nam}_we_{weight}'
data_dir=f'{path_dir_nam}_we_3'
data_dir=path_dir
path_heads_show=[model_type,]

dict_param=get_dict_param(nam=nam,
                    n_step = 0,
                    weight=weight,
                    size_threshold=size_threshold,
                    path_heads_show=path_heads_show,
                    )

# if model_type.startswith(('vol')):
#     dict_param['param_dic']['data']['get_dend_name']=dict(dict_dend_path='old',
#                                     drop_dic_name='resize_5000') 
# nams=nam.split('_')
# drop_dic_name= nam if len(nams)<3 else '_'.join(nams[-2:])

# nam_gen
# if drop_dic_name.startswith(('resizne')):
if resize is not None:
    dict_param['param_dic']['data']['get_dend_name']=dict(dict_dend_path='old',
                                                        old_path=None,
                                                        drop_dic_name=drop_dic_name,
                                                        nam_gen=nam_gen,
                                                        )   
dict_param['model_type_data']=model_type

param = algorithm_param(**dict_param)



dend_data = gdas.part(nam)  
mapp = app_run_param(param)
param=mapp.emerge_param()
param['dendrite_pred']['param']['size_threshold']=size_threshold
param['Resizing']['param']['target_number_of_triangles_faction']=150000
for val in ['get_pinn_features','get_skeleton','get_wrap','get_smooth']:
    param['dendrite_pred']['param']['param_dic']['tf_restart'][val]=False   #True   # 
param['dendrite_pred']['param']['param_dic']['tf_restart']['get_shaft_pred']=False   #True   #  
param['Smooth']['param']['n_step']=20


param['Smooth']['param']['dt']=1e-7
param['Smooth']['param']['method']='taubin' #'willmore'
param['Skeleton']['param']['path']="entry"     #'old'   #  
param['Skeleton']['param']['dict_mesh_to_skeleton_finder']['interval_target_number_of_triangles']=[200000]
param['annotations']['tf'] =False   #True   #  True/False: choose whether to generate annotations for training, accuracy, and recall
param['Spine-Shaft Segm']['tf'] =True    #False   #  True/False: choose whether to predict shafts/spines
param['skl_shaft_pred']['tf'] =False   #True  # 
param['Morphologic Param']['tf'] =False   #True   #    False   # True/False: choose whether to perform head/neck segmentation
param['iou']['tf'] =True       #False   # False   #  True/False: choose whether to compute IoU
param['roc']['tf'] =True       # False   # True/False: choose whether to compute IoU
param['graph_center']['tf'] = True   #False  # True   #False   #   True/False: compute central axis
param['cylinder_heatmap']['tf'] =True   # False   # False   # False   # True/False: generate cylindrical heatmap
param['dash_pages']['tf'] =True          #False   #  True/False: generate Dash pages
param['clean_path_dir']['tf'] = False          # True/False: delete computed data
param['intensity_rhs']['tf']=False #  True  #  
param['Resizing']['tf']= False # True  #  
param['Smooth']['tf'] =False   # True   # False   # True/False: choose whether to smooth the data
param['Skeleton']['tf']=True  #False #   
param['rhs']['tf']=  True  #  False #
param['model_shap']['tf']= True  #  False #

'''
param['Smooth']['tf'] = False   #True   #False   #  True/False: choose whether to smooth the data
param['annotations']['tf'] =False   #True   #  True/False: choose whether to generate annotations for training, accuracy, and recall
param['Spine-Shaft Segm']['tf'] =False   # True    # True/False: choose whether to predict shafts/spines
param['skl_shaft_pred']['tf'] =False   #True  # 
param['Morphologic Param']['tf'] = False   #True   #   False   # True/False: choose whether to perform head/neck segmentation
param['iou']['tf'] =False   #True       # False   #  True/False: choose whether to compute IoU
param['roc']['tf'] = False   # True       #True/False: choose whether to compute IoU
param['graph_center']['tf'] =False  # True   # True   #False   #   True/False: compute central axis
param['cylinder_heatmap']['tf'] =False   #True   #  False   # False   # True/False: generate cylindrical heatmap
param['dash_pages']['tf'] =True          #False   #  True/False: generate Dash pages
param['clean_path_dir']['tf'] = False          # True/False: delete computed data
param['Resizing']['tf']=False # True  #   
param['intensity_rhs']['tf']=False #  True  # 
param['Skeleton']['tf']=False # True  #  
param['rhs']['tf']=  False # True  # 
param['model_shap']['tf']= True  #  False #
'''
alg = algorithm(param)
alg.test(   
    dend_data = dend_data,  
    true_name = 'true_0', 
    dnn_mode = dnn_mode,
    model_type = model_type, 
    path_dir=path_dir,
    data_dir=data_dir,
    **dict_param
    # path_display = path_display, 
    # model_type_data =model_type_data, 
    # size_threshold=size_threshold,
    # path_display_dic=path_display_dic,
    # path_heads_show=path_heads_show,
    # path_shaft_dir=path_shaft_dir,
)
