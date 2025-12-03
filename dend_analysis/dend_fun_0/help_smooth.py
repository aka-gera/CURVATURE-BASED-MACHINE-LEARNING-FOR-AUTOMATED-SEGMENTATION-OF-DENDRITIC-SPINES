

import time
import numpy as np
import os
import dend_fun_0.curvature as cuHP 
from dend_fun_0.obj_get import  Obj_to_coord
from dend_fun_0.get_path import  get_files
import trimesh



def get_load_smooth(dend_path_original_m,
                    dend_name,
                    file_path, 
                    dend_path_original_new_smooth=None,
                    txt_vertices_0='vertices_0.txt',
                    txt_vertices_1='vertices_1.txt' ,
                    txt_faces=None,
                    get_data=True,
                    n_error = 1,
                    n_step = None,
                    dt=1e-6, 
                    disp_time=5000,): 
    print('-----------=================get_load_smooth===============----------',) 

    nbm=1
    dend_path_original=os.path.join(dend_path_original_m,f'{dend_name}.obj')  
     
    time_start = time.time()   
    mesh=trimesh.load_mesh(os.path.join(dend_path_original_new_smooth,f'{dend_name}.obj'))   
    np.savetxt(os.path.join(file_path,txt_vertices_0),mesh.vertices, fmt='%f') 
    np.savetxt(os.path.join(file_path,txt_faces),mesh.faces, fmt='%d') 

    vertices,_=Obj_to_coord(file_path_original=dend_path_original,)  
    np.savetxt(os.path.join(file_path,txt_vertices_1), vertices, fmt='%f') 


    mytime0 = time.time() - time_start 
    hours, rem = divmod(mytime0, 3600)
    minutes, seconds = divmod(rem, 60) 
    print(f'get_smooth_one completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s') 





class get_smooth(get_files):
    def __init__(self, file_path_org, 
                    dend_data, 
                    obj_org_path=None,
                    dend_path_inits=None ,
                    dend_names=None ,
                    dend_namess=None,
                    new_data=True,
                    get_data=True,
                    true_keys=None,
                    obj_org_path_dict=None,
                    model_sufix_dic=None, 
                    n_error=1,
                    n_step=None,
                    dt=1e-6,
                    disp_time=100):

        self.new_data = new_data
        self.get_data = get_data
        self.n_error = n_error
        self.n_step = n_step
        self.dt = dt  
        self.disp_time = disp_time
  
        super().__init__(file_path_org=file_path_org, 
                        dend_data=dend_data,
                        obj_org_path=obj_org_path,
                        model_sufix='save',
                        dend_names=dend_names,
                        dend_namess=dend_namess, 
                        dend_path_inits=dend_path_inits,
                        pinn_dir_data_all=['save'],
                        model_sufix_all=['save'],
                        path_heads=['save'],
                        model_type='save',
                        true_keys=true_keys,
                        obj_org_path_dict=obj_org_path_dict ,
                                model_sufix_dic=model_sufix_dic,
                            )

 

    def get_smooth_all(self, 
                        get_data=None,
                        n_error = None,
                        n_step = None,
                        dt=None,
                        disp_time=None, ): 
        get_data=get_data or self.get_data
        n_error =n_error or self.n_error
        n_step=n_step or self.n_step
        dt=dt or self.dt
        disp_time = disp_time or self.disp_time 
        for index,spine_group_name in enumerate(self.dend_names): 
            self.get_dend_name(index=index)  
            dend_path_original_m= self.dend_path_original_new if self.new_data else self.dend_path_original_m 
            file_path=self.file_path 
            dend_path_original_new_smooth=self.dend_path_original_new_smooth 
            get_smooth_one(dend_path_original_m=dend_path_original_m,
                        dend_name=spine_group_name,
                        dend_path_original_new_smooth=dend_path_original_new_smooth,
                        file_path=file_path,
                        txt_vertices_0=self.txt_vertices_0,
                        txt_vertices_1=self.txt_vertices_1,
                        txt_faces=self.txt_faces,
                        get_data=get_data,
                        n_error = n_error,
                        n_step = n_step,
                        dt=dt,
                        disp_time=disp_time, 
                        )



    def get_load_smooth_all(self, 
                        get_data=None,
                        n_error = None,
                        n_step = None,
                        dt=None,
                        disp_time=None, ):  
        get_data=get_data or self.get_data
        n_error =n_error or self.n_error
        n_step=n_step or self.n_step
        dt=dt or self.dt
        disp_time = disp_time or self.disp_time  
        for index,spine_group_name in enumerate(self.dend_names): 
            self.get_dend_name(index=index)  
            dend_path_original_m= self.dend_path_original_new if self.new_data else self.dend_path_original_m 
            file_path=self.file_path 
            dend_path_original_new_smooth=self.dend_path_original_new_smooth 
            print(f"Dendrite Name: {spine_group_name}------------------------{self.txt_faces}")
            get_load_smooth(dend_path_original_m=dend_path_original_m,
                        dend_name=spine_group_name,
                        dend_path_original_new_smooth=dend_path_original_new_smooth,
                        file_path=file_path,
                        txt_vertices_0=self.txt_vertices_0,
                        txt_vertices_1=self.txt_vertices_1,
                        txt_faces=self.txt_faces,
                        get_data=get_data,
                        n_error = n_error,
                        n_step = n_step,
                        dt=dt,
                        disp_time=disp_time, 
                        )



def get_smooth_one(dend_path_original_m,
                    dend_name,
                    file_path, 
                    dend_path_original_new_smooth=None,
                    txt_vertices_0='vertices_0.txt',
                    txt_vertices_1='vertices_1.txt' ,
                    txt_faces=None,
                    get_data=True,
                    n_error = 1,
                    n_step = None,
                    dt=1e-6, 
                    disp_time=5000,):
    print('-------------------;;',file_path,dend_name,dend_path_original_m) 
    dend_path_original=os.path.join(dend_path_original_m,f'{dend_name}.obj') 
    if get_data:
        vertices,faces=Obj_to_coord(file_path_original=dend_path_original,)  
        np.savetxt(os.path.join(file_path,txt_vertices_0),vertices, fmt='%f') 
        np.savetxt(os.path.join(file_path,txt_faces), faces, fmt='%d')   
    else:
        vertices = np.loadtxt(os.path.join(file_path,txt_vertices_0), dtype=float) 
    faces = np.loadtxt(os.path.join(file_path,txt_faces), dtype=int) 
    len_v0,len_f0=vertices.shape[0],faces.shape[0] 
    smooth_path=os.path.join(dend_path_original_new_smooth,f'{dend_name}.obj') 
    print(f"Number of vertices: before {len(vertices)} | after {len_v0}")
    print(f"Number of faces:    before {len(faces)}    | after {len_f0}") 
    time_start = time.time()   
    if n_step>0:
        print('Job started')
        simu=cuHP.simulation(vertices=vertices,
                        faces=faces,
                        dt=dt,
                        n_step=n_step,
                        n_error=n_error,
                        disp_time=disp_time,
                        save_file=file_path,
                        smooth_path=smooth_path, 
                        txt_vertices=txt_vertices_0,
                        ) 
        print('Job done!!')
    mesh=trimesh.Trimesh(vertices=np.loadtxt(os.path.join(file_path,txt_vertices_0), dtype=float),faces=faces) 
    mesh.export(os.path.join(dend_path_original_new_smooth,f'{dend_name}.obj') )      
    np.savetxt(os.path.join(file_path,txt_vertices_0),mesh.vertices, fmt='%f') 
    np.savetxt(os.path.join(file_path,txt_faces),mesh.faces, fmt='%d')  

    mesh=trimesh.load_mesh(dend_path_original) 
    np.savetxt(os.path.join(file_path,txt_vertices_1),mesh.vertices, fmt='%f')  
    mytime0 = time.time() - time_start 
    hours, rem = divmod(mytime0, 3600)
    minutes, seconds = divmod(rem, 60) 
    print(f'get_smooth_one completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s') 


