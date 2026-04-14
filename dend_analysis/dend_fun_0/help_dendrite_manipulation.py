 

import sys
import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import time   
DTYPE='float32' 
import pickle 
import pandas as pd
import trimesh

import copy


import random
np.random.seed(42) 
random.seed(42) 
from tqdm import tqdm 

from dend_fun_0.help_save_iou import iou_train 
from dend_fun_0.help_pinn_data_fun import pinn_data 
from dend_fun_2.help_pinn_data_fun import pinn_data 
import dend_fun_0.help_funn as hff   
from dend_fun_0.help_funn import get_intensity ,mappings_vertices,pca_projector


from dend_fun_0.get_path import assign_if_none,get_name,get_param,get_files


  

class dendrite_manipulate(get_files,get_name):
    def __init__(self, file_path_org,
                        data_studied, 
                        model_sufix,
                        dend_data, 
                        dend_names=None,
                        dend_namess=None,   
                        dend_path_inits=None,
                        name_spine_id=None,
                        name_head_id=None,
                        name_neck_id=None,
                        name_shaft_id=None,
                        path_train=None,
                        txt_true_file=None,
                        txt_save=False,
                        txt_save_pred=False,
                        size_threshold=100,
                        gauss_threshold=10, 
                        name_path_fin='save',
                        cts=6,
                        stoppage=4,
                        zoom_threshold=1000,
                        radius_threshold=0.05,
                        name_path_fin_save_index=20,
                        spine_filter=True,
                        numNeighbours=5,
                        zoom_threshold_min=1,
                        zoom_threshold_max=4, 
                        line_num_points_shaft=200,
                        line_num_points_inter_shaft=300, 
                        spline_smooth_shaft=1, 
                        DTYPE='float32', 
                        disp_infos=False,
                        pre_portion=None,
                        pinn_dir_data=None, 
                        pinn_dir_data_all=None,
                        model_sufix_all=None,
                        path_heads =None,
                        true_keys=None,
                        list_features=None,
                        base_features_list=None,
                        metrics={},
                        model_type=None,
                        data_mode=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        obj_org_path_dict=None,
                        path_display=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        model_sufix_dic=None,
                        path_display_dic=None,
                            kmean_n_run=None,
                            kmean_max_iter=None,
                            param_dic=None,

        ): 
        get_name.__init__(self)  

        get_files.__init__(self,file_path_org=file_path_org,
                                dend_data=dend_data,
                                dend_names=dend_names,
                                dend_namess=dend_namess, 
                                data_studied=data_studied,
                                name_spine_id=name_spine_id,
                                name_head_id=name_head_id,
                                name_neck_id=name_neck_id,
                                name_shaft_id=name_shaft_id,
                                txt_true_file=txt_true_file,
                                txt_save=txt_save,
                                txt_save_pred=txt_save_pred,
                                size_threshold=size_threshold,
                                gauss_threshold=gauss_threshold, 
                                name_path_fin=name_path_fin,
                                cts=cts,
                                stoppage=stoppage,
                                zoom_threshold=zoom_threshold,
                                radius_threshold=radius_threshold,
                                name_path_fin_save_index=name_path_fin_save_index,
                                spine_filter=spine_filter,
                                numNeighbours=numNeighbours,
                                zoom_threshold_min=zoom_threshold_min,
                                zoom_threshold_max=zoom_threshold_max, 
                                line_num_points_shaft=line_num_points_shaft,
                                line_num_points_inter_shaft=line_num_points_inter_shaft, 
                                spline_smooth_shaft=spline_smooth_shaft, 
                                DTYPE=DTYPE, 
                                model_sufix=model_sufix,
                                dend_path_inits=dend_path_inits,
                                disp_infos=disp_infos,
                                path_train=path_train,
                                pre_portion=pre_portion,
                                pinn_dir_data=pinn_dir_data, 
                                pinn_dir_data_all=pinn_dir_data_all,
                                model_sufix_all=model_sufix_all,
                                path_heads=path_heads,
                                true_keys=true_keys,
                                list_features=list_features,
                                base_features_list=base_features_list,
                                metrics=metrics,
                                model_type=model_type,
                                data_mode=data_mode,
                                thre_target_number_of_triangles=thre_target_number_of_triangles,
                                voxel_resolution=voxel_resolution,
                                obj_org_path_dict=obj_org_path_dict,
                                model_sufix_dic=model_sufix_dic,
                                path_display_dic=path_display_dic,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )







 
 
    def get_wrap(self,   
                    entry_name=None,
                    exit_name='wrap',
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None,
                    alpha_fraction=0.005,
                    offset_fraction=0.001,
                    wrap_part='mesh', 
                    path='entry',
                    dict_wrap=None,
                    param_dic=None,
                    drop_dic_name=None,
                    old_path=None,
                    param_dend_name=None,
                    ):   
        param_dic =param_dic if param_dic is not None else self.param_dic
        from dend_fun_0.get_wrap import get_wrap ,get_wrap_o3d
        print('get_wrap  1 ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        if dict_wrap is None: 
            dict_wrap=dict(  
                        number_of_points=5000,
                        radius=0.1, 
                        max_nn=30,
                        )
        number_of_points=dict_wrap['number_of_points']
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',**param_dend_name )
            # self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path,nam_gen=nam_gen, )  
            dend_name=self.dend_name 
            print(dend_name)    

            dend_path_exit_data =self.dend_path_org_exit
            dend_path_entry_data = {
                                    "entry": self.dend_path_org_entry,
                                    "old":   self.dend_path_org_old,
                                    "exit":  self.dend_path_org_exit
                                }.get(path, self.dend_path_org_entry)

            if 'shaft' in wrap_part.lower(): 
                dend_path_exit=os.path.join(dend_path_exit_data,f'{self.dend_first_name}_{wrap_part}_wrap.obj') 
                dend_path_entry=os.path.join(dend_path_entry_data,f'{self.dend_first_name}_{wrap_part}.obj') 
            else:
                dend_path_entry=os.path.join(dend_path_entry,f'{dend_name}.obj') 
                dend_path_exit=os.path.join(dend_path_exit,f'{dend_name}.obj') 
            if os.path.exists(dend_path_entry): 
                get_feat = param_dic['tf_restart']['get_wrap'] 
                print('[[[[]]]]',dend_path_entry,get_feat)
                if os.path.exists(dend_path_exit) and not get_feat: 
                    if param_dic['get_info']['get_skeleton']:
                        print('[[[[[[get_wrap--  Exist Already or Restarting --]]]]]]',dend_path_exit)
                    continue 
                meshh = trimesh.load_mesh(dend_path_entry,process=False)
                dict_wrap['number_of_points'] =int(max( number_of_points, min(5000,meshh.vertices.shape[0]//25))) # int(max(number_of_points, meshh.vertices.shape[0]/5)) 
                simplified_vertices,simplified_faces=get_wrap_o3d(vertices=meshh.vertices,
                                                            faces=meshh.faces,
                                                            **dict_wrap,
                                                            )
                dict_wrap['number_of_points'] = int(number_of_points)
                mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces,process=False)  
                mesh.export(dend_path_exit)

            else:
                print(f'Unavailable path: {dend_path_entry} ')





    def get_skeleton(self,   
                    entry_name='wrap',
                    exit_name=None,
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None,
                    alpha_fraction=0.005,
                    offset_fraction=0.001,
                    weights=None, 
                    weight=None, 
                    shaft_thre=0.9, 
                    smooth_tf=None,
                    neck_lim=None,
                    dict_wrap=None,
                    size_threshold=None,
                    wrap_part=None,
                    dict_mesh_to_skeleton_finder=None,
                    tf_skl_shaft_distance=None,
                    path='entry',
                    tf_restart=False,
                    param_dic=None,
                    drop_dic_name=None,
                    old_path=None,
                    ):   
        param_dic =param_dic if param_dic is not None else self.param_dic
        from dend_fun_2.help_pinn_data_fun import def_mesh_to_skeleton_finder
        from dend_fun_0.help_funn import order_points_along_pca,project_vertices_onto_skeleton
        from scipy.spatial import cKDTree
        print(f' get_skeleton  ---- started') 
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,wrap_part=wrap_part,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path, )  
            dend_name=self.dend_name 
            print(dend_name)   
            spine_path=self.path_file[path_train['dest_spine_path']]
            shaft_path=self.path_file[path_train['dest_shaft_path']] 
            dend_path_exit_data =self.dend_path_org_exit


            dend_path_entry_data = self.dend_data_all.get(path, {}).get(
                'path',
                self.dend_path_org_entry
            ) 
            if wrap_part is None:
                txt_distance=self.dend_data_all[path]['distance'] 
                txt_vertices=self.dend_data_all[path]['vertices']
                nami=f'{dend_name}.obj'  
                namm=f'{self.dend_first_name}_skeleton.obj'  
            else:
                txt_distance=self.dend_data_all[path][f'distance_{wrap_part}'] 
                txt_vertices=self.dend_data_all[path][f'vertices_{wrap_part}'] 
                nami=  f'{self.dend_first_name}_{wrap_part}.obj'
                namm= f'{self.dend_first_name}_{wrap_part}_skeleton.obj' 


            dend_path_exit=os.path.join(dend_path_exit_data,namm)
            get_feat = param_dic['tf_restart']['get_skeleton'] 
            if os.path.exists(dend_path_exit) and not get_feat: 
                if param_dic['get_info']['get_skeleton']:
                    print('[[[[[[get_skeleton--  Exist Already or Restarting --]]]]]]',dend_path_exit)
                pass
            else: 
                dend_path_entry=os.path.join(dend_path_entry_data,nami )  
                if not os.path.exists(dend_path_entry):
                    print(f'Unavailable path: {dend_path_entry} ')
                    continue
                mesh=trimesh.load_mesh( dend_path_entry,process=False ) 
                print('[[[[[[]]]]]]',dend_path_entry,mesh.vertices.shape)
                nskl=def_mesh_to_skeleton_finder(vertices=mesh.vertices,
                                            faces=mesh.faces,
                                            ** dict_mesh_to_skeleton_finder,
                                                )  
    
                indices = cKDTree(nskl.skeleton_points).query(mesh.vertices)[1].flatten() 
                mesh_sk=trimesh.Trimesh(vertices=nskl.skeleton_points[indices],faces=mesh.faces, process=False ) 
                mesh_sk.export( dend_path_exit) 
                dtree = cKDTree(nskl.skeleton_points).query(mesh.vertices)
                dend_path=self.file_path_model_data 
                file_path_feat_exit=self.file_path_feat_exit#os.path.join(dend_path_pp, 'data', f'{dend_name}','data','feat')
                fpath_true=self.dend_path_true_final_exit#os.path.join(dend_path_pp ,'data', f'{dend_name}','true_0')
                os.makedirs(fpath_true,exist_ok=True)
                os.makedirs(file_path_feat_exit,exist_ok=True) 
                np.savetxt(os.path.join(file_path_feat_exit,txt_distance),dtree[0], fmt='%f')
                np.savetxt(os.path.join(file_path_feat_exit,txt_vertices),nskl.skeleton_points, fmt='%f') 

 
    def get_rhs(self,   
                    entry_name=None,
                    exit_name=None,  
                    dend_namess=None,
                    path_train=None, 
                        dend_names=None,
                        file_path_org=None,
                        file_path=None,
                        file_path_feat=None,
                        data_studied=None,
                        file_path_model_data=None,
                        save_dend_data=False,
                        numNeighbours=None, 
                        dend_path_inits = None, 
                        dtype = float,  
                        radius_threshold=None,
                        disp_infos=None,
                        restart=False,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        obj_org_path=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        annotation_resized_train_tf=False,
                        min_target_number_of_triangles_faction=600000,
                        target_number_of_triangles_faction=100000000,
                        size_threshold=5,
                        wrap_part='shaft_wrap',
                        path='entry',
                        tf_restart=False,
                        weight=None,
                        weights=None,
                        shaft_thre=None,
                        smooth_tf=None,
                        neck_lim=None,
                        dict_wrap=None, 
                        tf_skl_shaft_distance=None,
                        dict_mesh_to_skeleton_finder=None, 
                        drop_dic_name=None,
                    old_path=None,
                    nam_gen=None,
                    param_dend_name=None,
            ):   
        from scipy.spatial import cKDTree
          
        print('get_rhs   ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 



        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh 
        disp_infos = disp_infos or self.disp_infos
        radius_threshold = radius_threshold or self.radius_threshold 
        dend_path_inits = dend_path_inits or self.dend_path_inits
        dend_names = dend_names or self.dend_names
        file_path_org = file_path_org or self.file_path_org
        data_studied = data_studied or self.data_studied 
        numNeighbours = numNeighbours or self.numNeighbours
        obj_org_path = obj_org_path or self.obj_org_path 
 


        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',**param_dend_name )
            # self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path,nam_gen=nam_gen, )  
            dend_name=self.dend_name 
            print(dend_name)    
            for key in self.dend_path_original_mm['dir']:  
                pid=pinn_data(file_path=self.file_path,
                                file_path_feat=self.file_path_feat, 
                                shaft_path=     self.path_file[key] ,
                                spine_path=     self.path_file[key] , 
                                shaft_path_pre= self.path_file[key] ,
                                spine_path_pre= self.path_file[key] ,  
                                dend_path_original_m=self.dend_path_original_mm['dir'][key],
                                dend_first_name=self.dend_namess[index][1], 
                                name_spine_id=self.name_spine_id,
                                thre_target_number_of_triangles=thre_target_number_of_triangles,
                                voxel_resolution=voxel_resolution,
                                dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                        )  
 
                dend_path_entry_data = self.dend_data_all.get(path, {}).get(
                    'path',
                    self.dend_path_org_entry
                )   

                txt_distance=self.dend_data_all[path]['distance'] 
                txt_vertices=self.dend_data_all[path]['vertices']
                txt_skl_index=self.dend_data_all[path]['skl_index']
                nami=f'{dend_name}.obj'  
                namm=f'{self.dend_first_name}_skeleton.obj' 

                dend_path_exit_data=self.dend_path_org_exit 
                os.makedirs(dend_path_exit_data,exist_ok=True) 
                mesh_org = trimesh.load_mesh(os.path.join(dend_path_entry_data,f'{dend_name }.obj'),process=False  ) 
                vertices_00=mesh_org.vertices
                faces=mesh_org.faces
                target_number_of_triangles_faction=min(vertices_00.shape[0] , 2*target_number_of_triangles_faction) 
                file_path_feat_exit=self.file_path_feat_exit
                fpath_true=self.dend_path_true_final_exit
                os.makedirs(fpath_true,exist_ok=True)
                os.makedirs(file_path_feat_exit,exist_ok=True)
 
                spine_index_all,count=[],[]
                intensity=np.zeros_like(vertices_00[:,0],dtype=int) 
                intensity_org=np.zeros_like(vertices_00[:,0],dtype=int) 
                mapp=mappings_vertices(vertices_0=vertices_00)

                vertices_00n = trimesh.load_mesh(os.path.join(self.dend_path_org_entry,f'{dend_name }.obj'),process=False ).vertices 
                   
                file_path_parent=os.path.dirname(file_path_feat_exit)
                np.savetxt(os.path.join(file_path_parent, 'faces_1.txt' ), faces, fmt='%d')
                np.savetxt(os.path.join(file_path_parent, 'vertices_0.txt' ), vertices_00n, fmt='%f') 
                pathi=os.path.join(self.obj_org_path_old,f'{self.dend_name}','data',f'{dend_name }.obj') 
                if os.path.exists(pathi):
                    mesh_old=trimesh.load_mesh(pathi,process=False ) 
                    vertices_000=mesh_old.vertices
                    np.savetxt(os.path.join(file_path_parent, self.txt_vertices_old ), vertices_000, fmt='%f') 
                    
                else:
                    print(f'Unavailable path in get_rhs====: {pathi} ')
                mesh_orgg = trimesh.load_mesh(os.path.join(dend_path_entry_data,f'{dend_name }.obj'),process=False  )  
                mapp=mappings_vertices(vertices_0=mesh_orgg.vertices)
                xxx=[ff for ff in os.listdir(dend_path_entry_data) if ff.endswith('.obj') ]
                pa_list=[ff for ff in xxx
                        if ff.startswith(f'{self.dend_first_name}{self.name_spine_id}')
                        and not ff.endswith('shaft.obj')
                        ]    
                if not pa_list:
                    continue
                spine_path=fpath_true 
                for pa in pa_list:
                    # nhn = trimesh.load_mesh(os.path.join(dend_path_entry_data,f'{dend_name }.obj'),process=False  ) 
                    nhn = trimesh.load_mesh(os.path.join(dend_path_entry_data,pa),process=False  ) 
                    vertices_index=mapp.Mapping_inverse(nhn.vertices) 
                    ip = int(pa.split("_")[-1].split(".")[0]) 
                    intensity_org[vertices_index ]=ip              
                    intensity[vertices_index]=1
                    spine_index_all.extend(vertices_index) 
                    # np.savetxt(os.path.join(shaft_path_resized, f'{self.name_spine_index}_{ip}.txt'), nhn.vertices_index, fmt='%d') 
                    # np.savetxt(os.path.join(shaft_path_resized, f'{self.name_spine_faces}_{ip}.txt'), nhn.faces, fmt='%d') 
                    # np.savetxt(os.path.join(shaft_path_resized, f'{self.name_spine_index_unique}_{ip}.txt'), nhn.vertices_faces_unique, fmt='%d')

                    np.savetxt(os.path.join(spine_path, f'{self.name_spine_index}_{ip}.txt'), vertices_index, fmt='%d') 
                    np.savetxt(os.path.join(spine_path, f'{self.name_spine_faces}_{ip}.txt'), nhn.faces, fmt='%d')  
                    
                    count.append(ip)  



                np.savetxt(os.path.join(spine_path,'spine_count.txt'),np.array(count), fmt='%d')  
                np.savetxt(os.path.join(spine_path, self.txt_spine_intensity ), intensity_org, fmt='%d')
            
                intensity_1hot=np.zeros_like(vertices_00[:,:-1],dtype=int) 
                if count:   
                    intensity_1hot[:,1:2][spine_index_all]=1 
                    intensity_1hot[:,0:1][list(set(np.arange(vertices_00.shape[0]))-set(spine_index_all))]=1
                    np.savetxt(os.path.join(spine_path,'intensity_shaft_spine.txt'), intensity, fmt='%d')
                    np.savetxt(os.path.join(spine_path,'intensity_1hot_shaft_spine.txt'), intensity_1hot, fmt='%d')


 

            dend_path_exit_data =self.dend_path_org_exit
            dend_path_entry_data=self.dend_path_org_entry
 
            skl_name=os.path.join(dend_path_entry_data,f'{self.dend_first_name}_skeleton.obj') 
            print('[[[[[[[[[[[[[vertices_00]]]]]]]]]]]]]',vertices_00.shape,dend_path_entry_data)
            if os.path.exists(skl_name): 
                mesh=trimesh.load_mesh(skl_name, process=False)   
                dtree = cKDTree(mesh.vertices).query(vertices_00)  
                vv=np.linalg.norm(vertices_00-mesh.vertices,axis=1)
                np.savetxt(os.path.join(file_path_feat_exit,txt_vertices), mesh.vertices, fmt='%f')
                np.savetxt(os.path.join(file_path_feat_exit,txt_distance), dtree[0].flatten(), fmt='%f')
                np.savetxt(os.path.join(file_path_feat_exit,txt_skl_index),dtree[1].flatten(), fmt='%d') 
                print('[[[[[[[[[[[[[vertices_00]]]skl_distance]]]]]]]]]]',skl_name,vertices_000.shape,vertices_00.shape,)
                print('[[[[[[[[[[[[[vertices_00]]]skl_distance]]]]]]]]]]',os.path.join(file_path_feat_exit,txt_vertices),)
 
            skl_name=os.path.join(dend_path_entry_data,f'{self.dend_first_name}_shaft_wrap.obj') 
            print('[[[[[[[[[[[[[]]]]]]]]]]]]]',skl_name)
            if os.path.exists(skl_name): 
                mesh=trimesh.load_mesh(skl_name, process=False) 
                mesh.export(os.path.join(file_path_feat_exit,f'{self.dend_first_name}_shaft_wrap.obj') ) 

 


            for pa in xxx:
                if not pa.endswith('skeleton.obj') or pa==f'{self.dend_first_name}_skeleton.obj':
                    continue
                # namm=f'{self.dend_first_name}_skeleton.obj',f'{self.dend_first_name}_{wrap_part}_skeleton.obj' or pa==f'{self.dend_first_name}_skeleton.obj'
                txt_distance=self.txt_skl_shaft_distance if wrap_part in pa else self.txt_skl_distance
                txt_vertices=self.txt_skl_shaft_vertices if wrap_part in pa else self.txt_skl_vertices 
                mesh_skl=trimesh.load_mesh(os.path.join(dend_path_entry_data,pa) , process=False)  
                dtree = cKDTree(mesh_skl.vertices).query(vertices_00)  
                np.savetxt(os.path.join(file_path_feat_exit,txt_distance),dtree[0].flatten(), fmt='%f')
                np.savetxt(os.path.join(file_path_feat_exit,txt_vertices),mesh_skl.vertices, fmt='%f')  


    def get_smooth(self,   
                  entry_name=None,
                    exit_name='smooth',
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None,  
                    txt_vertices_0='vertices_0.txt',
                    txt_vertices_1='vertices_1.txt' ,
                    txt_faces=None,
                    get_data=True,
                    n_error = 1,
                    n_step = None,
                    dt=1e-6, 
                    disp_time=5000,
                    method='willmore',
                    param_dic=None,
                    drop_dic_name=None,
                    old_path=None,
                    ):    
        param_dic = param_dic if param_dic is not None else self.param_dic
        import dend_fun_0.curvature as cuHP 
        from dend_fun_2.help_pinn_data_fun import smooth_curvature
        print(' get_smooth  ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            # self.get_dend_name(data_studied=data_studied,index=index ) 
            self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path, )   
            dend_name=self.dend_name 
            print(dend_name)   
            spine_path=self.path_file[path_train['dest_spine_path']]
            shaft_path=self.path_file[path_train['dest_shaft_path']]
            pid=pinn_data(file_path=self.file_path,
                            shaft_path=shaft_path,
                            spine_path=spine_path,
                            dend_path_original_m=self.dend_path_original_m, 
                            dend_first_name=self.dend_namess[index][1], 
                                    ) 
 

            dend_path_entry_data=self.dend_path_org_entry
            dend_path_exit_data =self.dend_path_org_exit

            os.makedirs(dend_path_exit_data,exist_ok=True)  
            mesh_org = trimesh.load_mesh(os.path.join(dend_path_entry_data,f'{dend_name }.obj') ,process=False )  

            faces =   mesh_org.faces 
            if get_data:
                vertices =mesh_org.vertices
            else:
                mesh_org = trimesh.load_mesh(os.path.join(dend_path_exit_data,f'{dend_name }.obj'),process=False  ) 
                vertices =mesh_org.vertices;faces=mesh_org.faces
            len_v0,len_f0=vertices.shape[0],faces.shape[0] 
            dend_path_exit=os.path.join(dend_path_exit_data,f'{dend_name}.obj') 
            get_feat = param_dic['tf_restart']['get_smooth'] 
            if os.path.exists(dend_path_exit) and not get_feat: 
                if param_dic['get_info']['get_smooth']:
                    print('[[[[[[get_smooth--  Exist Already or Restarting --]]]]]]',dend_path_exit)
                pass
            else:
                n_stepp = max(10, min(1e6, len(mesh_org.edges_unique) // 500)) if n_step is None else int(n_step) 
                print(f"Number of vertices: before {len(vertices)} | after {len_v0}")
                print(f"Number of faces:    before {len(faces)}    | after {len_f0}")  
                print(f"Number of iteration:       {n_stepp}        |n_edges: {len(mesh_org.edges_unique)}") 
                print(f"Method:                    {method}")  
                n_error=int(n_error)
                time_start = time.time()   
                if n_stepp>0:
                    print(f'Smoothing started: ')
                    if method.lower()=='taubin':
                        scurv=smooth_curvature(mesh=mesh_org,iterations=n_stepp)
                        # mesh_smooth=scurv.mesh_smooth
                        scurv.mesh_smooth.export(dend_path_exit)
                    else:# method.lower()=='willmore':
                        simu=cuHP.simulation(vertices=vertices,
                                        faces=faces,
                                        dt=dt,
                                        n_step=n_stepp,
                                        n_error=n_error,
                                        disp_time=disp_time,
                                        save_file=dend_path_exit_data,
                                        smooth_path=dend_path_exit, 
                                        txt_vertices=txt_vertices_0,
                                        )  
                # print('Job done!!',mesh_smooth.vertices.shape,mesh_org.vertices.shape)
            mesh_smooth=trimesh.load_mesh(dend_path_exit,process=False)  
            mytime0 = time.time() - time_start 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60) 
            print(f'get_smooth_one completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s') 


            maap=mappings_vertices(mesh_org.vertices)
            kname=[f for f in os.listdir(dend_path_entry_data) if  f.endswith('.obj') and f.startswith((f'{self.dend_first_name}{self.name_spine_id}',f'{self.dend_first_name}_shaft.obj'))]
            kname.append(f'{dend_name }.obj')
            for val in kname: 
                mesh = trimesh.load_mesh(os.path.join(dend_path_entry_data,val),process=False )  
                mmpp=maap.Mapping_inverse(mesh.vertices).astype(int)
                mesh.vertices=mesh_smooth.vertices[mmpp] 
                mesh.export(os.path.join(dend_path_exit_data,val))


 

    def get_scale(self,   
                  entry_names=[None],
                    exit_names=['scale'],
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None,
                    scaling=None,
                    param_dic=None,
                    drop_dic_name=None,
                    old_path=None,
                    ):   
        scalingT=copy.deepcopy(scaling)
        print('   ---- started')
        time_start = time.time()
        param_dic=param_dic if param_dic is not None else self.param_dic
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            if scalingT is None:
                scaling=np.random.uniform(0.9, 1.1)*100
            for entry_name,exit_name in zip(entry_names,exit_names):
                self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path, )  
                dend_name=self.dend_name 
                print(dend_name)   
                spine_path=self.path_file[path_train['dest_spine_path']]
                shaft_path=self.path_file[path_train['dest_shaft_path']]
                pid=pinn_data(file_path=self.file_path,
                                shaft_path=shaft_path,
                                spine_path=spine_path,
                                dend_path_original_m=self.dend_path_original_m, 
                                dend_first_name=self.dend_namess[index][1], 
                                        ) 

                if entry_name is None:
                    dend_path_entry=os.path.join(f'{self.obj_org_path}',self.dend_name,'data_org',)
                    dend_path_entry_data=os.path.join(f'{self.obj_org_path}',self.dend_name,'data',)
                else:
                    dend_path_entry=os.path.join(f'{self.obj_org_path}_{entry_name}',self.dend_name,'data_org',)
                    dend_path_entry_data=os.path.join(f'{self.obj_org_path}_{entry_name}',self.dend_name,'data',)
                if exit_name is None:
                    dend_path_exit=os.path.join(f'{self.obj_org_path}',self.dend_name,'data_org',)
                    dend_path_exit_data=os.path.join(f'{self.obj_org_path}',self.dend_name,'data',)
                else:
                    dend_path_exit=os.path.join(f'{self.obj_org_path}_{exit_name}',self.dend_name,'data_org',)
                    dend_path_exit_data=os.path.join(f'{self.obj_org_path}_{exit_name}',self.dend_name,'data',)
                #  os.makedirs(dend_path_exit,exist_ok=True) 
                os.makedirs(dend_path_exit_data,exist_ok=True) 
    

                kname=[f for f in os.listdir(dend_path_entry_data) if  f.endswith('.obj')]
                for val in kname: 
                    get_feat = param_dic['tf_restart']['get_scale'] 
                    dend_path_exit=os.path.join(dend_path_exit_data,val)
                    if os.path.exists(dend_path_exit) and not get_feat: 
                        if param_dic['get_info']['get_scale']:
                            print('[[[[[[--  Exist Already or Restarting --]]]]]]',dend_path_exit)
                        pass
                    else: 
                        mesh = trimesh.load_mesh(os.path.join(dend_path_entry_data,val),process=False )
                        mesh.apply_scale(scaling=scaling/100)  
                        mesh.export(dend_path_exit)

            scalingT=copy.deepcopy(scaling)

    def get_resize(self,   
                  entry_names=[None],
                    exit_names=[None],
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None, 
                    target_number_of_triangles_faction=None,
                    path=None,
                    size_threshold=3,
                    drop_dic_name=None,
                    old_path=None,
                    ):    
        from dend_fun_2.help_pinn_data_fun import mesh_resize
        if target_number_of_triangles_faction is None:
            target_number_of_triangles_faction=int(np.random.uniform(0.9, 1.1)*1000)
        print(' get_resize  ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        
        for  index,dend_name in enumerate(dend_names):
            skl=None
            for entry_name,exit_name in zip(entry_names,exit_names):
                self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name, old_path=old_path,)  
                dend_name=self.dend_name   
                print(dend_name)   
                spine_path=self.path_file[path_train['dest_spine_path']]
                shaft_path=self.path_file[path_train['dest_shaft_path']] 
                dend_path_entry_data = {
                                        "entry": self.dend_path_org_entry,
                                        "old":   self.dend_path_org_old,
                                        "exit":  self.dend_path_org_exit
                                    }.get(path, self.dend_path_org_entry)

                dend_path_exit_data=self.dend_path_org_exit 
                print('[[[[[[[[[[[[[]]]]]]]]]]]]]',dend_path_exit_data)
                os.makedirs(dend_path_exit_data,exist_ok=True) 

                mesh_org = trimesh.load_mesh(os.path.join(dend_path_entry_data,f'{dend_name }.obj'),process=False  ) 
                vertices_00=mesh_org.vertices
                faces=mesh_org.faces
                target_number_of_triangles=min(2*vertices_00.shape[0] ,2*target_number_of_triangles_faction)
                if skl is None:
                    skl=mesh_resize(vertices=vertices_00,
                                            faces=faces,
                                            target_number_of_triangles=target_number_of_triangles ,) 
                xxx=[ff for ff in os.listdir(dend_path_entry_data) if ff.endswith('.obj') ]
                kname=[f for f in xxx if  f.endswith('.obj') and 
                    f.startswith((f'{self.dend_first_name}{self.name_spine_id}',f'{self.dend_first_name}_shaft.obj'))]
                kname.append(f'{dend_name }.obj')
                for pa in kname:  
                    meshi = trimesh.load_mesh(os.path.join(dend_path_entry_data,pa),process=False  )  
                    print(meshi.vertices.shape)
                    meshhh = skl.reduce_rhs(label=meshi.vertices,
                                            tf_mesh=True,
                                            size_threshold=size_threshold,
                                            label_gen=vertices_00,) 
                    print(meshhh.vertices.shape)
                    # print('=======,ve',pa)
                    if meshhh is not None:
                        meshhh.export(os.path.join(dend_path_exit_data,pa) ) 
                    else:
                        meshi.export(os.path.join(dend_path_exit_data,pa) ) 

                xpath=os.path.join(dend_path_entry_data,f'{self.dend_first_name}_skeleton.obj')
                if os.path.exists(xpath):
                    meshi = trimesh.load_mesh(xpath,process=False ) 
                    meshhh = skl.reduce_rhs(meshi.vertices,tf_mesh=True,size_threshold=size_threshold,)  
                    meshhh.export(os.path.join(dend_path_exit_data,f'{self.dend_first_name}_skeleton.obj') ) 
                    kname.append(f'{self.dend_first_name}_skeleton.obj')
 
                for pa in xxx:
                    if pa in kname:
                        continue
                    skl_name=os.path.join(dend_path_entry_data,pa) 
                    # print('[[[[[[[[[[[[[]]]]]]]]]]]]]',skl_name)
                    if os.path.exists(skl_name): 
                        mesh=trimesh.load_mesh(skl_name, process=False) 
                        mesh.export(os.path.join(dend_path_exit_data,pa) )  




    def get_gaussian_jitter(self,   
                  entry_name=None,
                    exit_name=None,
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None, 
                    sigma=None,
                    drop_dic_name=None,
                    old_path=None,
                    ):   
        if sigma is None:
            sigma=np.random.uniform(0.9, 1.1)/1000
        print('   ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path, )  
            dend_name=self.dend_name 
            print(dend_name)   
            spine_path=self.path_file[path_train['dest_spine_path']]
            shaft_path=self.path_file[path_train['dest_shaft_path']]
            pid=pinn_data(file_path=self.file_path,
                            shaft_path=shaft_path,
                            spine_path=spine_path,
                            dend_path_original_m=self.dend_path_original_m, 
                            dend_first_name=self.dend_namess[index][1], 
                                    )  




            dend_path_exit_data =self.dend_path_org_exit
            dend_path_entry_data=self.dend_path_org_entry 
            os.makedirs(dend_path_exit_data,exist_ok=True) 


            mesh_org = trimesh.load_mesh(os.path.join(dend_path_entry_data,f'{dend_name }.obj'),process=False ) 
            vertices=mesh_org.vertices+ np.random.normal(scale=sigma, size=mesh_org.vertices.shape) 

            maap=mappings_vertices(mesh_org.vertices)
            kname=[f for f in os.listdir(dend_path_entry_data) if  f.endswith('.obj')]
            for val in kname: 
                mesh = trimesh.load_mesh(os.path.join(dend_path_entry_data,val),process=False ) 
                if not val.endswith('skeleton.obj'): 
                    mmpp=maap.Mapping_inverse(mesh.vertices).astype(int)
                    mesh.vertices=vertices[mmpp]
                    print('---------------',mesh.vertices.shape,mmpp.shape) 
                mesh.export(os.path.join(dend_path_exit_data,val)) 


 


    def get_pca(self,   
                    entry_names=[None] ,
                    exit_names=[None] ,
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None, 
                    drop_dic_name=None,
                    old_path=None,
                    ):   
          
        print(' get_pca  ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            pcap=None
            for entry_name,exit_name in zip(entry_names,exit_names):
                self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path, )  
                dend_name=self.dend_name 
                print(dend_name)   
                spine_path=self.path_file[path_train['dest_spine_path']]
                shaft_path=self.path_file[path_train['dest_shaft_path']]
                pid=pinn_data(file_path=self.file_path,
                                shaft_path=shaft_path,
                                spine_path=spine_path,
                                dend_path_original_m=self.dend_path_original_m, 
                                dend_first_name=self.dend_namess[index][1], 
                                        )  

                dend_path_exit_data =self.dend_path_org_exit
                dend_path_entry_data=self.dend_path_org_entry 
                os.makedirs(dend_path_exit_data,exist_ok=True) 
    


                mesh_org = trimesh.load_mesh(os.path.join(dend_path_entry_data,f'{dend_name }.obj'),process=False )
                if pcap is None:
                    pcap=pca_projector(mesh_org.vertices)  

                kname=[f for f in os.listdir(dend_path_entry_data) if  f.endswith('.obj')]
                for val in kname: 
                    mesh = trimesh.load_mesh(os.path.join(dend_path_entry_data,val),process=False )
                    mesh.vertices=pcap.project(mesh.vertices)
                    mesh.export(os.path.join(dend_path_exit_data,val)) 






    def get_rotation(self,   
                    entry_names=[None] ,
                    exit_names=[None] ,
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None, 
                    angle=None,
                    drop_dic_name=None,
                    old_path=None,
                    ):    
        print('   ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names
        angleT=copy.deepcopy(angle) 
        for  index,dend_name in enumerate(dend_names):
            if angleT is None: 
                ax = np.random.uniform(-np.pi, np.pi) 
                ay = np.random.uniform(-np.pi, np.pi) 
                az = np.random.uniform(-np.pi, np.pi)
            else:
                # Convert to radians
                angle_x,angle_y,angle_z=angleT
                ax = np.deg2rad(angle_x)
                ay = np.deg2rad(angle_y)
                az = np.deg2rad(angle_z)

            R_x = trimesh.transformations.rotation_matrix(ax, [1, 0, 0])
            R_y = trimesh.transformations.rotation_matrix(ay, [0, 1, 0])
            R_z = trimesh.transformations.rotation_matrix(az, [0, 0, 1])

            R = R_z @ R_y @ R_x
 
            for entry_name, exit_name in zip(entry_names,exit_names):
                self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path, )  
                dend_name=self.dend_name 
                print(dend_name)   
                spine_path=self.path_file[path_train['dest_spine_path']]
                shaft_path=self.path_file[path_train['dest_shaft_path']]
                pid=pinn_data(file_path=self.file_path,
                                shaft_path=shaft_path,
                                spine_path=spine_path,
                                dend_path_original_m=self.dend_path_original_m, 
                                dend_first_name=self.dend_namess[index][1], 
                                        )  

                dend_path_exit_data =self.dend_path_org_exit
                dend_path_entry_data=self.dend_path_org_entry
    
                os.makedirs(dend_path_exit_data,exist_ok=True)  
                mesh_org = trimesh.load_mesh(os.path.join(dend_path_entry_data,f'{dend_name }.obj'),process=False ) 


                mesh_org.apply_transform(R)

                kname=[f for f in os.listdir(dend_path_entry_data) if  f.endswith('.obj')]
                for val in kname: 
                    mesh = trimesh.load_mesh(os.path.join(dend_path_entry_data,val),process=False )
                    mesh.apply_translation(-mesh_org.centroid)
                    mesh.apply_transform(R)
                    mesh.apply_translation(mesh_org.centroid)  
                    mesh.export(os.path.join(dend_path_exit_data,val)) 
            angleT=copy.deepcopy(angle) 


    def get_translation(self,   
                    exit_names=[None] ,
                    entry_names=[None] ,
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None, 
                    translation=None,
                    drop_dic_name=None,
                    old_path=None,
                    ):    
        print('get_translation   ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            translationT=copy.deepcopy(translation)
            if translationT is None:
                translationT= np.random.uniform(-10, 10, size=3)


            for entry_name, exit_name in zip(entry_names,exit_names):
                self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name,old_path=old_path, )  
                dend_name=self.dend_name 
                print(dend_name)    

                dend_path_exit_data =self.dend_path_org_exit
                dend_path_entry_data=self.dend_path_org_entry
    
                os.makedirs(dend_path_exit_data,exist_ok=True) 
                mesh_org = trimesh.load(os.path.join(dend_path_entry_data,f'{dend_name }.obj'),process=False ) 
    
                T = trimesh.transformations.translation_matrix(translationT)  


                kname=[f for f in os.listdir(dend_path_entry_data) if  f.endswith('.obj')]
                for val in kname: 
                    mesh = trimesh.load_mesh(os.path.join(dend_path_entry_data,val),process=False )
                    mesh.apply_translation(-mesh_org.centroid)
                    mesh.apply_transform(T)
                    mesh.apply_translation(mesh_org.centroid)  
                    mesh.export(os.path.join(dend_path_exit_data,val))
            translationT=copy.deepcopy(translation)





    def get_path_remove(self,   
                    exit_name ,
                    entry_name=None ,
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None, 
                    drop_dic_name=None,
                    old_path=None,
                    ):   
         
        from dend_fun_0.help_funn import remove_directory
        print('   ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index,entry_name=entry_name,exit_name=exit_name,data_org='data',drop_dic_name=drop_dic_name, old_path=old_path,)  
            dend_name=self.dend_name 
            print(dend_name)   
            spine_path=self.path_file[path_train['dest_shaft_path']]
            shaft_path=self.path_file[path_train['dest_shaft_path']]
            pid=pinn_data(file_path=self.file_path,
                            shaft_path=shaft_path,
                            spine_path=spine_path,
                            dend_path_original_m=self.dend_path_original_m, 
                            dend_first_name=self.dend_namess[index][1], 
                                    )  



            dend_path_exit_data =self.dend_path_org_exit
            dend_path_entry_data=self.dend_path_org_entry
 
        remove_directory(os.path.join(f'{self.obj_org_path}_{entry_name}') )  
 
 







    def get_diane_spines(self, 
                    entry_name_shaft=None,   
                    exit_name='wrap',
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None,
                    k_query=7,
                    size_threshold=5,
                    ):   
          
        print('   ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index )  
            dend_name=self.dend_name 
            dend_first_name=self.dend_namess[index][1]
            name_spine_id=self.name_spine_id
            print(dend_name)   
            spine_path=self.path_file[path_train['dest_spine_path']]
            shaft_path=self.path_file[path_train['dest_shaft_path']]
            pid=pinn_data(file_path=self.file_path,
                            shaft_path=shaft_path,
                            spine_path=spine_path,
                            dend_path_original_m=self.dend_path_original_m, 
                            dend_first_name=self.dend_namess[index][1], 
                                    ) 

            if entry_name_shaft is None:
                dend_path_entry=os.path.join(f'{self.obj_org_path}',self.dend_name,'data_org',)
            else:
                dend_path_entry=os.path.join(f'{self.obj_org_path}_{entry_name_shaft}',self.dend_name,'data_org',)
            if exit_name is None:
                dend_path_exit=os.path.join(f'{self.obj_org_path}',self.dend_name,'data_org',)
            else:
                dend_path_exit=os.path.join(f'{self.obj_org_path}_{exit_name}',self.dend_name,'data_org',)

            # pid.get_dend_data()  
            # mesh_org_path_file=os.path.join(f'{self.obj_org_path}_{entry_name}',self.dend_name,'data_org',)
            shaft_path_file=dend_path_entry#os.path.join(self.obj_org_path,self.dend_name,'data_org')  
            mesh_org_path_file=spines_true_path=dend_path_exit#os.path.join(f'{self.obj_org_path}_{exit_name}',self.dend_name,'data_org',)
            os.makedirs(spines_true_path,exist_ok=True)  
            pid.get_diane_spines(dend_name,dend_first_name,name_spine_id,mesh_org_path_file,shaft_path_file,spines_true_path,k_query=k_query,size_threshold=size_threshold)









    def get_skl_shaft_pred(self, 
                        path_train=None, 
                        pre_portion=None, 
                        n_col=None,
                        hidden_layers=None, 
                        neurons_per_layer=None, 
                        activation_init=None,
                        activation_hidden=None,
                        activation_last=None, 
                        model= None,
                        curv=None,
                        shaft_path=None,
                        path_shaft_dir=None,
                        size_threshold=None,
                        dend_names=None, 
                        model_dir=None, 
                        loss_save_dir=None,
                        iou_save_dir=None,
                        auc_save_dir=None,
                        model_sufix=None, 
                        data_studied=None,
                        file_path_org=None,  
                        line_num_points_shaft=None,
                        line_num_points_inter_shaft=None,
                        spline_smooth_shaft=None ,  
                        disp_infos=None, 
                        DTYPE=None,
                        list_features=None,
                        base_features_list=None,
                        shaft_thre=None,
                        train_spines=False,
                        weight=None,
                        weights=None,
                        neck_lim=None,
                        n_clusters=3, 
                        kmean_max_iter=300,
                        smooth_tf=False,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        reconstruction_tf=False, 
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        tf_skl_shaft_distance=False,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        param_dic=None,
                        drop_dic_name=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution
        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        size_threshold=size_threshold or self.size_threshold
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos  
        file_path_org = file_path_org or self.file_path_org
        model_sufix=model_sufix or self.model_sufix
        self.get_model_opt_name(model_sufix=model_sufix,
                                model_type=model_type,)  
        line_num_points_shaft = line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft = line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft = spline_smooth_shaft or self.spline_smooth_shaft
        DTYPE=DTYPE or self.DTYPE 
        data_studied = data_studied or self.data_studied 

        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last

        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou']    
        auc_save_dir = auc_save_dir or  model_dirs['auc'] 
        model_dir = model_dir or  model_dirs['model']   

        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name]) 
        print('-------------',n_col,model_dir)

  
        print(f'get_skl_shaft_pred started')
        print('--------------------------------')
        print(f"Model Type Gen : {model_type}")
        print(f"Model Type     : {model_sufix}")
        print(f"Model Dir.     : {model_dir}")
        print(f"Model Portion  : {pre_portion}") 
        print(f"Data Dir.      : {path_train['data_shaft_path']}") 
        print(f"Destin. Dir.   : {path_train['dest_shaft_path']}") 
        print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
        print(f"Feat. list     : {[mm for _,mm in list_features]}")
        print('--------------------------------')
        time_start = time.time()
        rhs_name='rhs_name'
        dend_names = dend_names or self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, 
                                ** param_dic['data']['get_dend_name'] )  
            dend_name=self.dend_name  
            spine_portion_path=self.path_file[path_train['data_spine_path']] 
            shaft_portion_path=self.path_file[path_train['data_shaft_path']]  
            cxc=param_dic['data']['get_dend_name']['dict_dend_path']

            file_path_feat=self.dict_dend['path'][cxc]['file_path_feat']
            file_path=self.dict_dend['path'][cxc]['file_path']
            print('[[[[[[[[]]]]]]]]',file_path_feat,file_path)
            pid = pinn_data(file_path= file_path,
                            file_path_feat= file_path_feat,
                            path_file=self.path_file,
                            spine_path = spine_portion_path,
                            shaft_path =  shaft_portion_path , 
                            spine_path_pre=spine_portion_path,
                            shaft_path_pre=  shaft_portion_path ,  
                            dend_path_original_m=self.dend_path_original_m,
                            dend_first_name=self.dend_namess[index][1],
                            model_sufix=model_sufix,
                            path_train=path_train,
                            line_num_points_shaft=line_num_points_shaft,
                            line_num_points_inter_shaft=line_num_points_inter_shaft,
                            spline_smooth_shaft=spline_smooth_shaft,
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                        )  
            pid.save_pinn_data() 
            pid.get_dend_data()
            con=0
            path_shaft_dir=path_shaft_dir or f'{self.model_type}_{self.model_sufix}_{self.path_dir}'
            shaft_path=  self.path_file[path_shaft_dir]  
            # shaft_path= self.path_file[f'{self.model_type}_{self.model_sufix}_{self.path_dir}']
            shaft_path=  self.path_file[path_train['data_shaft_path']]  
            dest_path = self.path_file[path_train['dest_shaft_path']] 
            # print('[[[[[get_skl_shaft_pred location]]]]]',shaft_path)

            pid.get_skl_shaft_pred( 
                            shaft_path=shaft_path,
                            dest_path=dest_path,
                            weight=weight,
                            weights=weights, 
                            path_train=path_train, 
                            pre_portion=pre_portion,
                            gauss_threshold=self.gauss_threshold,
                            size_threshold=size_threshold, 
                            shaft_thre=shaft_thre,
                            smooth_tf=smooth_tf,
                            neck_lim=neck_lim,
                            dict_mesh_to_skeleton_finder=dict_mesh_to_skeleton_finder,
                            dict_wrap=dict_wrap,
                            tf_skl_shaft_distance=tf_skl_shaft_distance, 
                            ) 
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                # print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 




 



    def get_shaft_process(self, 
                        path_train=None, 
                        pre_portion=None, 
                        n_col=None,
                        hidden_layers=None, 
                        neurons_per_layer=None, 
                        activation_init=None,
                        activation_hidden=None,
                        activation_last=None, 
                        model= None,
                        curv=None,
                        size_threshold=None,
                        dend_names=None, 
                        model_dir=None, 
                        loss_save_dir=None,
                        iou_save_dir=None,
                        auc_save_dir=None,
                        model_sufix=None, 
                        data_studied=None,
                        file_path_org=None,  
                        line_num_points_shaft=None,
                        line_num_points_inter_shaft=None,
                        spline_smooth_shaft=None ,  
                        disp_infos=None, 
                        DTYPE=None,
                        list_features=None,
                        base_features_list=None,
                        shaft_thre=None,
                        train_spines=False,
                        weight=None,
                        weights=None,
                        neck_lim=None,
                        n_clusters=3, 
                        kmean_max_iter=300,
                        smooth_tf=False,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        reconstruction_tf=False, 
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        tf_skl_shaft_distance=False,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        param_dic=None,
                        drop_dic_name=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution
        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        size_threshold=size_threshold or self.size_threshold
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos  
        file_path_org = file_path_org or self.file_path_org
        model_sufix=model_sufix or self.model_sufix
        self.get_model_opt_name(model_sufix=model_sufix,
                                model_type=model_type,)  
        line_num_points_shaft = line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft = line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft = spline_smooth_shaft or self.spline_smooth_shaft
        DTYPE=DTYPE or self.DTYPE 
        data_studied = data_studied or self.data_studied 

        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last

        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou']    
        auc_save_dir = auc_save_dir or  model_dirs['auc'] 
        model_dir = model_dir or  model_dirs['model']   

        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name]) 
        print('-------------',n_col,model_dir)

  
        print(f'get_shaft_process started')
        print('--------------------------------')
        print(f"Model Type Gen : {model_type}")
        print(f"Model Type     : {model_sufix}")
        print(f"Model Dir.     : {model_dir}")
        print(f"Model Portion  : {pre_portion}") 
        print(f"Data Dir.      : {path_train['data_spine_path']}") 
        print(f"Destin. Dir.   : {path_train['dest_spine_path']}") 
        print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
        print(f"Feat. list     : {[mm for _,mm in list_features]}")
        print('--------------------------------')
        time_start = time.time()
        rhs_name='rhs_name'
        dend_names = dend_names or self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type,  
                                **param_dic['data']['get_dend_name'])  
            dend_name=self.dend_name  
            spine_portion_path=self.path_file[path_train['data_spine_path']] 
            shaft_portion_path=self.path_file[path_train['data_shaft_path']] 
            cxc=param_dic['data']['get_dend_name']['dict_dend_path']

            file_path_feat=self.dict_dend['path'][cxc]['file_path_feat']
            file_path=self.dict_dend['path'][cxc]['file_path'] 
            pid = pinn_data(
                            file_path= file_path,
                            file_path_feat= file_path_feat,
                            path_file=self.path_file,
                            spine_path = spine_portion_path,
                            shaft_path =  shaft_portion_path , 
                            spine_path_pre=spine_portion_path,
                            shaft_path_pre=  shaft_portion_path ,  
                            dend_path_original_m=self.dend_path_original_m,
                            dend_first_name=self.dend_namess[index][1],
                            model_sufix=model_sufix,
                            path_train=path_train,
                            line_num_points_shaft=line_num_points_shaft,
                            line_num_points_inter_shaft=line_num_points_inter_shaft,
                            spline_smooth_shaft=spline_smooth_shaft,
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                        )  
            pid.save_pinn_data() 
            pid.get_dend_data()
            con=0 
            rhs0=[] 
            shaft_path = self.path_file[path_train['dest_shaft_path']] 
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(self.intensity_spines_logit): 
               rhs0.append(np.loadtxt(self.path_file_sub[tyy][key], dtype=float)) 
            rhs0=np.array(rhs0).T
            pid.get_shaft_pred(
                            # dend=pid.dend,
                            rhs=rhs0,
                            weight=weight,
                            weights=weights, 
                            path_train=path_train, 
                            pre_portion=pre_portion,
                            gauss_threshold=self.gauss_threshold,
                            size_threshold=size_threshold, 
                            shaft_thre=shaft_thre,
                            smooth_tf=smooth_tf,
                            neck_lim=neck_lim,
                            dict_mesh_to_skeleton_finder=dict_mesh_to_skeleton_finder,
                            dict_wrap=dict_wrap,
                            tf_skl_shaft_distance=tf_skl_shaft_distance, 
                            ) 
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 





