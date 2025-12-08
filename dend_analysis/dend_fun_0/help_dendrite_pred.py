

import sys
import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import time 
import tensorflow as tf  
tf.config.run_functions_eagerly(True) 
from tensorflow.keras.models import load_model
DTYPE='float32' 
import pickle 
import pandas as pd
 
import shap  

from dend_fun_0.curvature import curv_mesh as curv_mesh
import dend_fun_0.help_funn as hff   
from dend_fun_0.help_graph import graph_iou,graph_cylinder_heatmap,get_iou_graph 
from dend_fun_0.help_funn import get_intensity 
DTYPE='float32' 
DTYPE = tf.float32
 
 
from dend_fun_0.get_path import assign_if_none,get_name,get_param,get_files,remove_directory,safe_id
from dend_fun_0.side_bar import  get_text_dash_train,get_text_dash_all,get_text_dash_test,get_text_dash_dnn

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"  
from tqdm import tqdm 

from dend_fun_0.help_save_iou import iou_train 
from dend_fun_2.help_pinn_data_222 import pinn_data 
 

import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)



class dendrite_pred(get_files,get_name):
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
                        )
        self.dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh
        if data_mode is not None:
            path_train=data_mode['path_train']
            pre_portion=data_mode['pre_portion']
            pinn_dir_data=data_mode['pinn_dir_data']
            list_features=data_mode['list_features']
            base_features_list=data_mode['base_features_list']
        self.path_display=path_display if path_display is not None else ['dest_spine_path_pre','dest_spine_path','dest_shaft_path']
        path_dir=os.path.join(file_path_org, 'data')
        np.savetxt(os.path.join(path_dir, 'model_sufix_all.txt'), np.array(model_sufix_all), fmt='%s') 
        np.savetxt(os.path.join(path_dir, 'pinn_dir_data_all.txt'), np.array(pinn_dir_data_all), fmt='%s')
        np.savetxt(os.path.join(path_dir, 'path_heads.txt'), np.array(path_heads), fmt='%s')
        np.savetxt(os.path.join(path_dir, 'true_keys.txt'), np.array(true_keys), fmt='%s') 
    def get_intensity_rhs(self, 
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
                        ): 

        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution  
        disp_infos = disp_infos or self.disp_infos
        radius_threshold = radius_threshold or self.radius_threshold 
        dend_path_inits = dend_path_inits or self.dend_path_inits
        dend_names = dend_names or self.dend_names
        file_path_org = file_path_org or self.file_path_org
        data_studied = data_studied or self.data_studied 
        numNeighbours = numNeighbours or self.numNeighbours
        obj_org_path = obj_org_path or self.obj_org_path 

        print('get_intensity_rhs Started')
        
        for  index,dend_name in enumerate(dend_names): 
            self.get_dend_name(data_studied=data_studied,index=index,  )   
            if disp_infos:
                print(f"Annotation path original: {self.file_path}")  
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
                                            ) 
                    pid.get_intensity_rhs()



    def get_annotations(self, 
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
                        ): 
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution  
        disp_infos = disp_infos or self.disp_infos
        radius_threshold = radius_threshold or self.radius_threshold 
        dend_path_inits = dend_path_inits or self.dend_path_inits
        dend_names = dend_names or self.dend_names
        file_path_org = file_path_org or self.file_path_org
        data_studied = data_studied or self.data_studied 
        numNeighbours = numNeighbours or self.numNeighbours
        obj_org_path = obj_org_path or self.obj_org_path 

        print('Annotation Started')
        
        for  index,dend_name in enumerate(dend_names): 
            self.get_dend_name(data_studied=data_studied,index=index,  )  
            # if restart:
            if disp_infos:
                print(f"Annotation path original: {self.file_path}")  
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
                    pid.get_annotation(
                        dend_path_original_new_smooth=self.dend_path_original_new_smooth,
                        dend_name=dend_name,
                        dend_path_org_new=self.dend_path_org_new,
                        ) 

 


    def get_annotation_resized(self, 
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
                        target_number_of_triangles_faction=1000,
                        ): 
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh 
        disp_infos = disp_infos or self.disp_infos
        radius_threshold = radius_threshold or self.radius_threshold 
        dend_path_inits = dend_path_inits or self.dend_path_inits
        dend_names = dend_names or self.dend_names
        file_path_org = file_path_org or self.file_path_org
        data_studied = data_studied or self.data_studied 
        numNeighbours = numNeighbours or self.numNeighbours
        obj_org_path = obj_org_path or self.obj_org_path 

        print('Aget_annotation_resized Started')
        
        for  index,dend_name in enumerate(dend_names):

            self.get_dend_name(data_studied=data_studied,
                               index=index,  )  
            # if restart:
            if disp_infos:
                print(f"get_annotation_resized path original: {self.file_path}")  
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
                    file_path_resized=self.file_path_resized
                    shaft_path_resized=self.path_file[f'resized_{key}']
                    if disp_infos:
                        print(f"Annotation path resized: {file_path_resized}  ") 
                        print(f"Annotation spsh resized: {shaft_path_resized}") 
                    pid.get_annotation_resized(train_tf=annotation_resized_train_tf,
                                               file_path_resized=file_path_resized,
                                               shaft_path_resized=shaft_path_resized,
                                                thre_target_number_of_triangles=thre_target_number_of_triangles,
                                                dend_name=dend_name,
                                                dend_path_org_resized =self.dend_path_org_resized,
                                                dend_path_org_smooth_resized=self.dend_path_org_smooth_resized,
                                                min_target_number_of_triangles_faction=min_target_number_of_triangles_faction,
                                                target_number_of_triangles_faction=target_number_of_triangles_faction,
                                               ) 

        self.obj_org_path=obj_org_path



 
    def get_intensity_all(self, 
                            dend_names=None,
                            file_path_org=None,
                            file_path=None, 
                            file_path_feat=None,
                            dend_path_inits = None, 
                            radius_threshold=None,
                            disp_infos=None,
                            part='spine',
                            thr_gauss=45,
                            thr_mean=15,): 
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos
        radius_threshold = radius_threshold if radius_threshold is not None else self.radius_threshold 
        dend_path_inits = dend_path_inits if dend_path_inits is not None else self.dend_path_inits
        dend_names = dend_names if dend_names is not None else self.dend_names 


        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(index=index, ) 
            file_path=self.file_path 
            file_path_feat = self.file_path_feat 
            faces=np.loadtxt(os.path.join(file_path, self.txt_faces), dtype=int)
            name_gauss=[self.txt_gauss_curv_init,self.txt_gauss_curv_smooth]
            name_mean=[self.txt_mean_curv_init,self.txt_mean_curv_smooth]
            pathh=[os.path.join(self.file_path, pa) for pa in [self.txt_vertices_1,self.txt_vertices_0]]
            for pa,ga,me in zip(pathh,name_gauss,name_mean):
                if os.path.exists(path=pa): 
                    get_intensity(
                        vertices            =np.loadtxt(pa, dtype=float) , 
                        faces               =faces, 
                        file_path_gauss_full=os.path.join(file_path_feat, ga),
                        file_path_mean_full =os.path.join(file_path_feat, me),   
                        thr_gauss=thr_gauss,
                        thr_mean=thr_mean,
                        )

 
 

    def get_train_input(self,  
                        path_train, 
                        pre_portion,  
                        DTYPE=None, 
                        file_path_model_data=None,
                        data_studied=None, 
                        line_num_points_shaft=None,
                        line_num_points_inter_shaft=None,
                        spline_smooth_shaft=None,   
                        model_sufix=None,
                        disp_infos=None,
                        txt_save_file=None,
                        dend_names=None,
                        weight_positive=.5,  
                        list_features=None,
                        base_features_list=None,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        ): 
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos  
        model_sufix = model_sufix or self.model_sufix  
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied  
        line_num_points_shaft = line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft = line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft = spline_smooth_shaft or self.spline_smooth_shaft
        dend_names = dend_names if dend_names is not None else self.dend_names
        DTYPE=DTYPE or self.DTYPE 

        curv, rhs, weight, indices,adj,dend = [], [], [], [],[],[]

        for index, dend_name in enumerate(dend_names): 
            self.get_dend_name(data_studied=data_studied, index=index,model_type=model_type) 
            spine_portion_path=self.path_file[path_train['data_spine_path']] 
            shaft_portion_path=self.path_file[path_train['data_shaft_path']] 
            pid = pinn_data(file_path=self.file_path,
                            file_path_feat=self.file_path_feat,
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
            feat_paths=[]  
            for path_inten,name_inten in list_features:  
                pathh=os.path.join( self.file_path_feat ,name_inten)
                if os.path.exists(pathh):
                    feat_paths.append(pathh)
                    print('train data path ---->>',self.file_path_feat,pathh) 
                    print(np.loadtxt(pathh))
                else:
                    print('path doesnt exists ===----->>>',pathh)

                # print('path',self.path_file[path_train[path_inten]]) 
            if model_type.startswith('gcn'):
                adj.append(hff.adjoint(vertices=pid.dend.vertices,faces=pid.dend.faces,num_sub_nodes=num_sub_nodes))
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
                rhs.append(pid.get_pinn_rhs(pre_portion=pre_portion))  
                dend.append(pid.dend)  
            elif   model_type.startswith(('pinn','rpinn')):
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
                rhs.append(tf.cast(pid.get_pinn_rhs(pre_portion=pre_portion), dtype=DTYPE))
                dend.append(pid.dend)  
            elif model_type.startswith(('ML','cML')): 
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
                rhs.append(pid.get_pinn_rhs(pre_portion=pre_portion))  

        return curv, rhs ,adj,dend

 


    def train_model_spine(self,
                            path_train=None,
                            get_training=True, 
                            pre_portion=None,
                            full_dend=True, 
                            hidden_layers=None, 
                            neurons_per_layer=None, 
                            activation_init=None,
                            activation_hidden=None,
                            activation_last=None, 
                            curv=None,
                            rhs=None,
                            weight=None,
                            indices=None,
                            DTYPE=None, 
                            model_sufix=None,
                            file_path_model_data=None,
                            line_num_points_shaft=None,
                            line_num_points_inter_shaft=None, 
                            spline_smooth_shaft=None,
                            data_studied=None,
                            vv_cts=None,
                            new_model=True, 
                            itime = 10000, 
                            itime_div=1,
                            loss_save_dir=None,
                            iou_save_dir=None,   
                            index_save_dir=None,  
                            model_dir=None, 
                            loss_mode="bce",
                            ls=[2,3,4,5],
                            disp_infos=None, 
                            weight_positive= .5,
                            list_features=None,
                            base_features_list=None,
                            train_spines=False,
                            dest_path='dest_spine_path',
                            model_type=None,
                            num_sub_nodes=None,
                            rl_par= 0.5, 
                            dnn_par= 0.5,
                            thre_target_number_of_triangles=None,
                            voxel_resolution=None,
                            l1_values = [0,   1e-6,   1e-4, 1e-2],
                            l2_values = [0,   1e-6,   1e-4, 1e-2],
                            dict_mesh_to_skeleton_finder_mesh=None,
                        ): 
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh 
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos 
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        DTYPE = DTYPE or self.DTYPE
        vv_cts = vv_cts or self.vv_cts 
        model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type)
        line_num_points_shaft=line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft=line_num_points_inter_shaft or self.line_num_points_inter_shaft
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied 
        spline_smooth_shaft=self.spline_smooth_shaft
 
 
        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last
 
        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou'] 
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']   
 
        if model_type=='gcn':
            from dend_fun_0.help_gcn_one_hot import LOSS,PINN,aka_train,Get_iou 
            adj_tf=True
        elif model_type.startswith('pinn'):
            from dend_fun_0.help_pinn_one_hot import LOSS,PINN,aka_train,Get_iou 
            adj_tf=False
        elif model_type.startswith('rpinn'):
            from dend_fun_0.help_pinn_rein_one_hot  import LOSS,PINN,aka_train,Get_iou 
            adj_tf=False

 
        curv,rhs,adj,dend =self.get_train_input(   
                                path_train=path_train,  
                                pre_portion=pre_portion, 
                                line_num_points_shaft=line_num_points_shaft,
                                line_num_points_inter_shaft=line_num_points_inter_shaft, 
                                model_sufix=model_sufix, 
                                list_features=list_features,
                                base_features_list=base_features_list, 
                                model_type=model_type,
                                num_sub_nodes=num_sub_nodes, 
                                thre_target_number_of_triangles=thre_target_number_of_triangles,
                                voxel_resolution=voxel_resolution,
                                dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                ) 
        curv_train = [curv[i] for i in ls]
        rhs_train = [rhs[i] for i in ls] 
        dend_train=[dend[i] for i in ls]
        lss=np.arange(len(curv),dtype=int) 
        adj_train =[]
        if model_type=='gcn':
            adj_train =[adj[i] for i in ls]
 
                
        class PINN_tmp(PINN):
            def __init__(self, Dtype=DTYPE, 
                        hidden_layers=hidden_layers, 
                        neurons_per_layer=neurons_per_layer, 
                        n_col=rhs[0].shape[1],
                        activation_init=activation_init,
                        activation_hidden=activation_hidden,
                        activation_last=activation_last,
                        **kwargs):
                super(PINN_tmp, self).__init__(
                    Dtype=Dtype, 
                    hidden_layers=hidden_layers,
                    neurons_per_layer=neurons_per_layer,
                    n_col=n_col,
                    activation_init=activation_init,
                    activation_hidden=activation_hidden,
                    activation_last=activation_last,
                    **kwargs
                )
                
        with tf.device(device):
            model = PINN_tmp(hidden_layers=hidden_layers, 
                        neurons_per_layer=neurons_per_layer,
                        activation_init=activation_init,
                        activation_hidden=activation_hidden,
                        activation_last=activation_last,
                        n_col=rhs[0].shape[1],
                        line_num_points_shaft=line_num_points_shaft,
                        line_num_points_inter_shaft=line_num_points_inter_shaft,
                        spline_smooth_shaft=spline_smooth_shaft,)
 
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 5e-4])
        optimizer = tf.optimizers.Adam(learning_rate=lr) 
        if new_model:
            print(f"I'm starting a new model")
            print('--------------------------------')
            print(f"Model Type Gen : {model_type}")
            print(f"Model Type     : {model_sufix}")
            print(f"Model Dir.     : {model_dir}")
            print(f"Model Portion  : {pre_portion}")
            print(f"Data Dir.      : {path_train['data_spine_path']}") 
            print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
            print(f"Feat. list     : {[mm for _,mm in list_features]}") 
            print(f"Feature Size   : {curv[0].shape[1]}")
            print(f"Target Size    : {rhs[0].shape[1]}")
            print(f"Dend Trainer size    : {[len(fv) for fv in curv_train]}")
            print(f"Hidden Layers        : {hidden_layers}")
            print(f"neurons_per_layer    : {neurons_per_layer}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            
        print(f"Using device: {device}----------------------------------------------------------------")
        aka_train_ = aka_train()


        indd=[[np.where(rhs[i][:,label]==1)[0] for label in range(rhs[i].shape[1])] for i in lss ]
        fun = LOSS(rhs=rhs_train , 
                   curv=curv_train , 
                   weight=weight,#tf.cast(weight,dtype=DTYPE), 
                   loss_mode=loss_mode,
                   adj=adj_train,
                   dtype=DTYPE,
                   dend=dend_train, 
                   rl_par= rl_par, 
                   dnn_par= dnn_par,
                   )
        
        tf.random.set_seed(0)
        iou_tmp=[0,0,0]
        index_save=[]
        pbar = tqdm(range(itime), desc=f"Loss: {loss_tmp:.6f}| IoU: N/A")  
        for i in pbar: 
            loss = aka_train_.train_PINN(optimizer, fun, model)
            iou=Get_iou(model, curv, lab=indd,
                        adj=adj, 
                   dend=dend,)
            for ii in range(rhs[0].shape[1]):
                iou_save[ii].append(iou[ii]) 
                np.savetxt(iou_save_dir[ii] , np.array(iou_save[ii]), fmt='%f') 
            loss_save.append(loss.numpy())
            np.savetxt(loss_save_dir, np.array(loss_save), fmt='%f') 
            if loss.numpy() < loss_tmp:
                loss_tmp=loss.numpy()
                model.save(model_dir) 
                iou_tmp=iou  
                index_save.append(i)
                np.savetxt(index_save_dir, np.array(index_save), fmt='%d')  
            if pre_portion=='head_neck':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: {min(iou_tmp[0]):.4f} IoU nk: {min(iou_tmp[1]):.4f} IoU hd: {min(iou_tmp[2]):.4f}")
            elif pre_portion=='spine':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: {min(iou_tmp[0]):.4f} IoU sp: {min(iou_tmp[1]):.4f}")
            else:
                pbar.set_description(f"Loss: {loss_tmp:.6f} ")
  


  



    def train_model_spine_ML(self,
                            path_train=None,
                            get_training=True, 
                            pre_portion=None,
                            full_dend=True, 
                            hidden_layers=None, 
                            neurons_per_layer=None, 
                            activation_init=None,
                            activation_hidden=None,
                            activation_last=None, 
                            curv=None,
                            rhs=None,
                            weight=None,
                            indices=None,
                            DTYPE=None, 
                            model_sufix=None,
                            file_path_model_data=None,
                            line_num_points_shaft=None,
                            line_num_points_inter_shaft=None, 
                            spline_smooth_shaft=None,
                            data_studied=None,
                            vv_cts=None,
                            new_model=True, 
                            itime = 10000, 
                            itime_div=1,
                            loss_save_dir=None,
                            iou_save_dir=None,   
                            index_save_dir=None,  
                            model_dir=None, 
                            loss_mode="bce",
                            ls=[2,3,4,5],
                            disp_infos=None, 
                            weight_positive= .5,
                            list_features=None,
                            base_features_list=None,
                            train_spines=False,
                            dest_path='dest_spine_path',
                            model_type=None,
                            num_sub_nodes=None,
                            rl_par= 0.5, 
                            dnn_par= 0.5,
                            thre_target_number_of_triangles=None,
                            voxel_resolution=None,
                            l1_values = [0,   1e-6,   1e-4, 1e-2],
                            l2_values = [0,   1e-6,   1e-4, 1e-2],
                            classifiers=None,
                        ): 

        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos 
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        DTYPE = DTYPE or self.DTYPE
        vv_cts = vv_cts or self.vv_cts 
        model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type)
        line_num_points_shaft=line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft=line_num_points_inter_shaft or self.line_num_points_inter_shaft
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied 
        spline_smooth_shaft=self.spline_smooth_shaft
 
 
        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last
 
        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou'] 
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']   
 
        if model_type=='gcn':
            from dend_fun_0.help_gcn_one_hot import LOSS,PINN,aka_train,Get_iou 
            adj_tf=True
        elif model_type.startswith('pinn'):
            from dend_fun_0.help_pinn_one_hot import LOSS,PINN,aka_train,Get_iou 
            adj_tf=False
        elif model_type.startswith('rpinn'):
            from dend_fun_0.help_pinn_rein_one_hot  import LOSS,PINN,aka_train,Get_iou 
            adj_tf=False

 
        curv,rhs,adj,dend =self.get_train_input(   
                            path_train=path_train,  
                            pre_portion=pre_portion, 
                            line_num_points_shaft=line_num_points_shaft,
                            line_num_points_inter_shaft=line_num_points_inter_shaft, 
                            model_sufix=model_sufix, 
                            list_features=list_features,
                            base_features_list=base_features_list, 
                            model_type=model_type,
                            num_sub_nodes=num_sub_nodes, 
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            ) 
        curv_train = [curv[i] for i in ls]
        rhs_train = [rhs[i] for i in ls]  
        lss=np.arange(len(curv),dtype=int) 
        adj_train =[]
        if model_type=='gcn':
            adj_train =[adj[i] for i in ls]
  
        if new_model:
            print(f"I'm starting a new model")
            print('--------------------------------')
            print(f"Model Type Gen : {model_type}")
            print(f"Model Type     : {model_sufix}")
            print(f"Model Dir.     : {model_dir}")
            print(f"Model Portion  : {pre_portion}")
            print(f"Data Dir.      : {path_train['data_spine_path']}") 
            print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
            print(f"Feat. list     : {[mm for _,mm in list_features]}") 
            print(f"Feature Size   : {curv[0].shape[1]}")
            print(f"Target Size    : {rhs[0].shape[1]}")
            print(f"Dend Trainer size    : {[len(fv) for fv in curv_train]}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            

        from dend_fun_0.aka_ML_finder import aka_classification

        print(f"Using device: {device}")
        aka_train_ = aka_classification()


        indd=[[np.where(rhs[i][:,label]==1)[0] for label in range(rhs[i].shape[1])] for i in lss ]

        best_algorithms = {}
        metric_algorithms = {}
        if classifiers is None:
            clf_algorithms = aka_train_.classifiers
        else:
            clf_algorithms = classifiers
        clf_algorithms=aka_train_.train(X_train=curv_train[0], 
                                        y_train=rhs_train[0].argmax(axis=1),  
                                        clf_algorithms =clf_algorithms,
                                        )
        
        clf_best, df_metric_algorithms =aka_train_.train_and_find_best_classifier(
                                                                                X_test=curv[3], 
                                                                                y_test=rhs[3].argmax(axis=1),
                                                                                best_algorithms = best_algorithms,
                                                                                metric_algorithms =metric_algorithms,
                                                                                clf_algorithms =clf_algorithms,
                                                                                ) 
        df_metric_algorithms.to_csv(self.df_metric_algorithms_dir )

        with open(model_dir, 'wb') as f:
            pickle.dump(clf_best, f)
         
     

 
    def get_shaft_pred(self, 
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
                        ): 
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
        model_dir = model_dir or  model_dirs['model']   

        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name]) 
        print('-------------',n_col,model_dir)


        if (model_type in ['cML']) or model_type.startswith('ML'):
            with open(model_dir, 'rb') as f:
                model = pickle.load(f)  

        else:
            if model_type.startswith('gcn'):
                from dend_fun_0.help_gcn_one_hot import PINN ,run_model_on_graph
                adj_tf=True
            elif model_type.startswith('pinn'):
                from dend_fun_0.help_pinn_one_hot import PINN 
                adj_tf=False 
            elif model_type.startswith('rpinn'):
                from dend_fun_0.help_pinn_rein_one_hot import LOSS,PINN,aka_train,Get_iou 
                adj_tf=False


            class PINN_tmp(PINN):
                def __init__(self, Dtype=DTYPE, 
                            hidden_layers=hidden_layers, 
                            neurons_per_layer=neurons_per_layer, 
                            n_col=n_col,
                            activation_init=activation_init,
                            activation_hidden=activation_hidden,
                            activation_last=activation_last,
                            **kwargs):
                    super(PINN_tmp, self).__init__(
                        Dtype=Dtype, 
                        hidden_layers=hidden_layers,
                        neurons_per_layer=neurons_per_layer,
                        n_col=n_col,
                        activation_init=activation_init,
                        activation_hidden=activation_hidden,
                        activation_last=activation_last,
                        **kwargs
                    ) 

            model = load_model(model_dir, custom_objects={'PINN_tmp': PINN_tmp}) 
    
  


        print(f'get_shaft_pred started')
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
                                model_type=model_type, )  
            dend_name=self.dend_name  
            spine_portion_path=self.path_file[path_train['data_spine_path']] 
            shaft_portion_path=self.path_file[path_train['data_shaft_path']] 
            pid = pinn_data(file_path=self.file_path,
                            file_path_feat=self.file_path_feat,
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
            feat_paths=[]
            for path_inten,name_inten in list_features:
                inten_path=os.path.join( self.file_path_feat ,name_inten)
                if os.path.exists(inten_path):
                    feat_paths.append(inten_path)
                    print('path exists----->>>',inten_path)
                    con+=1
                else:
                    print('path doenst exists =----->>>',inten_path)
                    mmm=np.zeros(pid.vertices_00.shape[0]) 
            vvb=pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,)    
            base_features=np.hstack(vvb)
 

            if model_type.startswith(('cML', 'ML')): 
                    rhs0 = model.predict_proba( base_features ) 

            else:   
                model = load_model(model_dir, custom_objects={'PINN_tmp': PINN_tmp}) 
                if model_type.startswith(('pinn', 'rpinn')):
                    rhs0 = model(tf.cast( base_features, dtype=DTYPE)) 
                elif model_type.startswith(('gcn')): 
                    adj=hff.adjoint(vertices=pid.dend.vertices,faces=pid.dend.faces,num_sub_nodes=None)
                    rhs0=run_model_on_graph(model=model, X=base_features, adj=adj[0]) 
                    
                rhs0 = rhs0.numpy() 


 
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







    def model_shap(self,model=None,
                    dend_names=None,
                    path_train=None,
                    get_training=True, 
                    pre_portion=None,
                    full_dend=True, 
                    hidden_layers=None, 
                    neurons_per_layer=None, 
                    activation_init=None,
                    activation_hidden=None,
                    activation_last=None, 
                    curv=None,
                    rhs=None,
                    weight=None,
                    indices=None,
                    DTYPE=None, 
                    model_sufix=None,
                    file_path_model_data=None,
                    line_num_points_shaft=None,
                    line_num_points_inter_shaft=None, 
                    spline_smooth_shaft=None,
                    data_studied=None,
                    vv_cts=None,
                    new_model=True, 
                    itime = 10000, 
                    loss_save_dir=None,
                    iou_save_dir=None,   
                    index_save_dir=None,  
                    model_dir=None, 
                    loss_mode="bce",
                    ls=[2,3,4,5],
                    disp_infos=None, 
                    weight_positive= .5,
                    list_features=None,
                    base_features_list=None,
                    train_spines=False,
                    dest_path='dest_spine_path',
                    dend_names_ls=[0,1],
                    n_shap=150,
                    model_type=None,
                    thre_target_number_of_triangles=None,
                    voxel_resolution=None,
                    dict_mesh_to_skeleton_finder_mesh=None,
                        ): 
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        model_type=model_type or self.model_type
        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features=list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos 
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        DTYPE = DTYPE or self.DTYPE
        vv_cts = vv_cts or self.vv_cts 
        model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type)
        line_num_points_shaft=line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft=line_num_points_inter_shaft or self.line_num_points_inter_shaft
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied 
        spline_smooth_shaft=self.spline_smooth_shaft
 
 
        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last
 
        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou'] 
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']   

        feat_name=[] 
        for  mm in base_features_list:
            feat_name.append(mm)
        for _,name_inten in list_features: 
            feat_name.append(name_inten.removesuffix('.txt') )


 
 
                
        rhs_name='rhs_name'
        n_col=  len(self.model_dir_path[pre_portion][rhs_name]) 
        print('-------------',n_col,model_dir)

        if (model_type in ['cML']) or model_type.startswith('ML'):
            with open(model_dir, 'rb') as f:
                model = pickle.load(f)  

        else:
            if model_type=='gcn':
                from dend_fun_0.help_gcn_one_hot import PINN ,run_model_on_graph
                adj_tf=True 
            elif model_type.startswith('pinn'):
                from dend_fun_0.help_pinn_one_hot import PINN 
                adj_tf=False 
            elif model_type.startswith(('pinn','rpinn')):
                from dend_fun_0.help_pinn_rein_one_hot import LOSS,PINN,aka_train,Get_iou 
                adj_tf=False


            class PINN_tmp(PINN):
                def __init__(self, Dtype=DTYPE, 
                            hidden_layers=hidden_layers, 
                            neurons_per_layer=neurons_per_layer, 
                            n_col=n_col,
                            activation_init=activation_init,
                            activation_hidden=activation_hidden,
                            activation_last=activation_last,
                            **kwargs):
                    super(PINN_tmp, self).__init__(
                        Dtype=Dtype, 
                        hidden_layers=hidden_layers,
                        neurons_per_layer=neurons_per_layer,
                        n_col=n_col,
                        activation_init=activation_init,
                        activation_hidden=activation_hidden,
                        activation_last=activation_last,
                        **kwargs
                    ) 

            if model is None:
                model = load_model(model_dir, custom_objects={'PINN_tmp': PINN_tmp}) 
    


        print(f'SHAP started')
        print('--------------------------------')
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
        base_features=[]
        dend_names = dend_names or self.dend_names 
        for  index in dend_names_ls:
            self.get_dend_name(data_studied=data_studied,index=index )   
            spine_portion_path=self.path_file[path_train['data_spine_path']] 
            shaft_portion_path=self.path_file[path_train['data_shaft_path']] 
            pid = pinn_data(file_path=self.file_path,
                            file_path_feat=self.file_path_feat,
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
            con=0
            feat_paths=[]
            for path_inten,name_inten in list_features:
                feat_paths.append(os.path.join( self.path_file[ path_inten] ,name_inten) )
                if os.path.exists(os.path.join( self.path_file[ path_inten] ,name_inten)):
                    con+=1
                else:
                    print('path doenst exists =----->>>',os.path.join( self.path_file[ path_inten] ,name_inten))
                    mmm=np.zeros(pid.vertices_00.shape[0])
                    np.savetxt(os.path.join( self.path_file[ path_inten] ,name_inten),mmm, fmt='%f')
            vvb=pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,)    
            base_features.append(np.hstack(vvb))
 
 
        print([vd.shape for vd in base_features])
        n_shap = n_shap or min([150, base_features[1].shape[0]])
        niuu = min([base_features[0].shape[0] // 100, base_features[0].shape[0]]) 
        explain = shap.Explainer(model.predict, base_features[0][:niuu, :]) 
        shap_values = explain(base_features[1][:n_shap, :]) 
        raw_values = shap_values.values 
        mean_shap = np.abs(raw_values).mean(axis=0)
        mean_shap = mean_shap if mean_shap.ndim == 2 else mean_shap.reshape(-1, 1) 
        feat = {'Feature': feat_name}
        for idx in range(mean_shap.shape[1]):
            feat[f'Mean SHAP Values {idx+1}'] = mean_shap[:, idx]

        shap_df = pd.DataFrame(feat).sort_values(by='Mean SHAP Values 1', ascending=False)
        # shap_df.to_csv(self.shap_dir, index=False)


        head_neck_path = 'dest_shaft_path'
        path=path_train[head_neck_path]
        spine_path_save=     self.path_file[f'result_{path}']   
        shap_df.to_csv(os.path.join(spine_path_save,'shap.csv') , index=False)



    def clean_path_dir(self,   
                        path_head_clean,
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None,
                    ):   
        print(' path_head_clean ---- started')
        print('==========================================================')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index )  
            self.clean_dend_name(path_head_clean=path_head_clean)


   
  
 
    def get_head_neck_segss(self,    
                            metrics=None,
                            seg_dend='full',
                            path_train=None ,
                            get_refine_=False,
                            data_studied=None, 
                            disp_infos=None, 
                            dend_names=None,
                            dend_namess=None,
                            spine_shaft_txt=True,
                            size_threshold=None,
                            line_num_points_shaft=None,
                            line_num_points_inter_shaft=None,
                            spline_smooth_shaft=None ,  
                            zoom_thre=15,
                            skip_first_n=1,
                            skip_end_n=52,
                            subdivision_thre=3, 
                            subsample_thre=.02,
                            f=0.99,
                            N=10,
                            num_chunks=100, 
                            line_num_points=300,
                            line_num_points_inter=700,
                            spline_smooth=1.,
                            num_points=50,
                            ctl_run_thre=1,
                            end_thre=40,
                            head_neck_path= 'dest_spine_path',
                            pre_portion=None,
                            dict_mesh_to_skeleton_finder_mesh=None,
                        ):   
        time_start = time.time()
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        path_train=path_train or self.path_train
        pre_portion=pre_portion or self.pre_portion 
        size_threshold=size_threshold or self.size_threshold
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        line_num_points_shaft = line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft = line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft = spline_smooth_shaft or self.spline_smooth_shaft
        self.metrics=metrics if metrics is not None else self.metrics
        print('Head Neck Prediction started')
        print('--------------------------------') 
        print(f"learning length : {zoom_thre}") 
        print(f"skip_first_n    : {skip_first_n}") 
        print(f"skip_end_n      : {skip_end_n}") 
        print(f"subdivision_thre: {subdivision_thre}")
        print(f"destination     : {path_train['dest_spine_path']}")
        print(f"data path       : {path_train['data_spine_path']}")
        print(f"seg. org. dend  : {seg_dend}") 
        print('--------------------------------')  


        metrics_dats=[]
        metrics_name=[]
        metrics_dat={}
        metrics_sp=[]
        gnam=get_name()
        metricss=gnam.metrics 

        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index )
            dend_name=self.dend_name               
            gnam=get_name()
            metricss=gnam.metrics 
            metrics_name=[]
            pid=pinn_data(file_path=self.file_path,
                          file_path_feat=self.file_path_feat,
                            shaft_path=     self.path_file[path_train['dest_shaft_path']] ,
                            spine_path=     self.path_file[path_train['dest_spine_path']] , 
                            shaft_path_pre= self.path_file[path_train['data_shaft_path']] ,
                            spine_path_pre= self.path_file[path_train['data_spine_path']] , 
                            dend_path_original_m=self.dend_path_original_m, 
                            dend_first_name=self.dend_namess[index][1],  
                            path_file=self.path_file,
                            line_num_points_shaft=line_num_points_shaft,
                            line_num_points_inter_shaft=line_num_points_inter_shaft,
                            spline_smooth_shaft=spline_smooth_shaft, 
                                    ) 
            pid.get_head_neck_segss(  
                                pre_portion=pre_portion,
                                metrics=metricss,
                                seg_dend=seg_dend,
                                spine_shaft_txt=spine_shaft_txt, 
                                path_train=path_train,
                                get_refine_=get_refine_,
                                skip_first_n=skip_first_n,
                                skip_end_n=skip_end_n,
                                zoom_thre=zoom_thre,
                                subdivision_thre=subdivision_thre,
                                subsample_thre=subsample_thre,
                                f=f,
                                N=N,
                                num_chunks=num_chunks,
                                num_points=num_points, 
                                line_num_points= line_num_points,
                                line_num_points_inter= line_num_points_inter,
                                spline_smooth= spline_smooth,
                                ctl_run_thre=ctl_run_thre,  
                                size_threshold=size_threshold,
                                end_thre=end_thre,
                                head_neck_path=head_neck_path,
                                dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                        ) 
            for nam in self.metrics_keys:
                if len(metricss[nam].keys())>0:
                    metrics_name.append(nam)
                    if nam not in metrics_dat:
                        metrics_dat[nam]=[]
                    metrics_dat[nam].extend((list(metricss[nam].values())))
                    if nam==metrics_name[0]:
                        metrics_sp.extend(metricss[nam].keys())
            # print( (metrics_name),) 
            # print([len(metrics_dat[nam]) for nam in metrics_name],)
            metricss={}
        if len(metrics_dat)>0:
            metrics_dats=[metrics_dat[nam] for nam in metrics_dat.keys()]
            metrics_dats=np.array(metrics_dats).T 
            print('IM HEAD NECK',metrics_dats.shape,head_neck_path,path_train[head_neck_path])
            metrics_name=np.array(list(metrics_dat.keys()))
            metrics_sp=np.array(metrics_sp)
            path=path_train[head_neck_path] 
            spine_path_save=     self.path_file[f'result_{path}'] 
            with open(os.path.join( spine_path_save,'metrics.csv'), 'w') as f: 
                f.write(',' + ','.join(metrics_name) + '\n') 
                for i, row_name in enumerate(metrics_sp):
                    row_data = ','.join(map(str, metrics_dats[i]))
                    f.write(f'{row_name},{row_data}\n')


        metrics_dats=[]




        mytime0 = time.time() - time_start 
        hours, rem = divmod(mytime0, 3600)
        minutes, seconds = divmod(rem, 60)
        if disp_infos:
            print(f'Head Neck Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
        print('Head Neck Prediction completed')

 
 

    def get_iou(self,   
                    path_train=None ,
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    model_type=None, 
                    zoom_thre=10,
                    iou_thre=0.002):   
        print('IOU started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos    
        data_studied = data_studied or self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names or self.dend_names 
        model_type=model_type or self.model_type 
        for head_neck_path in self.path_display:
            iou_dict={}
            
            for  index,dend_name in enumerate(dend_names):
                self.get_dend_name(data_studied=data_studied,index=index )
                spine_path=self.path_file[self.path_train[head_neck_path]] 
                for true_path,true_key in self.dend_path_original_mm['keys'].items() : 
                    if true_path not in iou_dict: 
                        iou_dict[true_path]={}
                    iou_tr=iou_train(
                                    path_true=self.path_file[true_path],
                                    path_appr=self.path_file[self.path_train[head_neck_path]],
                                    iou_thre=iou_thre,
                                    iou_dict=iou_dict[true_path],
                                    dend_first_name=self.dend_first_name,
                                    file_path=self.file_path,
                                    path_result=self.path_file[f'result_{path_train[head_neck_path]}'],)
                    iou_tr.get_mapping(save=True)
                    iou_tr.get_iou_save(save=True) 
                    if true_key=='true_0':
                        iou_tr.get_iou_match(iou_thre=0.2)
                    scatter_graph=get_iou_graph(self,spine_path, save_data=False)
                    with open(os.path.join(spine_path,f'plot_iou_graph_{true_key}.pkl'), 'wb') as f:
                        pickle.dump(scatter_graph, f)  
            path=path_train[head_neck_path] 
            spine_path_save=     self.path_file[f'result_{path}']
            col_name=['id','id_true','iou_single','iou_union']
            for true_path,true_key in self.dend_path_original_mm['keys'].items():
                with open(os.path.join(spine_path_save,f'iou_{true_key}.csv'), 'w') as f: 
                    f.write(',' + ','.join(col_name) + '\n') 
                    for i, row_name in enumerate(list(iou_dict[true_path].keys())):
                        row_data = ','.join(map(str, iou_dict[true_path][row_name])) 
                        f.write(f'{row_name} ,{row_data}\n')
            iou_dict={}
 



    def get_graph_center(self,   
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None,
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
                    ):   
        print(' get_graph_center ---- started')
        print('==========================================================')
        time_start = time.time()
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 
        path_train=path_train or self.path_train 
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index )  
            dend_name=self.dend_name 
            print(dend_name)   
            spine_path=self.path_file[path_train['dest_spine_path']]
            shaft_path=self.path_file[path_train['dest_shaft_path']]
            pid=pinn_data(file_path=self.file_path,
                          file_path_feat=self.file_path_feat,
                            shaft_path=shaft_path,
                            spine_path=spine_path,
                            shaft_path_pre=self.path_file[path_train['dest_shaft_path']],
                            dend_path_original_m=self.dend_path_original_m, 
                            dend_first_name=self.dend_namess[index][1],  
                            path_train=path_train, 
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                            ) 
            pid.get_dend_data()  
            pid.get_central_data()  
            for center_path in self.path_display:
                giou=graph_iou( pid,self.path_file,path_train,center_path=center_path)
                giou.scatter_center() 



    def get_cylinder_heatmap(self,   
                    data_studied=None, 
                    disp_infos=None, 
                    dend_names=None,
                    dend_namess=None,
                    path_train=None,
                    ):   
        print(' get_cylinder_heatmap ---- started')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        dend_namess = dend_namess or self.dend_namess
        dend_names = dend_names if dend_names is not None else self.dend_names 
        for  index,dend_name in enumerate(dend_names):
            self.get_dend_name(data_studied=data_studied,index=index )  
            dend_name=self.dend_name 
            print(dend_name)   
            spine_path=self.path_file[path_train['dest_spine_path']]
            shaft_path=self.path_file[path_train['dest_shaft_path']]
            pid=pinn_data(file_path=self.file_path,
                            shaft_path=shaft_path,
                            spine_path=spine_path,
                            dend_path_original_m=self.dend_path_original_m, 
                            dend_first_name=self.dend_namess[index][1], 
                          file_path_feat=self.file_path_feat,
                                    ) 
            # pid.get_dend_data()    
            for center_path in self.path_display:
                gcyl=graph_cylinder_heatmap( pid,self.path_file,path_train,center_path=center_path, file_path_feat=self.file_path_feat)
                gcyl.get_cylinder_heatmap() 

 


    def get_dash_pages(self,  
                        data_studied=None, 
                        file_path_org=None, 
                        disp_infos=None,
                        dash_pages_path=None,
                        path_train=None,
                        model_type=None,
                        obj_org_path_dict=None,
                        model_sufix_dic=None,): 
        disp_infos = disp_infos or self.disp_infos 
        data_studied = data_studied or self.data_studied
        file_path_org = file_path_org or self.file_path_org  
        dash_pages_path = dash_pages_path or self.dash_pages_path
        path_train=path_train or self.path_train
        model_type=model_type or self.model_type
        obj_org_path_dict=obj_org_path_dict or self.obj_org_path_dict
        model_sufix_dic=model_sufix_dic or self.model_sufix_dic


        dend_names=self.dend_names
        dend_namess=self.dend_namess
        dend_path_inits=self.dend_path_inits
        model_sufix=self.model_sufix
  
        print('get_dash_pages') 
        print('--------------------------------')
        print(f"Model Type     : {model_sufix}") 
        print(f"Model Portion  : {self.pre_portion}") 
        print(f"Destination    : {path_train['dest_spine_path']}") 
        print('--------------------------------')
        for index,dend_name in enumerate(dend_names):   
            self.get_dash_pages_name(index,data_studied)
            dash_pages_name=self.dash_pages_name
            dend_name = dend_names[index] 
            dend_path_init=dend_path_inits[index]
            dend_namessi=dend_namess[index]
            dend_path_init=dend_path_inits[index]
            user_inputt=f'{index}' 
            if data_studied =='train':
                get_text_dash_train(user_inputt,
                                    file_path_org,
                                    dend_path_init,   
                                    dend_name, 
                                    dend_namessi, 
                                    data_studied, 
                                    model_sufix,
                                    dash_pages_name,
                                    disp_infos,
                                    path_train=path_train,
                                    index=index,)
            else:
                get_text_dash_test(user_inputt,
                                   file_path_org,
                                   dend_path_init,   
                                   dend_name, 
                                   dend_namessi, 
                                   data_studied, 
                                   model_sufix,
                                   dash_pages_name,
                                   disp_infos,
                                   path_train=path_train,
                                    path_file=self.path_file,
                                    pinn_dir_data=self.pinn_dir_data,
                                    dend_data=self.dend_data,
                                    index=index,
                                    model_type=model_type,
                                    obj_org_path_dict=obj_org_path_dict,
                                    model_sufix_dic=model_sufix_dic,
                                   )
            # id_name_end=self.dend_path_inits[index]#safe_id(self.dend_path_inits[index])
            # dash_pages_path=os.path.join('/',self.dash_pages_path,f'dsa_{self.model_type}_{self.model_sufix}_{id_name_end}.py')
            # dash_pages_path=f'{dash_pages_path}'
            # page_dir_txt=os.path.join(self.data_studied,self.model_type,self.model_sufix,id_name_end)
            # page_dir_txt=page_dir_txt.replace('_','-').replace('\\','/').lower()
            # get_text_dash_all(  
            #                     page_name=f'{id_name_end}',
            #                     page_dir_txt=page_dir_txt,
            #                     dash_pages_path=dash_pages_path,
            #                     disp_infos=False,
            #                     path_train=None, 
            #                     path_file=None, 
            #                     pinn_dir_data=None,
            #                     dend_data=None,
            #                     index=None,
            #                     )

            page_name=self.model_type.upper() #safe_id(self.dend_path_inits[index])
            dash_pages_path=os.path.join('/',self.dash_pages_path,self.data_studied, f'{self.model_type}_data.py')
            dash_pages_path=f'{dash_pages_path}'
            page_dir_txt=os.path.join(self.data_studied,self.model_type  ) 
            page_dir_txt=os.path.join(page_dir_txt,f'{self.model_type}-data-' ).replace('_','-').replace('\\','/').lower()  
            page_dir_txt=page_dir_txt
            get_text_dash_dnn(  
                                page_name=f'{page_name}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                dend_data=None,
                                index=None,
                                page_view='results', 
                                )


            id_name_end=self.dend_path_inits[index]
            page_name=self.model_sufix
            page_name=model_sufix_dic[self.model_sufix]
            dash_pages_path=os.path.join('/',self.dash_pages_path,self.data_studied,self.model_type ,f'{self.model_type}_data_{self.model_sufix}.py')
            dash_pages_path=f'{dash_pages_path}' 
            page_dir_txt=os.path.join(self.data_studied,self.model_type ,self.model_sufix ,f'{self.model_type}-data-{self.model_sufix}'  ).replace('_','-').replace('\\','/').lower() 
            get_text_dash_dnn(  
                                page_name=f'{page_name}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                dend_data=None,
                                index=None,
                                page_view='results', 
                                )

            id_name_end=self.dend_path_inits[index]
            dash_pages_path=os.path.join('/',self.dash_pages_path,self.data_studied,self.model_type ,self.model_sufix ,   f'{self.model_type}_data_{self.model_sufix}.py')
            dash_pages_path=f'{dash_pages_path}' 
            page_dir_txt=os.path.join(self.data_studied,self.model_type ,self.model_sufix ,id_name_end ).replace('_','-').replace('\\','/').lower() 
            get_text_dash_dnn(  
                                page_name=f'{id_name_end}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                dend_data=None,
                                index=None,
                                page_view='visualization', 
                                )


