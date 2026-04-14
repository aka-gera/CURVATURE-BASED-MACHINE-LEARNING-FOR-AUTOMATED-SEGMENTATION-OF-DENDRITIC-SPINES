











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
import trimesh


import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"  
from tqdm import tqdm 

from dend_fun_0.help_save_iou import iou_train 
from dend_fun_0.help_pinn_data_fun import pinn_data 
from dend_fun_2.help_pinn_data_fun import pinn_data 
import dend_fun_0.help_funn as hff   


from dend_fun_0.get_path import assign_if_none,get_name,get_param,get_files
 


class train_test_tf(get_files,get_name): 
    def __init__(self, **kwargs):
        get_name.__init__(self) 
        get_files.__init__(self,**kwargs)

    def get_train_input_ML(self,  
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
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
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

            curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
            dend.append(pid.dend)     
            if tf_train:
                rhs.append(pid.get_pinn_rhs(pre_portion=pre_portion) )

        return curv, rhs ,adj,dend











    def get_train_input_cml(self,  
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
                        model_init=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        model_init = model_init if model_init is not None else self.model_init
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

        indexx={dend_name:index for index, dend_name  in enumerate(self.dend_names)} 
        for entry_name in entry_names:
            for dend_name in dend_names: 
                index=indexx[dend_name]
                self.get_dend_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name) 
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
                                kmean_n_run=kmean_n_run,
                                kmean_max_iter=kmean_max_iter,
                                param_dic=param_dic,
                                    )
                print('path doesnt exists ===----->>>[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]',base_features_list)
                file_path_feat = self.file_path_feat if entry_name is None else self.file_path_feat_entry
                file_path = self.file_path if entry_name is None else self.file_path_entry
                pid.save_pinn_data(file_path_feat=file_path_feat,
                                file_path=file_path,
                                )    
                pid.get_dend_data(file_path_feat=file_path_feat,
                                file_path=file_path,)
                feat_paths=[]  
                for path_inten,name_inten in list_features: 
                    if model_init is not None: 
                        mode_dnn_1=self.model_sufix_dic['model_sufix_inverse'][model_init]
                        path_dir=self.model_sufix_dic['path_dir']
                        path_shaft_dir=f'{self.model_type}_{mode_dnn_1}_{path_dir}'
                        pathini=self.path_file[path_shaft_dir]
                    else:
                        pathini=file_path_feat

                    pathini=  self.path_file[path_train['data_shaft_path']]   

                    pathh=os.path.join( pathini ,name_inten)
                    if os.path.exists(pathh):
                        feat_paths.append(pathh)
                        print('train data path ---->>',pathini,pathh) 
                        print(np.loadtxt(pathh))
                    else:
                        print('path doesnt exists ===----->>>',pathh)
                dend.append(pid.dend)     
                if tf_train:
                    rhs.append( 
                        pid.get_pinn_rhs(pre_portion=pre_portion,
                                        file_path_feat=file_path_feat,
                                        file_path=file_path,
                                        dend_path_true_final=self.dend_path_true_final,
                                        ),  
                            )
                    pathh=os.path.join( self.dend_path_true_final ,'intensity_shaft_spine.txt')
                    mask=np.loadtxt(pathh,dtype=int)
                    unique=np.sort(np.unique(mask))
                    labels = {v: np.argwhere(mask == v) for v in unique}
                    counts = {v: labels[v].shape[0] for v in labels}
                    total = sum(counts.values()) 

                    w_prime = np.array([max(np.log(2 * total / counts[k]), 1) for k in unique ])
        
                    weight.append(w_prime / np.sum(w_prime))






                if len(list(list_features)+list(base_features_list))==0:
                    curv.append(pid.dend.vertices)  
                    continue
                    
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,
                                                            base_features_list=base_features_list,
                                                            file_path_feat=file_path_feat,
                                                            file_path=file_path,))) 

        return curv, rhs ,adj,dend,weight
 





    def train_model_spine_cml(self,
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
                            auc_save_dir=None,   
                            dice_save_dir=None,
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
                            dict_mesh_to_skeleton_finder_mesh=None,
                            entry_names=[],
                            kmean_n_run=None,
                            kmean_max_iter=None,
                            param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter

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
        auc_save_dir = auc_save_dir or model_dirs['auc']
        dice_save_dir = dice_save_dir or model_dirs['dice']
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']   
    
   
        curv,rhs,adj,dend,weights =self.get_train_input_cml(   
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
                                tf_train=True,
                                entry_names=entry_names, 
                                kmean_n_run=kmean_n_run,
                                kmean_max_iter=kmean_max_iter,
                                param_dic=param_dic,
                                ) 
        curv_train = [curv[i] for i in ls]
        rhs_train = [rhs[i] for i in ls]  
        curv_train=[np.vstack(curv_train)]
        rhs_train=[np.vstack(rhs_train)]
        lss=np.arange(len(curv),dtype=int)  
  
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
        print(rhs_train[0].shape)
        clf_algorithms=aka_train_.train(X_train=curv_train[0], 
                                        # y_train=rhs_train[0],  
                                        y_train=rhs_train[0].argmax(axis=1), 
                                        clf_algorithms =clf_algorithms,
                                        )
        
        clf_best, df_metric_algorithms =aka_train_.train_and_find_best_classifier(
                                                                                X_test=curv[3], 
                                                                                y_test=rhs[3].argmax(axis=1),
                                                                                # y_test=rhs[3],
                                                                                best_algorithms = best_algorithms,
                                                                                metric_algorithms =metric_algorithms,
                                                                                clf_algorithms =clf_algorithms,
                                                                                ) 
        df_metric_algorithms.to_csv(self.df_metric_algorithms_dir )

        with open(model_dir, 'wb') as f:
            pickle.dump(clf_best, f)
         
     
    def get_shaft_pred_cml(self, 
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
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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


        # if (model_type in ['cML']) or model_type.startswith('ML'):
        with open(model_dir, 'rb') as f:
            model = pickle.load(f)  

 
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
        for  index,dend_name  in enumerate( dend_names):
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )  
            key,tyy=f'{self.model_type}_{self.model_sufix}_{self.path_dir}', self.intensity_spines_logit[0]  
            path_ex = os.path.join(self.path_file_sub[tyy][key]) 
            if os.path.exists(path_ex) and not param_dic['tf_restart']['get_shaft_pred']: 
                continue
            pid.save_pinn_data() 
            pid.get_dend_data() 

            curv,_,_,_,_=self.get_train_input_cml(   
                                    dend_names=[dend_name],
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
                                    tf_train=False,
                                    kmean_n_run=kmean_n_run,
                                    kmean_max_iter=kmean_max_iter,
                                    param_dic=param_dic,
                                    )   
            rhs0 = model.predict_proba( curv[0] )
            # rhs0=model(curv[0]).numpy()   
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk], fmt='%f') 
 






            vert=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_0)) 
            vert_old=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_old)) 
            from scipy.spatial import KDTree 
            idx = KDTree(vert).query(vert_old)[1].flatten() 


            self.get_dend_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type,
                                ** param_dic['data']['get_dend_name'])    

            # cxc=param_dic['data']['get_dend_name']['dict_dend_path']
            # key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}' # if cxc in [None,'current'] else f'{cxc}_{self.model_type}_{self.model_sufix}_{self.path_dir}'
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk][idx], fmt='%f') 












            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 








    def get_train_input_gcn(self,  
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
                        model_init=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        model_init = model_init if model_init is not None else self.model_init
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

        indexx={dend_name:index for index, dend_name  in enumerate(self.dend_names)} 
        for entry_name in entry_names:
            for dend_name in dend_names: 
                index=indexx[dend_name]
                self.get_dend_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name) 
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
                                kmean_n_run=kmean_n_run,
                                kmean_max_iter=kmean_max_iter,
                                param_dic=param_dic,
                                    )
                # print('path doesnt exists ===----->>>[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]',base_features_list)
                file_path_feat = self.file_path_feat if entry_name is None else self.file_path_feat_entry
                file_path = self.file_path if entry_name is None else self.file_path_entry
                pid.save_pinn_data(file_path_feat=file_path_feat,
                                file_path=file_path,
                                )    
                pid.get_dend_data(file_path_feat=file_path_feat,
                                file_path=file_path,)
                feat_paths=[]  
                for path_inten,name_inten in list_features: 
                    if model_init is not None: 
                        mode_dnn_1=self.model_sufix_dic['model_sufix_inverse'][model_init]
                        path_dir=self.model_sufix_dic['path_dir']
                        path_shaft_dir=f'{self.model_type}_{mode_dnn_1}_{path_dir}'
                        pathini=self.path_file[path_shaft_dir]
                    else:
                        pathini=file_path_feat


                    pathini=  self.path_file[path_train['data_shaft_path']]   

                    pathh=os.path.join( pathini ,name_inten)
                    if os.path.exists(pathh):
                        feat_paths.append(pathh)
                        print('train data path ---->>',pathini,pathh) 
                        print(np.loadtxt(pathh))
                    else:
                        print('path doesnt exists ===----->>>',pathh)
                    # print('path',self.path_file[path_train[path_inten]])  
                # print('path doesnt exists ===----->>>[[[[[[[[[[[[[[[[[[[[[[[[ggghhhffgh]]]]]]]]]]]]]]]]]]]]]]]]',len(list(list_features)+list(base_features_list)),base_features_list)
                adj.append(hff.adjoint(vertices=pid.dend.vertices,faces=pid.dend.faces,num_sub_nodes=num_sub_nodes))
                # curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
                adjs=hff.adjoint(vertices=pid.dend.vertices,faces=pid.dend.faces,num_sub_nodes=num_sub_nodes) 
                feat=pid.dend.vertices[adjs[1]]
                if len(list(list_features)+list(base_features_list))>0:
                    feats=np.hstack(pid.get_pinn_features(feat_paths=feat_paths,
                                                        base_features_list=base_features_list,
                                                        file_path_feat=file_path_feat,
                                                        file_path=file_path,))

                    feat=np.hstack((feat,feats[adjs[1]]))  
                    
                curv.append( 
                    { 
                    "x": tf.convert_to_tensor(feat , dtype=tf.float32), 
                    "a": tf.convert_to_tensor(adjs[0], dtype=tf.float32), 
                    "mask": tf.convert_to_tensor(tf.ones((adjs[0].shape[0],), dtype=tf.float32), dtype=tf.float32), 
                    }
                    )
                dend.append(pid.dend)   
                # curv.append(get_inputs(X=pid.dend.vertices[adjs[1]] ,adj=adjs[0]))
                if tf_train: 
                    rhs.append(tf.cast(
                        pid.get_pinn_rhs(pre_portion=pre_portion,
                                        file_path_feat=file_path_feat,
                                        file_path=file_path,
                                        dend_path_true_final=self.dend_path_true_final,
                                        ), 
                            dtype=DTYPE), 
                                    )  
                    pathh=os.path.join( self.dend_path_true_final ,'intensity_shaft_spine.txt')
                    mask=np.loadtxt(pathh,dtype=int)
                    unique=np.sort(np.unique(mask))
                    labels = {v: np.argwhere(mask == v) for v in unique}
                    counts = {v: labels[v].shape[0] for v in labels}
                    total = sum(counts.values()) 

                    w_prime = np.array([max(np.log(2 * total / counts[k]), 1) for k in unique ])
        
                    weight.append(w_prime / np.sum(w_prime))



        return curv, rhs ,adj,dend,weight


    def train_model_spine_gcn(self,
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
                            auc_save_dir=None,   
                            dice_save_dir=None,
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
                        entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
        auc_save_dir = auc_save_dir or model_dirs['auc']
        dice_save_dir = dice_save_dir or model_dirs['dice']
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']   
   

 
        curv,rhs,adj,dend,weights =self.get_train_input_gcn(   
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
                                tf_train=True,
                                entry_names=entry_names, 
                                kmean_n_run=kmean_n_run,
                                kmean_max_iter=kmean_max_iter,
                                param_dic=param_dic,
                                )  
        weight=[np.array([weig]+list(weight)) for weig in weights] 
        ls =[i for i in ls if i in range(len(curv))]
        for ii in ls:
            print(' [[[[[]]]]]',ii,curv[ii]['x'].shape)
        curv_train = [curv[i] for i in ls]
        rhs_train = [rhs[i] for i in ls] 
        dend_train=[dend[i] for i in ls]
        lss=np.arange(len(curv),dtype=int) 
        adj_train =[] 
        adj_train =[adj[i] for i in ls]
  
        from dend_fun_0.help_gcn_one_hot import LOSS,aka_train,Get_iou,model_choice,model_metric

        with tf.device(device):
            mchoice = model_choice(model_type=model_type, )
            model = mchoice.get_model() 
            custom = mchoice.get_custom_objects(model_type) 
 
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 5e-4])
        optimizer = tf.optimizers.Adam(learning_rate=lr) 
        if new_model:
            print(f"I'm starting a new model")
            print('--------------------------------')
            print(f"Model Type Gen : {model_type}")
            print(f"Model Type     : {model_sufix}")
            print(f"Model Dir.     : {model_dir}")
            print(f"Model Portion  : {pre_portion}")
            print(f"Data Dir.      : {path_train['data_shaft_path']}") 
            print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
            print(f"Feat. list     :{ls} {[mm for _,mm in list_features]}")  
            print(f"Feature Size   :{len(curv)} {curv[0]['x'].shape[1]} * {len(curv)}")
            print(f"Target Size    : {rhs[0].shape[1]} * {len(rhs)}")
            print(f"Dend Trainer size    : {[(fv['x'].shape[0],fv['x'].shape[1]) for fv in curv_train]}")
            print(f"Hidden Layers        : {hidden_layers}")
            print(f"neurons_per_layer    : {neurons_per_layer}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            auc_save = {0: [], 1: [], 2: []}
            dice_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            auc_save = np.loadtxt(auc_save_dir, dtype=float).tolist() 
            dice_save = np.loadtxt(dice_save_dir, dtype=float).tolist() 
            
        print(f"Using device: {device}------------------ ----------------------------------------------")
        aka_train_ = aka_train()

        
        indd=[[np.where(rhs[i][:,label]==1)[0] for label in range(rhs[i].shape[1])] for i in lss ]
        y_true_one_hot=np.vstack(rhs)
        # rhs_index=np.argmax(y_true_one_hot, axis=1) 
        rhs_index_ind = [np.argmax(yy, axis=1) for yy in rhs] 
        print(']]]]]]]]]]]]]]]]',y_true_one_hot.shape,)
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
 
        modlist=['loss','iou','auc','dice']
        metr={mm:{f'{nn}_tmp':[0,0] for nn in modlist }for mm in modlist}
        metr['loss']['loss_tmp']=1e6 
        tf.random.set_seed(0)
        iou_tmp= {0: 0, 1: 0, 2: 0};auc_tmp= {0: 0, 1: 0, 2: 0};dice_tmp= {0: 0, 1: 0, 2: 0}
        index_save=[]
        pbar = tqdm(range(itime), desc=f"Loss: {loss_tmp:.6f}| IoU: N/A")  
        for i in pbar: 
            loss = aka_train_.train_PINN(optimizer, fun, model)
            # iou=Get_iou(model, curv, lab=indd,
            #             adj=adj, 
            #        dend=dend,)
            # auc=get_auc(model, curv, rhs=rhs)
            mtr=model_metric(model=model,curv=curv,rhs_index=rhs_index_ind, )
            mtric=mtr.metrics
            iou=mtric['iou']
            auc=mtric['auc']
            dice=mtric['dice']
            for ii in range(rhs[0].shape[1]):
                iou_save[ii].append(mtric['iou_ind'][ii]) 
                np.savetxt(iou_save_dir[ii] , np.array(iou_save[ii]), fmt='%f') 
                auc_save[ii].append(mtric['auc_ind'][ii]) 
                np.savetxt(auc_save_dir[ii] , np.array(auc_save[ii]), fmt='%f') 
                dice_save[ii].append(mtric['dice_ind'][ii]) 
                np.savetxt(dice_save_dir[ii] , np.array(dice_save[ii]), fmt='%f') 
            loss_save.append(loss.numpy())
            np.savetxt(loss_save_dir, np.array(loss_save), fmt='%f') 

            
            for hh in modlist:
                if hh !='loss':
                    if mtric[hh][0] > metr[hh][f'{hh}_tmp'][0]:  
                        model.save(model_dirs[f'model_{hh}']) 
                        metr[hh][f'dice_tmp']=dice
                        metr[hh][f'auc_tmp']=auc
                        metr[hh][f'iou_tmp']=iou
                        metr[hh][f'loss_tmp']=loss 
                else:
                    # print('[[[[]]]]',loss.numpy(),metr)
                    if loss.numpy() < metr[hh][f'{hh}_tmp']:  
                        model.save(model_dirs[f'model_{hh}']) 
                        metr[hh][f'dice_tmp']=dice
                        metr[hh][f'auc_tmp']=auc
                        metr[hh][f'iou_tmp']=iou
                        metr[hh][f'loss_tmp']=loss 
    
            # if loss.numpy() < loss_tmp: 
            #     loss_tmp=loss.numpy()
            #     model.save(model_dir['model_loss']) 
            #     metr['loss']=dice[0]
            mmj=np.argmax([metr[mm]['dice_tmp'][0] for mm in modlist])
            mmjl=modlist[mmj]
            # optimizer=metr[mmjl][f'optimizer']
            # old_opt = old_model.optimizer
 
            modelsa = load_model(model_dirs[f'model_{mmjl}'], custom_objects=custom) 
            modelsa.save(model_dir) 
            index_save.append(i)
            np.savetxt(index_save_dir, np.array(index_save), fmt='%d') 
            loss_tmp,iou_tmp,auc_tmp,dice_tmp=metr[mmjl][f'loss_tmp'],metr[mmjl][f'iou_tmp'],metr[mmjl][f'auc_tmp'],metr[mmjl][f'dice_tmp']
            # if loss.numpy() < loss_tmp: 
            if pre_portion=='head_neck':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: { iou_tmp[0]:.4f} IoU nk: {iou_tmp[1]:.4f} IoU hd: {iou_tmp[2]:.4f}")
            elif pre_portion=='spine':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU(sh: {iou_tmp[0]:.2f}, sp: {iou_tmp[1]:.2f})|  DICE(sh: {dice_tmp[0]:.2f}, sp: {dice_tmp[1]:.2f}) |  AUC(sh: {auc_tmp[0]:.2f}, sp: {auc_tmp[1]:.2f})")
            else:
                pbar.set_description(f"Loss: {loss_tmp:.6f} ")
   
 
 
    def get_shaft_pred_gcn(self, 
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
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
        model_type_split=self.path_dir.lower().split('_') 
        for mo in ['dice','iou','auc','loss']:
            if mo in model_type_split:
                model_dir =  model_dirs[f'model_{mo}']  
                break 

        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name]) 
        print('-------------',n_col,model_dir)

        from dend_fun_0.help_gcn_one_hot import model_choice
        mchoice = model_choice()
        custom = mchoice.get_custom_objects(model_type) 
        model = load_model(model_dir, custom_objects=custom)
 
 
        print(f'get_shaft_pred started')
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )  
            key,tyy=f'{self.model_type}_{self.model_sufix}_{self.path_dir}', self.intensity_spines_logit[0]  
            path_ex = os.path.join(self.path_file_sub[tyy][key]) 
            if os.path.exists(path_ex) and not param_dic['tf_restart']['get_shaft_pred']: 
                continue

            pid.save_pinn_data() 
            pid.get_dend_data() 

            curv,_,_,_,_=self.get_train_input_gcn(   
                                    dend_names=[dend_name],
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
                                    tf_train=False,
                                    kmean_n_run=kmean_n_run,
                                    kmean_max_iter=kmean_max_iter,
                                    param_dic=param_dic,
                                    )  
            rhs0=model(curv[0]).numpy()

            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk], fmt='%f') 

              




            vert=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_0)) 
            vert_old=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_old)) 
            from scipy.spatial import KDTree 
            idx = KDTree(vert).query(vert_old)[1].flatten() 


            self.get_dend_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type,
                                ** param_dic['data']['get_dend_name'])    

            cxc=param_dic['data']['get_dend_name']['dict_dend_path']
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}' # if cxc in [None,'current'] else f'{cxc}_{self.model_type}_{self.model_sufix}_{self.path_dir}'
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk][idx], fmt='%f') 


 
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 

 


 




    def get_train_input_volume(self,  
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
                        mshape=None,
                        tf_train=True,
                        mskl_margin=2,
                        mskl_multiple=16,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
        from dend_fun_0.help_data_volume import mesh_without_vertices,get_report,plot_slice, mesh_to_volume,plot_voxel_volume,vol_cropper
        from dend_fun_0.help_cnn_one_hot import model_choice

        mchoice = model_choice(model_type=model_type,)
        cropper_params = mchoice.get_cropper_params( )


        curv, rhs, weight, indices,adj,dend = [], [], [], [],[],[]
        volumes=[]
        masks=[]
        idx_originals=[]
        meshes=[]
        mskls=[]
        spine_indexs=[] 
        indexx={dend_name:index for index, dend_name  in enumerate(self.dend_names)} 
        for entry_name in entry_names:
            for dend_name in dend_names: 
                index=indexx[dend_name]
                self.get_dend_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name)
                obj_org_path= self.obj_org_path if entry_name is None else self.obj_org_path_entry
                file_path= self.file_path if entry_name is None else self.file_path_entry
                file_path_feat=self.file_path_feat if entry_name is None else self.file_path_feat_entry
                # path=os.path.join(obj_org_path,self.dend_name,'data') 
                vertices_0 = np.loadtxt(os.path.join(file_path, self.txt_vertices_0), dtype=float)
                faces = np.loadtxt(os.path.join(file_path, self.txt_faces), dtype=int)

                mesh_org=trimesh.Trimesh(vertices=vertices_0,faces=faces)
                # mesh_org=trimesh.load_mesh(os.path.join(path,f'{self.dend_name}.obj'))
                # vcrop=vol_cropper(margin=2,multiple=16) 
                mskl=mesh_to_volume(mesh_org.vertices, mesh_org.faces,
                                    margin=cropper_params['margin'],
                                    multiple=cropper_params['multiple'])
                vol=mskl.get_volume() 
                if not tf_train: 
                    vol,idx_new=mskl.get_volume_crop(mesh_shaft_wrap=None,fill_method="orthographic")
                    volumes.append(tf.convert_to_tensor(tf.cast(vol[None,...,None], tf.float32), dtype=tf.float32)) 
                    idx_originals.append(idx_new)
                    print('[[[]]]=============================',mskl.volume.shape,idx_originals[-1].shape)
                    meshes.append(mesh_org)
                    mskls.append(mskl)
                    continue
 
                mesh_shaft_wrap = trimesh.load_mesh(os.path.join(file_path_feat, f'{self.dend_first_name}_shaft_wrap.obj'))

                vol,mask,idx_org =mskl.get_volume_crop(mesh_shaft_wrap=mesh_shaft_wrap,fill_method="orthographic") 
                mask_onehot = tf.one_hot(mask, depth=3)[None,...] 

                masks.append(mask_onehot)
                mask_onehot=mask_onehot.numpy()
                c2 = mask_onehot[0,..., 1:2][idx_org[:,0],idx_org[:,1],idx_org[:,2]].flatten()
                c3 = mask_onehot[0,..., 2:3][idx_org[:,0],idx_org[:,1],idx_org[:,2]].flatten()
                rhs.append(np.array([c2,c3],dtype=int).T)
                # masks.append(mask[None,...,None])
                volumes.append(tf.convert_to_tensor(tf.cast(vol[None,...,None], tf.float32), dtype=tf.float32))
                unique=np.sort(np.unique(mask))
                labels = {v: np.argwhere(mask == v) for v in unique}
                counts = {v: labels[v].shape[0] for v in labels}
                total = sum(counts.values()) 

                w_prime = np.array([max(np.log(2 * total / counts[k]), 1) for k in unique ])
    
                weight.append(w_prime / np.sum(w_prime))

                weights = {v: total / counts[v] for v in counts} 
                weights = {v: weights[v] / max(weights.values()) for v in weights}
                weights[0]=0
                weight_map = np.zeros_like(mask, dtype=np.float32) 
                for v, coords in labels.items():
                    weight_map[coords[:,0], coords[:,1], coords[:,2]] = weights[v]
                weight_map_tf = tf.convert_to_tensor(weight_map, dtype=tf.float32)
                # weight_map_tf = weight_map_tf[None, ...]   # match pred shape



                idx_originals.append(weight_map_tf[None, ...])
                print('[[[]]]=============================',mskl.volume.shape,idx_originals[-1].shape)
                meshes.append(mesh_org)
                mskls.append(mskl) 
    
        return volumes, masks,mskls,spine_indexs,idx_originals,weight,rhs
 

    def train_model_spine_cnn(self,
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
                            auc_save_dir=None,   
                            dice_save_dir=None,
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
                        mskl_margin=2,
                        mskl_multiple=16,
                        entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
        auc_save_dir = auc_save_dir or model_dirs['auc']
        dice_save_dir = dice_save_dir or model_dirs['dice']
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']   


        from dend_fun_0.help_cnn_one_hot import vol_UNet3D,LOSS,aka_train,vol_FastFCN3D,model_choice,Get_iou,get_auc,model_metric




        curv,rhs,mskls,spine_indexs,idx_originals,weight_mask,rhshr =self.get_train_input_volume(   
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
                                tf_train=True,
                                mskl_margin=mskl_margin,
                                mskl_multiple=mskl_multiple,
                                entry_names=entry_names,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                )   
        adj_train =[]  
        # weight=[np.array([weig]+list(weight)) for weig in weights]
        ls =[i for i in ls if i in range(len(curv))]
        print('========',ls,len(curv))
        curv_train = [curv[i] for i in ls]
        rhs_train = [rhs[i] for i in ls] 
        idx_originals_train=[idx_originals[i] for i in ls]
        # dend_train=[dend[i] for i in ls]
        lss=np.arange(len(curv),dtype=int) 
        # model_type="fastfcn3d" 
        with tf.device(device):
            mchoice = model_choice(model_type=model_type,)
            model = mchoice.get_model( )
            custom = mchoice.get_custom_objects(model_type) 
    
        if new_model:
            print(f"I'm starting a new model")
            print('--------------------------------')
            print(f"Model Type Gen : {model_type}")
            print(f"Model Type     : {model_sufix}")
            print(f"Model Dir.     : {model_dir}")
            print(f"Model Portion  : {pre_portion}")
            print(f"Data Dir.      : {path_train['data_shaft_path']}") 
            # print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
            # print(f"Feat. list     : {[mm for _,mm in list_features]}") 
            print(f"Feature Size   : {curv[0].shape[1]} * {len(curv)}")
            print(f"Target Size    : {rhs[0].shape[1]} * {len(rhs)}")
            print(f"Dend Trainer size    : {[tuple(fv.shape) for fv in curv_train]}")
            print(f"Hidden Layers        : {hidden_layers}")
            print(f"neurons_per_layer    : {neurons_per_layer}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            auc_save = {0: [], 1: [], 2: []}
            dice_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            auc_save = np.loadtxt(auc_save_dir, dtype=float).tolist() 
            dice_save = np.loadtxt(dice_save_dir, dtype=float).tolist() 
            
            
        print(f"Using device: {device}---------------- ---------------------------------------------") 

        # optimizer = tf.keras.optimizers.Adam(1e-4)
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 5e-4])
        optimizer = tf.optimizers.Adam(learning_rate=lr)

        fun = LOSS(volumes=curv_train, masks=rhs_train,idx_originals=idx_originals_train, loss_mode='wbce',weight=weight_mask)

        indd=[[np.where(rhs[i][:,label]==1)[0] for label in range(rhs[i].shape[1])] for i in lss ]
        aka_train_ = aka_train() 

        modlist=['loss','iou','auc','dice']
        metr={mm:{f'{nn}_tmp':[0,0] for nn in modlist }for mm in modlist}
        metr['loss']['loss_tmp']=1e6 

        rhs_index_ind = [np.argmax(yy, axis=1) for yy in rhshr]
        tf.random.set_seed(0)
        iou_tmp= {0: 0, 1: 0, 2: 0};auc_tmp= {0: 0, 1: 0, 2: 0};dice_tmp= {0: 0, 1: 0, 2: 0}
        index_save=[]
        pbar = tqdm(range(itime), desc=f"Loss: {loss_tmp:.6f}| IoU: N/A")  
        for i in pbar: 
            loss = aka_train_.train_PINN(optimizer, fun, model)
            # iou=Get_iou(model,lab=indd, mskls=mskls,)  
            # auc=get_auc(model,mskls=mskls, rhs=rhshr)
            mtr=model_metric(model=model,mskls=mskls,rhs_index=rhs_index_ind, ) 
            mtric=mtr.metrics
            iou=mtric['iou']
            auc=mtric['auc']
            dice=mtric['dice']
            for ii in range(rhshr[0].shape[1]): 
                iou_save[ii].append(mtric['iou_ind'][ii]) 
                np.savetxt(iou_save_dir[ii] , np.array(iou_save[ii]), fmt='%f') 
                auc_save[ii].append(mtric['auc_ind'][ii]) 
                np.savetxt(auc_save_dir[ii] , np.array(auc_save[ii]), fmt='%f') 
                dice_save[ii].append(mtric['dice_ind'][ii]) 
                np.savetxt(dice_save_dir[ii] , np.array(dice_save[ii]), fmt='%f') 
            loss_save.append(loss.numpy())
            np.savetxt(loss_save_dir, np.array(loss_save), fmt='%f') 

            
            for hh in modlist:
                if hh !='loss':
                    if mtric[hh][0] > metr[hh][f'{hh}_tmp'][0]:  
                        model.save(model_dirs[f'model_{hh}']) 
                        metr[hh][f'dice_tmp']=dice
                        metr[hh][f'auc_tmp']=auc
                        metr[hh][f'iou_tmp']=iou
                        metr[hh][f'loss_tmp']=loss 
                else:
                    # print('[[[[]]]]',loss.numpy(),metr)
                    if loss.numpy() < metr[hh][f'{hh}_tmp']:  
                        model.save(model_dirs[f'model_{hh}']) 
                        metr[hh][f'dice_tmp']=dice
                        metr[hh][f'auc_tmp']=auc
                        metr[hh][f'iou_tmp']=iou
                        metr[hh][f'loss_tmp']=loss 
    
            # if loss.numpy() < loss_tmp: 
            #     loss_tmp=loss.numpy()
            #     model.save(model_dir['model_loss']) 
            #     metr['loss']=dice[0]
            mmj=np.argmax([metr[mm]['dice_tmp'][0] for mm in modlist])
            mmjl=modlist[mmj]
            # optimizer=metr[mmjl][f'optimizer']
            # old_opt = old_model.optimizer
 
            modelsa = load_model(model_dirs[f'model_{mmjl}'], custom_objects=custom) 
            modelsa.save(model_dir) 
            index_save.append(i)
            np.savetxt(index_save_dir, np.array(index_save), fmt='%d') 
            loss_tmp,iou_tmp,auc_tmp,dice_tmp=metr[mmjl][f'loss_tmp'],metr[mmjl][f'iou_tmp'],metr[mmjl][f'auc_tmp'],metr[mmjl][f'dice_tmp']
            # if loss.numpy() < loss_tmp:
            # # if auc[0] > auc_tmp[0]:
            # # if dice[0] > dice_tmp[0]:
            #     # iou_tmp=iou  
            #     # dice_tmp=dice 
            #     loss_tmp=loss.numpy()
            #     model.save(model_dir) 
            #     index_save.append(i)
            #     np.savetxt(index_save_dir, np.array(index_save), fmt='%d')  
            # print(auc_tmp)
            if pre_portion=='head_neck':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: { iou_tmp[0]:.4f} IoU nk: {iou_tmp[1]:.4f} IoU hd: {iou_tmp[2]:.4f}")
            elif pre_portion=='spine':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU(sh: {iou_tmp[0]:.2f}, sp: {iou_tmp[1]:.2f})|  DICE(sh: {dice_tmp[0]:.2f}, sp: {dice_tmp[1]:.2f}) |  AUC(sh: {auc_tmp[0]:.2f}, sp: {auc_tmp[1]:.2f})")
            else:
                pbar.set_description(f"Loss: {loss_tmp:.6f} ")
   


    def get_shaft_pred_cnn(self, 
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
                        mskl_margin=2,
                        mskl_multiple=16,
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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

        from dend_fun_0.help_vol_one_hot import model_choice 


        mchoice = model_choice(model_type=model_type)
        mmnn=['vol']
        mmnn.extend(model_type.split('_')[1:])
        model_type_l='_'.join(mmnn)
        print('[[[[[[[]]]]]]]',model_type_l)
        custom = mchoice.get_custom_objects(model_type_l) 
        model = load_model(model_dir, custom_objects=custom)
     

        print(f'get_shaft_pred started')
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )  
            key,tyy=f'{self.model_type}_{self.model_sufix}_{self.path_dir}', self.intensity_spines_logit[0]  
            path_ex = os.path.join(self.path_file_sub[tyy][key]) 
            if os.path.exists(path_ex) and not param_dic['tf_restart']['get_shaft_pred']: 
                continue
            pid.save_pinn_data() 
            pid.get_dend_data()
            con=0 
            
            curv,rhs,mskls,spine_indexs,idx_originals,weight_mask,rhshr  =self.get_train_input_volume(   
                                    dend_names=[dend_name],
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
                                    tf_train=False,
                                    mskl_margin=mskl_margin,
                                    mskl_multiple=mskl_multiple,
                                    )   
            rhs0=mskls[0].get_pred_rhs_crop(model,)[:,[1,0]]  
            # rhs00=mskls[0].get_pred_rhs(model,)[:,[1,0]]  
            print('-------------------=========rhs' ,mskls[0].vertices.shape)

            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk], fmt='%f') 

            vert=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_0)) 
            vert_old=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_old)) 
            from scipy.spatial import KDTree 
            idx = KDTree(vert).query(vert_old)[1].flatten() 


            self.get_dend_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type,
                                ** param_dic['data']['get_dend_name'])    

            cxc=param_dic['data']['get_dend_name']['dict_dend_path']
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}' # if cxc in [None,'current'] else f'{cxc}_{self.model_type}_{self.model_sufix}_{self.path_dir}'
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk][idx], fmt='%f')  

            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 








    def get_train_input_dnn(self,  
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
                        model_init=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        model_init = model_init if model_init is not None else self.model_init
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

        indexx={dend_name:index for index, dend_name  in enumerate(self.dend_names)} 
        for entry_name in entry_names:
            for dend_name in dend_names: 
                index=indexx[dend_name]
                self.get_dend_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name) 
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
                                kmean_n_run=kmean_n_run,
                                kmean_max_iter=kmean_max_iter,
                                param_dic=param_dic,
                                    )
                print('path doesnt exists ===----->>>[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]',base_features_list)
                file_path_feat = self.file_path_feat if entry_name is None else self.file_path_feat_entry
                file_path = self.file_path if entry_name is None else self.file_path_entry
                pid.save_pinn_data(file_path_feat=file_path_feat,
                                file_path=file_path,
                                )    
                pid.get_dend_data(file_path_feat=file_path_feat,
                                file_path=file_path,)
                feat_paths=[]  
                for path_inten,name_inten in list_features: 
                    if model_init is not None: 
                        mode_dnn_1=self.model_sufix_dic['model_sufix_inverse'][model_init]
                        path_dir=self.model_sufix_dic['path_dir']
                        path_shaft_dir=f'{self.model_type}_{mode_dnn_1}_{path_dir}'
                        pathini=self.path_file[path_shaft_dir]
                    else:
                        pathini=file_path_feat

                    pathini=  self.path_file[path_train['data_shaft_path']]   

                    pathh=os.path.join( pathini ,name_inten)
                    if os.path.exists(pathh):
                        feat_paths.append(pathh)
                        print('train data path ---->>',pathini,pathh) 
                        print(np.loadtxt(pathh))
                    else:
                        print('path doesnt exists ===----->>>',pathh)
                dend.append(pid.dend)     
                if tf_train:
                    rhs.append(tf.cast(
                        pid.get_pinn_rhs(pre_portion=pre_portion,
                                        file_path_feat=file_path_feat,
                                        file_path=file_path,
                                        dend_path_true_final=self.dend_path_true_final,
                                        ), 
                        dtype=DTYPE)
                            )
                    pathh=os.path.join( self.dend_path_true_final ,'intensity_shaft_spine.txt')
                    mask=np.loadtxt(pathh,dtype=int)
                    unique=np.sort(np.unique(mask))
                    labels = {v: np.argwhere(mask == v) for v in unique}
                    counts = {v: labels[v].shape[0] for v in labels}
                    total = sum(counts.values()) 

                    w_prime = np.array([max(np.log(2 * total / counts[k]), 1) for k in unique ])
        
                    weight.append(w_prime / np.sum(w_prime))






                if len(list(list_features)+list(base_features_list))==0:
                    curv.append(pid.dend.vertices)  
                    continue
                    
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,
                                                            base_features_list=base_features_list,
                                                            file_path_feat=file_path_feat,
                                                            file_path=file_path,))) 

        return curv, rhs ,adj,dend,weight
 
 
    def train_model_spine_dnn(self,
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
                            auc_save_dir=None,   
                            dice_save_dir=None,
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
                        entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
        auc_save_dir = auc_save_dir or model_dirs['auc']
        dice_save_dir = dice_save_dir or model_dirs['dice']
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']   
   

 
        curv,rhs,adj,dend,weights =self.get_train_input_dnn(   
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
                                tf_train=True,
                                entry_names=entry_names, 
                                kmean_n_run=kmean_n_run,
                                kmean_max_iter=kmean_max_iter,
                                param_dic=param_dic,
                                ) 
        weight=[np.array([weig]+list(weight)) for weig in weights] 
        ls =[i for i in ls if i in range(len(curv))]
        for ii in ls:
            print(' [[[[[]]]]]',ii,curv[ii].shape)
        curv_train = [curv[i] for i in ls ]
        rhs_train = [rhs[i] for i in ls] 
        dend_train=[dend[i] for i in ls]
        lss=np.arange(len(curv),dtype=int) 
        adj_train =[] 
        from dend_fun_0.help_dnn_one_hot import LOSS,aka_train,Get_iou,model_choice,get_auc,model_metric
        with tf.device(device):
            mchoice = model_choice(model_type=model_type, )
            model = mchoice.get_model() 
            custom = mchoice.get_custom_objects(model_type) 
                 
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 5e-4])
        optimizer = tf.optimizers.Adam(learning_rate=lr) 
        if new_model:
            print(f"I'm starting a new model")
            print('--------------------------------')
            print(f"Model Type Gen : {model_type}")
            print(f"Model Type     : {model_sufix}")
            print(f"Model Dir.     : {model_dir}")
            print(f"Model Portion  : {pre_portion}")
            print(f"Data Dir.      : {path_train['data_shaft_path']}") 
            print(f"Base Feat. list: {[mm for mm in base_features_list] if base_features_list is not None else list(self.base_features_dict.keys())}") 
            print(f"Feat. list     : {[mm for _,mm in list_features]}") 
            print(f"Feature Size   : {curv[0].shape[1]} * {len(curv)}")
            print(f"Target Size    : {rhs[0].shape[1]} * {len(rhs)}")
            print(f"Dend Trainer size    : {[len(fv) for fv in curv_train]}")
            print(f"Hidden Layers        : {hidden_layers}")
            print(f"neurons_per_layer    : {neurons_per_layer}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            auc_save = {0: [], 1: [], 2: []}
            dice_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            auc_save = np.loadtxt(auc_save_dir, dtype=float).tolist() 
            dice_save = np.loadtxt(dice_save_dir, dtype=float).tolist() 
            
        print(f"Using device: {device}------------------ ----------------------------------------------")
        aka_train_ = aka_train()

        
        indd=[[np.where(rhs[i][:,label]==1)[0] for label in range(rhs[i].shape[1])] for i in lss ]
        
        # indd=[[np.where(rhs[i][:,label]==1)[0] for label in range(rhs[i].shape[1])] for i in lss ]
        y_true_one_hot=np.vstack(rhs)
        # rhs_index=np.argmax(y_true_one_hot, axis=1) 
        rhs_index_ind = [np.argmax(yy, axis=1) for yy in rhs]
        # rhs_index= sum([list(mm) for mm in rhs_index_ind],[]) 
        # rhs_index=[np.where(y_true_one_hot[:,label]==1)[0] for label in range(rhs[0].shape[1])]rhs_index.shape 
        print(']]]]]]]]]]]]]]]]',y_true_one_hot.shape,)
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
        modlist=['loss','iou','auc','dice']
        metr={mm:{f'{nn}_tmp':[0,0] for nn in modlist }for mm in modlist}
        metr['loss']['loss_tmp']=1e6 
        tf.random.set_seed(0)
        iou_tmp= {0: 0, 1: 0, 2: 0};auc_tmp= {0: 0, 1: 0, 2: 0};dice_tmp= {0: 0, 1: 0, 2: 0}
        index_save=[]
        pbar = tqdm(range(itime), desc=f"Loss: {loss_tmp:.6f}| IoU: N/A")  
        for i in pbar: 
            loss = aka_train_.train_PINN(optimizer, fun, model)
            # iou=Get_iou(model, curv, lab=indd,
            #             adj=adj, 
            #        dend=dend,)
            # auc=get_auc(model, curv, rhs=rhs)
            mtr=model_metric(model=model,curv=curv,rhs_index=rhs_index_ind, )
            mtric=mtr.metrics
            iou=mtric['iou']
            auc=mtric['auc']
            dice=mtric['dice']
            for ii in range(rhs[0].shape[1]):
                iou_save[ii].append(mtric['iou_ind'][ii]) 
                np.savetxt(iou_save_dir[ii] , np.array(iou_save[ii]), fmt='%f') 
                auc_save[ii].append(mtric['auc_ind'][ii]) 
                np.savetxt(auc_save_dir[ii] , np.array(auc_save[ii]), fmt='%f') 
                dice_save[ii].append(mtric['dice_ind'][ii]) 
                np.savetxt(dice_save_dir[ii] , np.array(dice_save[ii]), fmt='%f') 
            loss_save.append(loss.numpy())
            np.savetxt(loss_save_dir, np.array(loss_save), fmt='%f') 

            
            for hh in modlist:
                if hh !='loss':
                    if mtric[hh][0] > metr[hh][f'{hh}_tmp'][0]:  
                        model.save(model_dirs[f'model_{hh}']) 
                        metr[hh][f'dice_tmp']=dice
                        metr[hh][f'auc_tmp']=auc
                        metr[hh][f'iou_tmp']=iou
                        metr[hh][f'loss_tmp']=loss 
                else:
                    # print('[[[[]]]]',loss.numpy(),metr)
                    if loss.numpy() < metr[hh][f'{hh}_tmp']:  
                        model.save(model_dirs[f'model_{hh}']) 
                        metr[hh][f'dice_tmp']=dice
                        metr[hh][f'auc_tmp']=auc
                        metr[hh][f'iou_tmp']=iou
                        metr[hh][f'loss_tmp']=loss 
    
            # if loss.numpy() < loss_tmp: 
            #     loss_tmp=loss.numpy()
            #     model.save(model_dir['model_loss']) 
            #     metr['loss']=dice[0]
            mmj=np.argmax([metr[mm]['dice_tmp'][0] for mm in modlist])
            mmjl=modlist[mmj]
            # optimizer=metr[mmjl][f'optimizer']
            # old_opt = old_model.optimizer
 
            modelsa = load_model(model_dirs[f'model_{mmjl}'], custom_objects=custom) 
            modelsa.save(model_dir) 
            index_save.append(i)
            np.savetxt(index_save_dir, np.array(index_save), fmt='%d') 
            loss_tmp,iou_tmp,auc_tmp,dice_tmp=metr[mmjl][f'loss_tmp'],metr[mmjl][f'iou_tmp'],metr[mmjl][f'auc_tmp'],metr[mmjl][f'dice_tmp']
            # if loss.numpy() < loss_tmp:
            # # if auc[0] > auc_tmp[0]:
            # # if dice[0] > dice_tmp[0]:
            #     # iou_tmp=iou  
            #     # dice_tmp=dice 
            #     loss_tmp=loss.numpy()
            #     model.save(model_dir) 
            #     index_save.append(i)
            #     np.savetxt(index_save_dir, np.array(index_save), fmt='%d')  
            # print(auc_tmp)
            if pre_portion=='head_neck':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: { iou_tmp[0]:.4f} IoU nk: {iou_tmp[1]:.4f} IoU hd: {iou_tmp[2]:.4f}")
            elif pre_portion=='spine':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU(sh: {iou_tmp[0]:.2f}, sp: {iou_tmp[1]:.2f})|  DICE(sh: {dice_tmp[0]:.2f}, sp: {dice_tmp[1]:.2f}) |  AUC(sh: {auc_tmp[0]:.2f}, sp: {auc_tmp[1]:.2f})")
            else:
                pbar.set_description(f"Loss: {loss_tmp:.6f} ")
   
 
    def get_shaft_pred_dnn(self, 
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
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
        model_type_split=self.path_dir.lower().split('_') 
        for mo in ['dice','iou','auc','loss']:
            if mo in model_type_split:
                model_dir =  model_dirs[f'model_{mo}']  
                break 

        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name])  

 
        if model_type.startswith(('dnn',)):
            from dend_fun_0.help_dnn_one_hot import model_choice  

        mchoice = model_choice(model_type=model_type)
        custom = mchoice.get_custom_objects(model_type) 
        model = load_model(model_dir, custom_objects=custom)

        print(f'get_shaft_pred started')
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
                                model_type=model_type, )  
            dend_name=self.dend_name  
            spine_portion_path=self.path_file[path_train['data_shaft_path']] 
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )  
            key,tyy=f'{self.model_type}_{self.model_sufix}_{self.path_dir}', self.intensity_spines_logit[0]  
            path_ex = os.path.join(self.path_file_sub[tyy][key]) 
            if os.path.exists(path_ex) and not param_dic['tf_restart']['get_shaft_pred']: 
                continue
            pid.save_pinn_data() 
            pid.get_dend_data()
            con=0

            curv,_,_,_,_=self.get_train_input_dnn(   
                                    dend_names=[dend_name],
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
                                    tf_train=False,
                                    kmean_n_run=kmean_n_run,
                                    kmean_max_iter=kmean_max_iter,
                                    param_dic=param_dic,
                                    )   
            rhs0=model(curv[0]).numpy()   
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk], fmt='%f') 
 






            vert=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_0)) 
            vert_old=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_old)) 
            from scipy.spatial import KDTree 
            idx = KDTree(vert).query(vert_old)[1].flatten() 


            self.get_dend_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type,
                                ** param_dic['data']['get_dend_name'])    

            # cxc=param_dic['data']['get_dend_name']['dict_dend_path']
            # key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}' # if cxc in [None,'current'] else f'{cxc}_{self.model_type}_{self.model_sufix}_{self.path_dir}'
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk][idx], fmt='%f') 


 
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 







    def get_train_input_pnet(self,  
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
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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

        indexx={dend_name:index for index, dend_name  in enumerate(self.dend_names)} 
        for entry_name in entry_names:
            for dend_name in dend_names: 
                index=indexx[dend_name]
                self.get_dend_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name) 
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                    )
                file_path_feat = self.file_path_feat if entry_name is None else self.file_path_feat_entry
                file_path = self.file_path if entry_name is None else self.file_path_entry
                pid.save_pinn_data(file_path_feat=file_path_feat,
                                file_path=file_path,
                                )    
                pid.get_dend_data(file_path_feat=file_path_feat,
                                file_path=file_path,)
                feat_paths=[]  
                for path_inten,name_inten in list_features:  
                    pathh=os.path.join( file_path_feat ,name_inten)
                    if os.path.exists(pathh):
                        feat_paths.append(pathh)
                        print('train data path ---->>',file_path_feat,pathh) 
                        print(np.loadtxt(pathh))
                    else:
                        print('path doesnt exists ===----->>>',pathh)
                dend.append(pid.dend)     
                if tf_train:
                    rhspp=tf.cast(

                        pid.get_pinn_rhs(pre_portion=pre_portion,
                                        file_path_feat=file_path_feat,
                                        file_path=file_path,
                                        dend_path_true_final=self.dend_path_true_final,
                                        ),  
                        dtype=DTYPE)
                    rhs.append(rhspp)
                if len(list(list_features)+list(base_features_list))==0: 
                    points = tf.expand_dims(tf.convert_to_tensor(pid.dend.vertices), axis=0)   
                    curv.append(points) 
                    continue
                    
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,
                                                            base_features_list=base_features_list,
                                                            file_path_feat=file_path_feat,
                                                            file_path=file_path,))) 
        return curv, rhs ,adj,dend
 
 
    def train_model_spine_pnet(self,
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
                            auc_save_dir=None,
                            dice_save_dir=None,
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
                        entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
        auc_save_dir = auc_save_dir or model_dirs['auc']
        dice_save_dir = dice_save_dir or model_dirs['dice']
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        model_dir = model_dir or  model_dirs['model']  
   

 
        curv,rhs,adj,dend =self.get_train_input_pnet(   
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
                                tf_train=True,
                                entry_names=entry_names,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                ) 
        ls =[i for i in ls if i in range(len(curv))]
        curv_train = [curv[i] for i in ls]
        rhs_train = [tf.expand_dims(tf.convert_to_tensor(rhs[i]), axis=0) for i in ls] 
        dend_train=[dend[i] for i in ls] 
        lss=np.arange(len(curv),dtype=int) 
        adj_train =[] 

        from dend_fun_0.help_pnet_one_hot import LOSS,aka_train,Get_iou,model_choice,get_auc
        with tf.device(device):
            mchoice = model_choice()
            model = mchoice.get_model(model_type=model_type, )

                 
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
            print(f"Feature Size   : {curv[0].shape[1]} * {len(curv)}")
            print(f"Target Size    : {rhs[0].shape[1]} * {len(rhs)}")
            print(f"Dend Trainer size    : {[len(fv) for fv in curv_train]}")
            print(f"Hidden Layers        : {hidden_layers}")
            print(f"neurons_per_layer    : {neurons_per_layer}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            auc_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            auc_save = np.loadtxt(auc_save_dir, dtype=float).tolist() 
            
        print(f"Using device: {device}------------------ ----------------------------------------------")
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
        iou_tmp=[0,0,0];auc_tmp=[0,0,0];index_save=[]
        
        pbar = tqdm(range(itime), desc=f"Loss: {loss_tmp:.6f}| IoU: N/A")  
        for i in pbar: 
            loss = aka_train_.train_PINN(optimizer, fun, model)
            iou=Get_iou(model, 
                        curv, 
                        lab=indd,
                        adj=adj, 
                   dend=dend,)
            auc=get_auc(model, curv, rhs=rhs)
            for ii in range(rhs[0].shape[1]):
                iou_save[ii].append(iou[ii]) 
                np.savetxt(iou_save_dir[ii] , np.array(iou_save[ii]), fmt='%f') 
                auc_save[ii].append(auc[ii]) 
                np.savetxt(auc_save_dir[ii] , np.array(auc_save[ii]), fmt='%f') 
            loss_save.append(loss.numpy())
            np.savetxt(loss_save_dir, np.array(loss_save), fmt='%f') 
            # if loss.numpy() < loss_tmp:
            if auc[0] > auc_tmp[0]:
                loss_tmp=loss.numpy()
                model.save(model_dir) 
                iou_tmp=iou  
                auc_tmp=auc

                index_save.append(i)
                np.savetxt(index_save_dir, np.array(index_save), fmt='%d')  
            if pre_portion=='head_neck':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: {min(iou_tmp[0]):.4f} IoU nk: {min(iou_tmp[1]):.4f} IoU hd: {min(iou_tmp[2]):.4f}")
            elif pre_portion=='spine':
                pbar.set_description(f"Loss: {loss_tmp:.6f} |  IoU sh: {min(iou_tmp[0]):.4f} IoU sp: {min(iou_tmp[1]):.4f} |  AUC sh: {auc_tmp[0]:.4f} AUC sp: {auc_tmp[1]:.4f}")
            else:
                pbar.set_description(f"Loss: {loss_tmp:.6f} ")
   
 
    def get_shaft_pred_pnet(self, 
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
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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

 
        if model_type.startswith('pnet'):
            from dend_fun_0.help_pnet_one_hot import model_choice  

        with tf.device(device):
            mchoice = model_choice(model_type=model_type)
            custom = mchoice.get_custom_objects(model_type) 
            model = load_model(model_dir, custom_objects=custom)



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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )  
            key,tyy=f'{self.model_type}_{self.model_sufix}_{self.path_dir}', self.intensity_spines_logit[0]  
            path_ex = os.path.join(self.path_file_sub[tyy][key]) 
            if os.path.exists(path_ex) and not param_dic['tf_restart']['get_shaft_pred']: 
                continue
            pid.save_pinn_data() 
            pid.get_dend_data()
            con=0

            curv,_,_,_ =self.get_train_input_pnet(   
                                    dend_names=[dend_name],
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
                                    tf_train=False,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                    )   
            rhs0=model(curv[0]).numpy()[0,...] 

            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk], fmt='%f') 
 
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 












    def get_train_input_PINN(self,  
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
                        tf_train=True,
                        entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
 
        indexx={dend_name:index for index, dend_name  in enumerate(self.dend_names)} 
        for entry_name in entry_names:
            for dend_name in dend_names: 
                index=indexx[dend_name]
                self.get_dend_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name) 
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                    )
                file_path_feat = self.file_path_feat if entry_name is None else self.file_path_feat_entry
                file_path = self.file_path if entry_name is None else self.file_path_entry
                pid.save_pinn_data(file_path_feat=file_path_feat,
                                file_path=file_path,
                                )    
                pid.get_dend_data(file_path_feat=file_path_feat,
                                file_path=file_path,)
                feat_paths=[]  
                for path_inten,name_inten in list_features:  
                    pathh=os.path.join( self.file_path_feat ,name_inten)
                    if os.path.exists(pathh):
                        feat_paths.append(pathh)
                        print('train data path ---->>',self.file_path_feat,pathh) 
                        print(np.loadtxt(pathh))
                    else:
                        print('path doesnt exists ===----->>>',pathh)

                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
                dend.append(pid.dend)     
                if tf_train:
                            rhs.append(
                                pid.get_pinn_rhs(pre_portion=pre_portion,
                                                file_path_feat=file_path_feat,
                                                file_path=file_path,
                                                dend_path_true_final=self.dend_path_true_final,
                                                ), 
                                            )
        return curv, rhs ,adj,dend
 
 
    def train_model_spine_PINN(self,
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
                            entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
  
        if model_type.startswith('pinn'):
            from dend_fun_0.help_pinn_one_hot import LOSS,PINN,aka_train,Get_iou 
            adj_tf=False
        elif model_type.startswith('rpinn'):
            from dend_fun_0.help_pinn_rein_one_hot  import LOSS,PINN,aka_train,Get_iou 
            adj_tf=False

 
        curv,rhs,adj,dend =self.get_train_input_PINN(   
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
                                tf_train=True,
                                entry_names=entry_names,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                ) 
        ls =[i for i in ls if i in range(len(curv))]
        curv_train = [curv[i] for i in ls]
        rhs_train = [rhs[i] for i in ls] 
        dend_train=[dend[i] for i in ls]
        lss=np.arange(len(curv),dtype=int) 
        adj_train =[] 
 
                
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
            
        print(f"Using device: {device}------------------ ----------------------------------------------")
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
   
 
    def get_shaft_pred_PINN(self, 
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
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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

 
        if model_type.startswith('pinn'):
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )  
            key,tyy=f'{self.model_type}_{self.model_sufix}_{self.path_dir}', self.intensity_spines_logit[0]  
            path_ex = os.path.join(self.path_file_sub[tyy][key]) 
            if os.path.exists(path_ex) and not param_dic['tf_restart']['get_shaft_pred']: 
                continue
            pid.save_pinn_data() 
            pid.get_dend_data()
            con=0

            curv,_,_,_,_=self.get_train_input_dnn(   
                                    dend_names=[dend_name],
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
                                    tf_train=False,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                    )   
            rhs0=model(curv[0]).numpy()   
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(self.intensity_spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk], fmt='%f') 

            # pid.get_shaft_pred(
            #                 # dend=pid.dend,
            #                 rhs=rhs0,
            #                 weight=weight,
            #                 weights=weights, 
            #                 path_train=path_train, 
            #                 pre_portion=pre_portion,
            #                 gauss_threshold=self.gauss_threshold,
            #                 size_threshold=size_threshold, 
            #                 shaft_thre=shaft_thre,
            #                 smooth_tf=smooth_tf,
            #                 neck_lim=neck_lim,
            #                 dict_mesh_to_skeleton_finder=dict_mesh_to_skeleton_finder,
            #                 dict_wrap=dict_wrap,
            #                 tf_skl_shaft_distance=tf_skl_shaft_distance, 
            #                 ) 
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {dend_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 










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
                        tf_train=True,
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
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

            curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
            dend.append(pid.dend)     
            if tf_train:
                rhs.append(tf.cast(pid.get_pinn_rhs(pre_portion=pre_portion), dtype=DTYPE))

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
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter

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
                                tf_train=True,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                                ) 
        ls =[i for i in ls if i in range(len(curv))]
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
            
        print(f"Using device: {device}------------------ ----------------------------------------------")
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
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
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


 






    def model_shap(self, 
                        dend_names_ls=[0,1],
                        n_shap=25,  
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
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
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
        import shap  

        # feat_name=[] 
        # for  mm in base_features_list:
        #     feat_name.append(mm)
        # for _,name_inten in list_features: 
        #     feat_name.append(name_inten.removesuffix('.txt') )


 
 
                
        rhs_name='rhs_name'
        n_col=  len(self.model_dir_path[pre_portion][rhs_name]) 
        # print('-------------',n_col,model_dir)

        if  model_type.startswith(('cml','ml',)):
            with open(model_dir, 'rb') as f:
                model = pickle.load(f)  

        else:
            model_type_l=model_type
            if model_type.startswith(('gcn',)):
                from dend_fun_0.help_gcn_one_hot import model_choice
                adj_tf=True 
            elif model_type.startswith('dnn'):
                from dend_fun_0.help_dnn_one_hot import model_choice 
                adj_tf=False 
            elif model_type.startswith(('cnn','vol',)):
                from dend_fun_0.help_vol_one_hot import model_choice 
                mmnn=['vol']
                mmnn.extend(model_type.split('_')[1:])
                model_type_l='_'.join(mmnn)
                print('[[[[[[[]]]]]]]',model_type_l) 
                adj_tf=False
            mchoice = model_choice()
            custom = mchoice.get_custom_objects(model_type_l) 
            model = load_model(model_dir, custom_objects=custom)


        feat_name=[] 
        for  mm in base_features_list:
            feat_name.append(mm)
        for _,name_inten in list_features: 
            feat_name.append(name_inten.removesuffix('.txt') )


 


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
        time_start = time.time()
        rhs_name='rhs_name'
        dend_names = dend_names or [self.dend_names[jj] for jj in dend_names_ls ]
        base_features=[]
        for  index,dend_name  in enumerate( dend_names):
            if index>2:
                break
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
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                        )   
            pid.save_pinn_data() 
            pid.get_dend_data()
            con=0
            parami=dict(   
                        dend_names=[dend_name],
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
                        tf_train=False,
                        kmean_n_run=kmean_n_run,
                        kmean_max_iter=kmean_max_iter,
                        param_dic=param_dic,
                        ) 

            if  model_type.startswith(('cml','ml',)):
                curv,_,_,_,_=self.get_train_input_cml(**parami)  
                curvd=curv[0]
            else:
                if model_type.startswith(('gcn',)):
                    curv,_,_,_,_=self.get_train_input_gcn(**parami) 
                    curvd=curv[0] 
                elif model_type.startswith('dnn'):
                    curv,_,_,_,_=self.get_train_input_dnn(**parami) 
                    curvd=curv[0]  
                elif model_type.startswith(('cnn','vol',)):
                    curv,rhs,mskls,spine_indexs,idx_originals,weight_mask,rhshr  =self.get_train_input_volume(**parami)  
                    curvd=curv[0].numpy()
            base_features.append(curvd)
            # rhs0=model(curv[0]).numpy() [0] 
        # print([vd.shape for vd in base_features])
        n_shap = n_shap or min([150, base_features[1].shape[0]])
        niuu = min([base_features[0].shape[0] // 100, base_features[0].shape[0]]) 
        explain = shap.Explainer(model.predict, base_features[0][:niuu, :]) 
        shap_values = explain(base_features[1][:n_shap, :]) 
        raw_values = shap_values.values 

        '''
        n_shap = n_shap or min([150, base_features[1].shape[0]])
        niuu = min([base_features[0].shape[0] // 100, base_features[0].shape[0]])

        background = base_features[0][:niuu, :]

        masker = shap.maskers.Independent(background)
        explain = shap.Explainer(model.predict, masker)

        shap_values = explain(base_features[0][:n_shap, :])
        raw_values = shap_values.values
'''



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







