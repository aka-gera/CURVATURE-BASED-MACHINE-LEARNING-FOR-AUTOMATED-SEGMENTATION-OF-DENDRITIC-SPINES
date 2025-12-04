 
import os
import shutil
import stat

def remove_directory(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.chmod(file_path, stat.S_IWRITE)   
            os.remove(file_path) 
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.chmod(dir_path, stat.S_IWRITE)
            os.rmdir(dir_path) 
    shutil.rmtree(path, ignore_errors=True)  

def assign_if_none(self, **kwargs):
    for arg, value in kwargs.items():
        if value is None: 
            value = getattr(self, arg, None) 
        setattr(self, arg, value)
  
 
class get_name:
    def __init__(self): 
        self.txt_vertices_0 = 'vertices_0.txt'
        self.txt_vertices_1 = 'vertices_1.txt'
        self.txt_faces = 'faces_1.txt' 
        self.name_count = 'count'
        self.name_spine = 'spine'
        self.name_shaft = 'shaft'
        self.name_head = 'head'
        self.name_neck = 'neck'
        self.name_spine_group = 'spine_group'
        self.name_dend = 'dend'
        self.name_index = 'index'
        self.name_faces = 'faces'
        self.name_intensity = 'intensity'
        self.name_count = 'count'
        self.name_centroid = 'centroid'
        self.name_centroid_curve = 'centroid_curve_vert'
        self.name_centr = 'centr'
        self.name_shaft_iou = 'shaft_iou'
        self.name_center='center'
        self.name_length='length'
        self.txt_intensity_spines_segment='intensity_spines_segment.txt'
        self.txt_dend_org_mean='dend_init_mean.txt'
        self.txt_dend_org_gauss='dend_init_gauss.txt'   


        self.name_spine_new =f'{self.name_spine}_new'
        self.txt_spine_new =f'{self.name_spine}_new.txt'
        self.name_spine_new_index=f'{self.name_spine_new}_{self.name_index}'
        self.name_spine_new_face=f'{self.name_spine_new}_{self.name_faces}'

        self.txt_spine_new_intensity=f'{self.name_spine_new}_{self.name_intensity}.txt'

        self.txt_dend_smooth_mean='dend_smooth_mean.txt'
        self.txt_dend_smooth_gauss='dend_smooth_gauss.txt'

        self.txt_spine_intensity=f'{self.name_spine}_{self.name_intensity}.txt'
        self.txt_head_intensity= f'{self.name_head}_{self.name_intensity}.txt'
        self.txt_neck_intensity= f'{self.name_neck}_{self.name_intensity}.txt'
        self.txt_spine_volume='spine_volume.txt'

        self.name_spine_metric=f'{self.name_spine}_metric'
        self.name_head_metric=f'{self.name_head}_metric'
        self.name_neck_metric=f'{self.name_neck}_metric'
        self.name_shaft_metric=f'{self.name_shaft}_metric'

        self.txt_spine_metric=f'{self.name_spine_metric}.txt'
        self.txt_head_metric=f'{self.name_head_metric}.txt'
        self.txt_neck_metric=f'{self.name_neck_metric}.txt'
        self.txt_shaft_metric=f'{self.name_shaft_metric}.txt'
        self.txt_metric=f'metric.txt'

        self.txt_spine_center_length=f'{self.name_spine}_{self.name_center}_{self.name_length}.txt'
        self.txt_spine_shaft_length=f'{self.name_spine}_{self.name_shaft}_{self.name_length}.txt'
        self.txt_shaft_init_index=f'{self.name_shaft}_init_{self.name_index}.txt' 
        self.txt_shaft_init_faces=f'{self.name_shaft}_init_{self.name_faces}.txt'
        self.txt_shaft_init_index_unique=f'{self.name_shaft}_init_{self.name_index}_unique.txt'
        # Indexed names
        self.name_spine_index = f'{self.name_spine}_{self.name_index}'
        self.name_shaft_index = f'{self.name_shaft}_{self.name_index}'
        self.name_head_index = f'{self.name_head}_{self.name_index}'
        self.name_neck_index = f'{self.name_neck}_{self.name_index}'
        self.name_spine_group_index = f'{self.name_spine_group}_{self.name_index}'

        # Face names
        self.name_spine_faces = f'{self.name_spine}_{self.name_faces}'
        self.name_shaft_faces = f'{self.name_shaft}_{self.name_faces}'
        self.name_head_faces = f'{self.name_head}_{self.name_faces}'
        self.name_neck_faces = f'{self.name_neck}_{self.name_faces}'
        self.name_spine_group_faces = f'{self.name_spine_group}_{self.name_faces}'

        # Unique indices
        self.name_spine_group_index_unique = f'{self.name_spine_group_index}_unique'
        self.name_index_unique = f'{self.name_index}_unique'
        self.name_shaft_index_unique = f'{self.name_shaft_index}_unique'
        self.name_spine_group_faces_unique = f'{self.name_spine_group_faces}_unique'
        self.name_spine_index_unique = f'{self.name_spine_index}_unique'
        self.name_head_index_unique = f'{self.name_head_index}_unique'
        self.name_neck_index_unique = f'{self.name_neck_index}_unique'
        self.txt_shaft_index_unique = f'{self.name_shaft_index}_unique.txt'
        # Centroid-related names
        self.name_spine_centr = f'{self.name_spine}_{self.name_centr}'
        self.name_spine_centroid = f'{self.name_spine}_{self.name_centroid}'
        self.name_spine_centroid_curve = f'{self.name_spine}_{self.name_centroid_curve}'
        self.name_spine_count = f'{self.name_spine}_count'
        self.name_head_count = f'{self.name_head}_count'
        self.name_neck_count = f'{self.name_neck}_count'
        # File names
        self.txt_spine_count = f'{self.name_spine}_count.txt'
        self.txt_head_count = f'{self.name_head}_count.txt'
        self.txt_neck_count = f'{self.name_neck}_count.txt'
        self.txt_count = f'{self.name_count}.txt'
        self.txt_shaft_iou = f'{self.name_shaft_iou}.txt'
        self.txt_shaft_index = f'{self.name_shaft_index}.txt'
        self.txt_shaft_faces = f'{self.name_shaft_faces}.txt'
        self.txt_shaft_index_unique = f'{self.name_shaft_index_unique}.txt'
        self.txt_shaft_vertices_center = f'{self.name_shaft}_vertices_center.txt'
        self.txt_shaft_vertices_center_to_vertices_length = f'{self.name_shaft}_vertices_center_to_vertices_length.txt'
        self.txt_spine_count=f'{self.name_spine}_count.txt'
        self.txt_head_count=f'{self.name_head}_count.txt'
        self.txt_neck_count=f'{self.name_neck}_count.txt'



        self.txt_spine_iou='iou.txt'

        self.name_spine_div =f'{self.name_spine}_div'
        self.txt_spine_div_intensity  =f'{self.name_spine}_{self.name_intensity}_div.txt'
        self.txt_spine_new_intensity = f'{self.name_spine}_new_intensity.txt'
        self.txt_shaft_intensity = f'{self.name_shaft}_intensity.txt' 
        self.txt_spine_count_name =f'{self.name_spine_count}_name.txt'
        self.txt_shaft_vertices_center='shaft_vertices_center.txt'
        self.txt_shaft_vertices_center_to_vertices_length='shaft_vertices_center_to_vertices_length.txt'

        self.name_vcv_length = 'vcv_length'
        self.txt_dend_vcv_length=f'{self.name_dend}_{self.name_vcv_length}.txt'
        self.name_spine_vcv_length = f'{self.name_spine}_{self.name_vcv_length}'
        self.name_head_vcv_length = f'{self.name_head}_{self.name_vcv_length}'
        self.name_neck_vcv_length = f'{self.name_neck}_{self.name_vcv_length}'
        self.txt_shaft_vcv_length = f'{self.name_shaft}_{self.name_vcv_length}.txt' 
        self.txt_shaft_vcv_length_improved=f'shaft_vcv_length_improved.txt'
 
        self.txt_gauss_curv_init='gauss_curv_init.txt'
        self.txt_mean_curv_init='mean_curv_init.txt'
        self.txt_gauss_curv_smooth='gauss_curv_smooth.txt'
        self.txt_mean_curv_smooth='mean_curv_smooth.txt'
        self.txt_faces_class_faces='faces_class_faces.txt'
        self.txt_vertex_neighbor='vertex_neighbor.txt'
        self.txt_skl_distance='skl_distance.txt'
        self.txt_skl_vectices='skl_vectices.txt'
        self.txt_skl_index='skl_index.txt'

        self.txt_skl_distance_org='skl_distance_org.txt'
        self.txt_skl_vectices_org='skl_vectices_org.txt'
        self.txt_skl_index_org='skl_index_org.txt'

        self.txt_skl_distance_true='skl_distance_true.txt'
        self.txt_skl_vectices_true='skl_vectices_true.txt'
        self.txt_skl_index_true='skl_index_true.txt'
        
        
        self.txt_skl_distance_con='skl_distance_con.txt'
        self.txt_skl_vectices_con='skl_vectices_con.txt'
        self.txt_skl_index_con='skl_index_con.txt'

        self.txt_skl_shaft_distance='skl_shaft_distance.txt'
        self.txt_skl_shaft_vectices='skl_shaft_vectices.txt'

        self.txt_gauss_sq_curv_smooth='gauss_sq_curv_smooth.txt'
        self.txt_mean_sq_curv_smooth='mean_sq_curv_smooth.txt' 
        self.txt_gauss_sq_curv_init='gauss_sq_curv_init.txt'
        self.txt_mean_sq_curv_init='mean_sq_curv_init.txt' 


        self.pkl_vertex_neighbor='vertex_neighbor.pkl'
        self.pkl_mp='mp.pkl'

        gff=['skl', ]#,'mean_qd''mean_gauss','skl']
        gffindex=['skl',]  #22-23
 
        gfff=['curv_k','curv_v','curv_kv' ,'curv_k2','curv_v2','curv_kv22'] #23-28
        gfffindex=['x','v','xv' ,'x2','v2','x2v2']  

        gf=['gauss','mean','gauss_sq','mean_sq','gauss_qd','mean_qd']# 0-5   ,'mean_qd''mean_gauss','skl']
        gfindex=['g','m','g2','m2','g4','m4' ]


        gfi=['igauss','imean','igauss_sq','imean_sq']#29-32    ,'mean_qd''mean_gauss','skl']
        gfiindex=['ig','im','ig2','im2' ]
         
        self.kmean_list=[100,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # 6-21
        gf2=[f'kmean_{rf}' for rf in self.kmean_list]
        gfindex2=[f'k{rf}' for rf in self.kmean_list] 
        gf22=[f'kmean_mean_{rf}' for rf in self.kmean_list]# 33-48
        gfindex22=[f'n{rf}' for rf in self.kmean_list] 
        gfg=[]
        gkg=[]
        gfg.extend(gf)
        gfg.extend(gf2)
        gfg.extend(gff)
        gfg.extend(gfff)
        gfg.extend(gfi)
        gfg.extend(gf22)
        gkg.extend(gfindex)
        gkg.extend(gfindex2)
        gkg.extend(gffindex)
        gkg.extend(gfffindex)
        gkg.extend(gfiindex)
        gkg.extend(gfindex22)
        self.base_features_dict= {}
        for nam,nm in zip(gfg,gkg):
            self.base_features_dict[nam]={}
            self.base_features_dict[nam]['index']=nm

        self.inten_pinn=['shaft_vcv_length','skl_shaft_distance','spine_shaft_length','pca_1_norm','pca_2_norm','pca_3_norm','division','spine_intensity_division']
        self.inten_pinn_name=['vcv','shkl','sh','p1','p2','p3','di','sid']
        self.inten_pinn_dic={key:val for key,val in zip(self.inten_pinn,self.inten_pinn_name)}
        
        self.metrics_keys=['spine_vol','head_vol','neck_vol','spine_area','head_area','neck_area','head_diameter','neck_diameter','head_length','neck_length','spine_length']
 
        self.metrics={}
        for val in self.metrics_keys:
            self.metrics[val]={}

        self.metrics_combine={'diam_head_neck_length':['head_diameter','neck_diameter','spine_length'],
                              'vol_head_neck_spine':['head_vol','neck_vol','spine_vol'],
                              'area_head_neck_spine':['head_area','neck_area','spine_area'],
                              'length_head_neck_spine':['head_length','neck_length','spine_length'],
                              }

        # self.file_model_train=['spine','head_neck','shaft']
        # self.path_heads=['pinn','rpinn','ML','true']
        self.pre_portions=['head_neck','spine']
        
        self.inten_file_model_train_spine_iou=[ 'iou_spine_sh','iou_spine_sp' ]
        self.inten_file_model_spine_iou=['model_sp_iou'] 
        self.inten_file_model_train_spine_loss=['loss_spine']
        self.inten_file_model_spine_loss=['model_sp_loss']

        self.inten_file_model_train_iou=['iou_head_neck_hd','iou_head_neck_nk','iou_head_neck_sh' ]
        self.inten_file_model_head_neck_iou=['model_hn_iou'] 
        self.inten_file_model_train_loss=['loss_head_neck']
        self.inten_file_model_head_neck_loss=['model_hn_loss']
 
        self.inten_file_model_train_shap=[ 'shap']
        self.inten_file_model_shap=['model_shap']

        self.inten_file_model_head_neck=[] 
        self.inten_file_model_head_neck.extend(self.inten_file_model_spine_iou)
        self.inten_file_model_head_neck.extend(self.inten_file_model_spine_loss)
        self.inten_file_model_head_neck.extend(self.inten_file_model_shap)
        self.inten_pca=['pca_1_norm','pca_2_norm','pca_3_norm','volume','area','energy','division',]
        self.inten_file_sub=["path","spine_intensity",  "intensity_head_neck_segm",  'intensity_spines_segment','intensity_spines_segment_shaft',]
        self.inten_file_sub_name=["path","Segmentation","Head Neck Segm.", 'intensity_spines_segment','intensity_spines_segment_shaft', ] 
        self.inten_file=[] 
        self.inten_file_train=['skl_distance','skl_shaft_distance',  ]#'spine_shaft_length',  'skl Shaft Distance',  'Length sp. skl to sh. skl.',"gauss_curv_smooth",'Annotation','spine_annot',"mean_curv_smooth","gauss_curv_init","mean_curv_init","intensity_shaft_neck_head",'intensity_shaft_spine','intensity_1hot_shaft_spine', 'intensity_1hot_shaft_neck_head']
        self.dend_file=['vertices_head','vertices_neck','vertices_spine','faces_head','faces_neck','faces_spine', "vertices_1","vertices_0",'faces_0']  
        self.inttt=[]
        self.inttt.extend(self.inten_file_sub)  
        # self.inttt.extend(self.inten_pca)  
        self.inttt.extend(self.inten_file_model_head_neck)  
        self.inttt.extend(self.inten_file_train)
        self.inttt.extend(gfg) 
        self.path_file_sub={}
        for ty in self.inttt:
            self.path_file_sub[ty]={} 
        self.dropdown_options_style = {'color': 'white', 'background-color': 'gray'}
 


        
        pre_portion='spine'
        pre_portion='head_neck'

        pinn_dir_data= 'save' 
        pinn_dir_dest= 'save' 


        self.data_mode={}
        self.data_mode['pinn_dir_data_all']=[]
        self.data_mode['model_sufix_all']=[]
        self.data_mode['mode_id']=[] 



class get_model_name:
    def __init__(self,pre_opt,train_test,seg_dend,dest_head,pre_portion): 
        self.seg_dend,self.dest_head,self.pre_portion=seg_dend,dest_head,pre_portion
        self.pre_opt,self.train_test=pre_opt,train_test
        get_name.__init__(self) 
    def vals(self, 
            inten_pinn_index ,
            base_features_index ,
            pre_opt=None,
            train_test=None,
            seg_dend=None,
            dest_head=None,
            pre_portion=None): 
        seg_dend=seg_dend or self.seg_dend
        dest_head=dest_head or self.dest_head
        pre_portion=pre_portion or self.pre_portion 
        pre_opt=pre_opt or self.pre_opt
        train_test=train_test or self.train_test  
        path_head,model_suf,path='pinn','pre_gmg2m2','save'
        id_path=f'{path_head}_{model_suf}_{path}' 
        self.inten_pinn_name_tmp=[self.inten_pinn_name[val] for val in inten_pinn_index]
        self.inten_pinn_tmp={
                                val:{
                                    'name':f'{self.inten_pinn[val]}.txt',
                                    'id_path':id_path
                                  } 
                                for val in inten_pinn_index
                                }

        self.inten_pinn_path=id_path
        # self.model_name=  '' if len(inten_pinn_index)==0 else '_'.join([self.inten_pinn[hh] for hh in self.inten_pinn])
        self.model_name = '_'.join([self.inten_pinn_name[i] for i in inten_pinn_index])# if inten_pinn_index else ''
        self.base_features_list=[list(self.base_features_dict.keys())[hhh] for hhh in base_features_index]
        self.dest_sufix = f"{pre_opt}_{''.join([self.base_features_dict[hh]['index'] for hh in self.base_features_list])}_{self.model_name}" if inten_pinn_index else f"{pre_opt}_{''.join([self.base_features_dict[hh]['index'] for hh in self.base_features_list])}"
        self.mode_id=f'{train_test}_{seg_dend}_{pre_portion[:2]}_{dest_head}_{self.dest_sufix}' 
        self.dest_dir=f'{seg_dend}_{pre_portion[:2]}'  

nh=24
def get_configs():
    return {
            "DNN-1": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3],
                "inten_pinn_index": []
            },
            "mode1": {
                "pre_opt": "pre",
                "base_features_index": [0,2,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode2": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode3": {
                "pre_opt": "pre",
                "base_features_index": [0,2,4,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode4": {
                "pre_opt": "pre",
                "base_features_index": [0,2,4,7,11,15],
                "inten_pinn_index": []
            },
            "mode5": {
                "pre_opt": "pre",
                "base_features_index": [0,2,3,4,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode6": {
                "pre_opt": "opt",
                "base_features_index": [0,1,2,3],
                "inten_pinn_index": [0]
            },
            "mode7": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,22],
                "inten_pinn_index": []
            },
            "DNN-2": {
                "pre_opt": "opt",
                "base_features_index": [0,1,2,3],
                "inten_pinn_index": [1]
            },
            "mode9": {
                "pre_opt": "opt",
                "base_features_index": [0,2,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [1]
            },
            "mode10": {
                "pre_opt": "pre",
                "base_features_index": [0,2,3,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode11": {
                "pre_opt": "pre",
                "base_features_index": [23,24,25,26,27, 7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode12": {
                "pre_opt": "opt",
                "base_features_index": [23,24,25,26,27 ],
                "inten_pinn_index": [0]
            },
            "mode13": {
                "pre_opt": "opt",
                "base_features_index": [23,24,25,26,27, 7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [0]
            },
            "mode14": {
                "pre_opt": "pre",
                "base_features_index": [23,24,25,26,27 ],
                "inten_pinn_index": []
            },
            "mode15": {
                "pre_opt": "opt",
                "base_features_index": [0,2,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [0]
            },
            "mode16": {
                "pre_opt": "pre",
                "base_features_index": [1,23,24,25,26,27, 7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [0]
            },
            "mode17": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,22],
                "inten_pinn_index": []
            },
            "mode18": {
                "pre_opt": "pre",
                "base_features_index": [0,2,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode19": {
                "pre_opt": "pre",
                "base_features_index": [0,1,2,3,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode20": {
                "pre_opt": "pre",
                "base_features_index": [7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode21": {
                "pre_opt": "pre",
                "base_features_index": [29,31,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode22": {
                "pre_opt": "pre",
                "base_features_index": [0,2,29,31,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": []
            },
            "mode23": {
                "pre_opt": "opt",
                "base_features_index": [0,1,2,3,7,8,9,10,11,12,13,14,15],
                "inten_pinn_index": [1]
            },
            "DNN-3": {
                "pre_opt": "pre",
                "base_features_index": [27+i for i in [0-27,2-27,7,8,9,10,11,12,13,14,15]],
                "inten_pinn_index": []
            }, 
        }


class get_param:
    def __init__(self,  
                    hidden_layers=4, 
                    neurons_per_layer=100, 
                    n_col=2,
                    activation_init="relu",
                    activation_hidden="relu",
                    activation_last="sigmoid",
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
                    disp_infos=False,
                    thre_gauss=45,
                    thre_mean=15,
                    thre_gen=100,
                    weight_positive=0.5,
                    txt_save=False,
                    txt_save_pred=False,
                    DTYPE='float32',
                    txt_true_file=None,
                    zoom_thre=10,  
                    thre_target_number_of_triangles=None,
                    voxel_resolution=None,
                     ): 

        self.hidden_layers=hidden_layers
        self.neurons_per_layer=neurons_per_layer
        self.n_col=n_col
        self.activation_init=activation_init
        self.activation_hidden=activation_hidden
        self.activation_last=activation_last
        self.thre_target_number_of_triangles=thre_target_number_of_triangles
        self.voxel_resolution=voxel_resolution
                         
                         
                         
                         
        self.disp_infos=disp_infos
        self.radius_threshold=radius_threshold
        self.name_path_fin=name_path_fin
        self.thre_gauss=thre_gauss 
        self.thre_mean=thre_mean
        self.thre_gen=thre_gen
        self.weight_positive=weight_positive 
        self.zoom_thre=zoom_thre

        self.DTYPE=DTYPE
        self.txt_save=txt_save 
        self.txt_save_pred=txt_save_pred
        self.txt_true_file=txt_true_file

        self.dend_cla=None
        self.dend_saves=None 
 
        self.line_num_points_shaft= line_num_points_shaft 
        self.line_num_points_inter_shaft= line_num_points_inter_shaft 
        self.spline_smooth_shaft= spline_smooth_shaft
        self.numNeighbours=numNeighbours
        self.gauss_threshold=gauss_threshold
        self.size_threshold=size_threshold 
        self.zoom_threshold=zoom_threshold 
        self.cts=cts 
        self.stoppage=stoppage 
        self.zoom_threshold_min=zoom_threshold_min 
        self.zoom_threshold_max=zoom_threshold_max 
        self.spine_filter=spine_filter
        self.radius_threshold=radius_threshold
        vv_fin=[name_path_fin_save_index]
        if txt_save:
            if txt_save_pred: 
                vv=[ii for ii in range(self.cts)] 
                vv.extend(vv_fin)
                self.vv_cts=vv 
            else:
                self.vv_cts=vv_fin
        else:
            self.vv_cts=vv_fin
        self.name_path_fin_save=f'{name_path_fin}_{name_path_fin_save_index}' 
        self.name_path_fin_save_index=name_path_fin_save_index



class get_files(get_name,get_param):
    def __init__(self,
                    file_path_org,
                    model_sufix,
                    dend_data=None, 
                    dend_names=None,
                    dend_namess=None, 
                    data_studied=None, 
                    dend_path_inits=None,
                    name_spine_id=None,
                    name_head_id=None,
                    name_neck_id=None,
                    name_shaft_id=None,
                    size_threshold=10,
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
                    zoom_threshold_max=3, 
                    line_num_points_shaft=200,
                    line_num_points_inter_shaft=300, 
                    spline_smooth_shaft=1, 
                    disp_infos=False,
                    thre_gauss=45,
                    thre_mean=15,
                    thre_gen=1000,
                    weight_positive=0.5,
                    txt_save=False,
                    txt_save_pred=False,
                    DTYPE='float32',
                    txt_true_file=None,
                    path_train=None,
                    pre_portion=None,
                    pinn_dir_data=None,
                    path_file=None,  
                    pinn_dir_data_all=None,
                    model_sufix_all=None, 
                    list_features=None,
                    base_features_list=None,
                    metrics={},
                    model_type=None,
                    data_mode=None,
                    path_heads=None,
                    true_keys=None,
                    thre_target_number_of_triangles=None,
                    voxel_resolution=None,
                    obj_org_path=None,
                    obj_org_path_dict=None,
                    model_sufix_dic=None,
        ) : 
        print('get_file',thre_target_number_of_triangles)
        get_name.__init__(self)  
        get_param.__init__(self,
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
                        thre_gauss=thre_gauss ,
                        thre_mean=thre_mean,
                        thre_gen=thre_gen,
                        weight_positive=weight_positive, 
                        disp_infos=disp_infos,
                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                        voxel_resolution=voxel_resolution,
                        ) 



        self.thre_target_number_of_triangles=thre_target_number_of_triangles
        self.voxel_resolution=voxel_resolution
        self.model_type=model_type
        self.base_features_list=base_features_list
        self.list_features=list_features
        self.pinn_dir_data_all=pinn_dir_data_all
        self.model_sufix_all=model_sufix_all
        self.path_heads=path_heads
        self.true_keys=true_keys
        self.path_train=path_train
        self.path_file=path_file
        self.pinn_dir_data=pinn_dir_data
        self.pre_portion=pre_portion
        self.file_path_org=file_path_org
        self.dend_names=dend_names 
        self.dend_namess=dend_namess  
        self.dend_path_inits=dend_path_inits 
        self.dend_data=dend_data  
        if dend_data is not None:
            self.dend_names=dend_names if dend_names is not None else dend_data['dend_names']
            self.dend_namess=dend_namess if dend_namess is not None else dend_data['dend_namess']
            self.dend_path_inits=dend_path_inits if dend_path_inits is not None else dend_data['dend_path_inits']
            self.name_spine_id =name_spine_id if name_spine_id is not None else dend_data['name_spine_id']
            self.name_head_id =name_head_id if name_head_id is not None else dend_data['name_head_id']
            self.name_neck_id = name_neck_id if name_neck_id is not None else dend_data['name_neck_id']
            self.name_shaft_id = name_shaft_id if name_shaft_id is not None else dend_data['name_shaft_id']
            self.obj_org_path_dict = obj_org_path_dict if obj_org_path_dict is not None else dend_data['obj_org_path_dict']
            self.obj_org_path = obj_org_path if obj_org_path is not None else dend_data['obj_org_path']
            self.part_id_mapping = {
                self.name_spine: self.name_spine_id,
                self.name_head: self.name_head_id,
                self.name_neck: self.name_neck_id,
                self.name_shaft: self.name_shaft_id
            }

        if data_mode is not None:
            self.path_train=path_train if path_train is not None else data_mode['path_train']
            self.pre_portion=pre_portion if pre_portion is not None else data_mode['pre_portion']
            self.pinn_dir_data=pinn_dir_data if pinn_dir_data is not None else data_mode['pinn_dir_data']
            self.list_features=list_features if list_features is not None else data_mode['list_features']
            self.base_features_list=base_features_list if base_features_list is not None else data_mode['base_features_list']
        
        self.data_studied=data_studied
        self.txt_true_file=txt_true_file
        self.model_sufix=model_sufix
        self.data_mode=data_mode
        self.obj_org_path=obj_org_path or self.obj_org_path
        self.obj_org_path_dict=obj_org_path_dict or self.obj_org_path_dict
        self.model_sufix_dic=model_sufix_dic
        # self.file_path_model=self.file_path_save =os.path.join(file_path_org,'pinn')
        # os.makedirs(self.file_path_model, exist_ok=True) 
        self.file_path_model_data= os.path.join(file_path_org,'data')
        os.makedirs(self.file_path_model_data, exist_ok=True) 
 

        self.pkl_path_model_data = os.path.join(self.file_path_model_data,fr'dend_{self.data_studied}_data.pkl')
        self.dash_pages_path=os.path.join(file_path_org,'app','pages')
  
        pass


        self.metrics=metrics
        for val in self.metrics_keys:
            self.metrics[val]={} 
        self.pinn_dir_data_all=pinn_dir_data_all= list(set(pinn_dir_data_all if pinn_dir_data_all is not None else self.pinn_dir_data_all ))
        self.model_sufix_all= list(set(model_sufix_all if model_sufix_all is not None else self.model_sufix_all  ))
        self.path_heads= list(set(path_heads if path_heads is not None else self.path_heads  )) 

        self.dropdown_true_keys_option=[]
        for intt in self.true_keys:
            self.dropdown_true_keys_option.append({'label': intt,          'value': intt,           'style': self.dropdown_options_style})   

 
        self.dropdown_path_head_option = []
        for nam in self.path_heads:
            self.dropdown_path_head_option.append({'label': nam.upper(), 'value': nam, 'style': self.dropdown_options_style})
        self.dropdown_path_head_option.append({'label': f'Annotation',   'value': 'true', 'style': self.dropdown_options_style})

        self.dropdown_model_suf_option=[]
        for intt in self.model_sufix_all:
            self.dropdown_model_suf_option.append({'label': self.model_sufix_dic[intt],          'value': intt,           'style': self.dropdown_options_style})   

 

        # self.dropdown_path_option=[]
        # for intt in self.pinn_dir_data_all:
        #     self.dropdown_path_option.append({'label': intt,          'value': intt,           'style': self.dropdown_options_style})   


        self.dropdown_path_option=[]
        for intt in ['save',]:
            self.dropdown_path_option.append({'label': intt,          'value': intt,           'style': self.dropdown_options_style})   






    def assign_if_none(self, **kwargs):
        for arg, value in kwargs.items():
            if value is None: 
                value = getattr(self, arg, None) 
            setattr(self, arg, value)

    def get_model_opt_name(self,  
                      model_type ,
                    file_path_model=None,
                    file_path_org=None, 
                    model_sufix=None,  
                     name_path_fin=None, ):   
        file_path_org = file_path_org or self.file_path_org
        name_path_fin = name_path_fin or self.name_path_fin
        self.name_path_fin = name_path_fin
        model_sufix = model_sufix or self.model_sufix 

  
        if file_path_model is None:
            file_path_model = os.path.join(file_path_org , 'model',model_type,model_sufix )
            os.makedirs(file_path_model, exist_ok=True)
        self.file_path_model = file_path_model


        self.iou_save_dir_all = [os.path.join(self.file_path_model, f'iou_{ii}.txt') for ii in range(3)]

        self.shap_dir = os.path.join(self.file_path_model, 'shap.csv')
        self.df_metric_algorithms_dir = os.path.join(self.file_path_model, 'df_metric_algorithms.csv')

        mo='pkl' if model_type=='ML' else 'keras'
        self.model_save_dir = os.path.join(self.file_path_model, f'model.{mo}')
        self.model_spine_save_dir = os.path.join(self.file_path_model, f'model_spine.{mo}')
        self.model_shaft_save_dir = os.path.join(self.file_path_model, f'model_shaft.{mo}')
        self.model_head_neck_save_dir = os.path.join(self.file_path_model, f'model_head_neck.{mo}')


        
        self.shaft_pred_dir = os.path.join(self.file_path_model, 'shaft.pkl')
        self.loss_save_dir  = os.path.join(self.file_path_model, 'loss.txt')
        self.iou_save_dir   = os.path.join(self.file_path_model, 'iou.txt')
        self.index_save_dir  = os.path.join(self.file_path_model, 'index.txt') 

        self.loss_spine_save_dir  = os.path.join(self.file_path_model, 'loss_spine.txt') 
        self.iou_spine_sp_save_dir   = os.path.join(self.file_path_model, 'iou_spine_sp.txt')
        self.iou_spine_sh_save_dir   = os.path.join(self.file_path_model, 'iou_spine_sh.txt')
        self.index_spine_save_dir  = os.path.join(self.file_path_model, 'index_spine.txt') 

        self.loss_shaft_save_dir  = os.path.join(self.file_path_model, 'loss_shaft.txt') 
        self.index_shaft_save_dir  = os.path.join(self.file_path_model, 'index_shaft.txt')  
        
        self.loss_head_neck_save_dir  = os.path.join(self.file_path_model, 'loss_head_neck.txt')
        self.iou_head_neck_hd_save_dir   = os.path.join(self.file_path_model, 'iou_head_neck_hd.txt')
        self.iou_head_neck_nk_save_dir   = os.path.join(self.file_path_model, 'iou_head_neck_nk.txt')
        self.iou_head_neck_sh_save_dir   = os.path.join(self.file_path_model, 'iou_head_neck_sh.txt')
        self.index_head_neck_save_dir  = os.path.join(self.file_path_model, 'index_head_neck.txt') 
                
                
        self.model_dir_path= {
            'spine': {
                'loss': self.loss_spine_save_dir, 
                'iou':{
                        1:self.iou_spine_sp_save_dir,
                        0:self.iou_spine_sh_save_dir,
                       } , 
                'index_save': self.index_spine_save_dir, 
                'model': self.model_spine_save_dir,
                'rhs_name':['shaft_pre_sp','spine_pre_sp'],
            },
            'head_neck': {
                'loss': self.loss_head_neck_save_dir,
                'iou':{
                        2: self.iou_head_neck_hd_save_dir,
                        1: self.iou_head_neck_nk_save_dir,
                        0: self.iou_head_neck_sh_save_dir, 
                        },
                'index_save': self.index_head_neck_save_dir, 
                'model': self.model_head_neck_save_dir,
                'rhs_name':['shaft_pre','neck_pre','head_pre'],
            },
            'shaft': {
                'loss': self.loss_shaft_save_dir, 
                'model': self.model_shaft_save_dir,
                'rhs_name':['shaft_pre'],
            },
            'default': {
                'loss': self.loss_save_dir,
                'iou': self.iou_save_dir,
                'model': self.model_save_dir,
                'rhs_name':['spine_pre'],
            }
        }
  

    def get_path(self, *names, name='pinn'):
        base = os.path.join(self.dend_path, name)
        
        if name.startswith( 'pinn'):
            if names:
                base = os.path.join(base, f'{names[0]}', *names[1:])
            else:
                base = os.path.join(base, f'{self.name_path_fin}')
        elif name == 'result':
            base = os.path.join(self.file_path_org, 'data',)
            if names: 
                if names[0]=='true': 
                    base = os.path.join(base,self.dend_path_inits[0],  name, f'{names[0]}')  
                else:  
                    base = os.path.join(base,self.dend_path_inits[0],  name, f'{names[0]}',    *names[1:])  

        elif name.startswith('true'): 
            base = os.path.join(self.dend_path_org, name) 
        elif name.startswith('resized'): 
            base = os.path.join(self.dend_path_resized, names[0])
        else:
            if names:
                base = os.path.join(base, f'{names[0]}', *names[1:])
            else:
                base = os.path.join(base, f'{self.name_path_fin}')

        return base
    

    def get_paths(self, names,name=None ):
        name=name if name is not None else names[0]
        return self.get_path(*names[1:],name=name)
 


    def get_dend_name(self, index,
                        dend_names=None,
                        dend_namess=None,
                        file_path_org=None, 
                        data_studied=None,
                        file_path_model_data=None,
                        name_path_fin=None,
                        name_path_fin_save=None, 
                        dend_path_inits=None,
                        model_sufix=None,
                        pinn_dir_data=None, 
                        pinn_dir_data_all=None,
                        model_sufix_all=None, 
                        path_heads =None,
                        model_type=None,
                        obj_org_path=None,

        ):  
        obj_org_path = obj_org_path or self.obj_org_path
        self.obj_org_path=obj_org_path

        model_type=model_type or self.model_type
        if file_path_org is None:
            file_path_org = self.file_path_org 
        self.model_sufix=model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type)
        dend_names = dend_names or self.dend_names
        file_path_model_data = file_path_model_data or self.file_path_model_data
        dend_namess = dend_namess or self.dend_namess
        name_path_fin = name_path_fin or self.name_path_fin
        dend_path_inits = dend_path_inits or self.dend_path_inits
        self.data_studied = data_studied or self.data_studied
        name_path_fin_save = name_path_fin_save or self.name_path_fin_save
        pinn_dir_data=pinn_dir_data or self.pinn_dir_data
 
        self.dend_path_init =dend_path_inits[index]
        # Assigning to self
        self.file_path_org_init= os.path.join(file_path_org, 'data', self.dend_path_init,'data')
        self.name_path_fin = name_path_fin
        self.name_path_fin_save = name_path_fin_save

        
        self.last_name,self.sp_name_shaft=dend_namess[index][0],dend_namess[index][1]  
 
        self.dend_name = f'{dend_names[index]}' 


        print('self.obj_org_path, self.dend_name',self.obj_org_path, self.file_path_org_init,self.dend_name)
        self.dend_path_original =self.dend_path_original_m = os.path.join(self.file_path_org_init, self.dend_name, 'data_org') 
        self.dend_path_original_new = os.path.join(self.obj_org_path, self.dend_name, 'data_org')  
        self.dend_path_original_new_smooth = os.path.join(self.obj_org_path, self.dend_name, 'data_smooth')  
        self.dend_path_org_new = os.path.join(self.obj_org_path, self.dend_name, 'data')   
        os.makedirs(self.dend_path_original_new_smooth, exist_ok=True)





        self.dend_path_org_resized=os.path.join(f'{self.obj_org_path}_resized', self.dend_name, 'data_org')
        self.dend_path_org_smooth_resized=os.path.join(f'{self.obj_org_path}_resized', self.dend_name, 'data_smooth')      

        self.dend_path_org=self.dend_path = os.path.join(self.file_path_org_init, self.dend_name )
        self.dend_path_resized = os.path.join(file_path_org, 'data', f'{self.dend_path_init}_resized','data',f'{self.dend_name}')
        os.makedirs(self.dend_path, exist_ok=True)
        self.file_path = os.path.join(self.dend_path, 'data')
        os.makedirs(self.file_path, exist_ok=True)

        
        self.file_path_resized= os.path.join(self.dend_path_resized, 'data' )  
        self.file_path_feat_resized= os.path.join(self.file_path_resized, 'feat' ) 
        self.file_path_feat = os.path.join(self.file_path, 'feat')
        os.makedirs(self.file_path_feat, exist_ok=True)
 
        self.vertices_1_path=os.path.join(self.file_path,  self.txt_vertices_1)
        self.vertices_0_path=os.path.join(self.file_path,   self.txt_vertices_0)
        self.faces_path=os.path.join(self.file_path,   self.txt_faces)   
        self.dend_first_name=self.dend_namess[index][1]
  


        self.path_file={} 
 
        pinn_dir_data_all= pinn_dir_data_all or self.pinn_dir_data_all 
        model_sufix_all= model_sufix_all or self.model_sufix_all 
        path_heads = path_heads or self.path_heads
        path_headss=path_heads
        path_headss.append('true') 
        for pa in pinn_dir_data_all:
            if pa is not None:
                for model_sufi in model_sufix_all:
                    for path_head in path_headss:
                        modd=os.path.join(self.file_path_org, 'model',path_head,model_sufi)
                        key=f'{path_head}_{model_sufi}_{pa}'
                        self.path_file_sub[self.inten_file_sub[0]][key] =self.get_paths([path_head, model_sufi, pa])
                        self.path_file[key]=self.get_paths([path_head, model_sufi, pa])
                        os.makedirs(self.path_file[key], exist_ok=True) 
                        for tyy in self.inten_file_sub[1:]: 
                            self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 

                        # for tyy in self.inten_pca: 
                        #     self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 


                        for tyy in self.inten_file: 
                            self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 

                        for tyy in self.inten_file_train: 
                            self.path_file_sub[tyy][key]= os.path.join(self.file_path_feat,f'{tyy}.txt') 

                        # self.file_model_train=['spine','head_neck','shaft']
                        # self.inten_file_model_train=['loss','iou'] 
                        '''
                        tyy =self.inten_file_model_head_neck_loss[0]
                        tyyy=self.inten_file_model_train_loss[0]
                        self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.txt')

                        tyy =self.inten_file_model_head_neck_iou[0]
                        self.path_file_sub[tyy][key]={}
                        for tyyy in  self.inten_file_model_train_iou:  
                                self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')

'''
                        tyy =self.inten_file_model_spine_loss[0]
                        tyyy=self.inten_file_model_train_spine_loss[0]
                        self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.txt')

                        tyy =self.inten_file_model_spine_iou[0]
                        self.path_file_sub[tyy][key]={}
                        for tyyy in  self.inten_file_model_train_spine_iou:  
                                self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')


                        tyy =self.inten_file_model_shap[0]
                        tyyy=self.inten_file_model_train_shap[0]
                        self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.csv')
 

                        for tyy in self.base_features_dict.keys(): 
                            self.path_file_sub[tyy][key]= os.path.join(self.file_path_feat,f'{tyy}.txt') 

 
        # if len(self.obj_org_path_dict)>0:
        #     for ii,(keys,val) in enumerate(self.obj_org_path_dict.items()): 
        #         path_head ,pa,model_sufi=f'true_{ii}','save',f'save' 
        #         key=f'{path_head}_{model_sufi}_{pa}' 
        #         pasd=self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([path_head, model_sufi, pa])
        #         remove_directory(self.path_file[key] )
        path_headss=[]
        self.dend_path_original_mm={ky:{} for ky in ['keys','dir']}  
        if len(self.obj_org_path_dict)>0:
            for ii,(keys,val) in enumerate(self.obj_org_path_dict.items()): 
                path_head ,pa,model_sufi=keys,'save',f'save' 
                key=f'{path_head}_{model_sufi}_{pa}'
                path_headss.extend(path_head) 
                self.dend_path_original_mm['dir'][key]=os.path.join(val, self.dend_name, 'data_org')
                self.dend_path_original_mm['keys'][key]=keys
                pasd=self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([path_head, model_sufi, pa])
                os.makedirs(self.path_file[key], exist_ok=True)
 
                path_head ,pa,model_sufi=keys,'save',f'save' 
                key=f'resized_{path_head}_{model_sufi}_{pa}' 
                pasd=self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths(['resized',path_head, model_sufi, pa],name='resized')
        # else: 
        #     path_head ,pa,model_sufi='true','save','save' 
        #     key=f'{path_head}_{model_sufi}_{pa}'
        #     path_headss.extend(path_head) 
        #     self.dend_path_original_mm['dir'][key]=os.path.join(self.dend_path_org, 'data_org')
        #     pasd=self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([path_head, model_sufi, pa])
        #     os.makedirs(self.path_file[key], exist_ok=True) 

        for keyss,path_head in  self.dend_path_original_mm['keys'].items():
            pasd= self.path_file[keyss]  
            for pa in pinn_dir_data_all:
                if pa is not None:
                    for model_sufi in model_sufix_all: 
                            modd=os.path.join(self.file_path_org, 'model',path_head,model_sufi)
                            key=f'{path_head}_{model_sufi}_{pa}' 
                            self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=pasd
                            # os.makedirs(self.path_file[key], exist_ok=True) 
                            for tyy in self.inten_file_sub[1:]: 
                                self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 

                            # for tyy in self.inten_pca: 
                            #     self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 


                            for tyy in self.inten_file: 
                                self.path_file_sub[tyy][key]= os.path.join(self.path_file[key],f'{tyy}.txt') 

                            for tyy in self.inten_file_train: 
                                self.path_file_sub[tyy][key]= os.path.join(self.file_path_feat,f'{tyy}.txt') 

                            # self.file_model_train=['spine','head_neck','shaft']
                            # self.inten_file_model_train=['loss','iou'] 
                            '''
                            tyy =self.inten_file_model_head_neck_loss[0]
                            tyyy=self.inten_file_model_train_loss[0]
                            self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.txt')

                            tyy =self.inten_file_model_head_neck_iou[0]
                            self.path_file_sub[tyy][key]={}
                            for tyyy in  self.inten_file_model_train_iou:  
                                    self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')
'''

                            tyy =self.inten_file_model_spine_loss[0]
                            tyyy=self.inten_file_model_train_spine_loss[0]
                            self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.txt')

                            tyy =self.inten_file_model_spine_iou[0]
                            self.path_file_sub[tyy][key]={}
                            for tyyy in  self.inten_file_model_train_spine_iou:  
                                    self.path_file_sub[tyy][key][tyyy]= os.path.join(modd,f'{tyyy}.txt')


                            tyy =self.inten_file_model_shap[0]
                            tyyy=self.inten_file_model_train_shap[0]
                            self.path_file_sub[tyy][key]= os.path.join(modd,f'{tyyy}.csv')

         
        for pa in pinn_dir_data_all:
            if pa is not None:
                for model_sufi in model_sufix_all:
                    for res in ['result',]:
                        for path_head in path_heads:
                            key=f'{res}_{path_head}_{model_sufi}_{pa}' 
                            self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([res,path_head, model_sufi, pa])
                            os.makedirs(self.path_file[key], exist_ok=True)  

        for nn in 'true_0':
            remove_directory(os.path.join(self.dend_path_org,  nn))
            remove_directory(os.path.join(self.file_path_org, 'data', self.dend_path_init,'result',nn))

        key=f'result_true'
        self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths(['result', 'true_0'])
        os.makedirs(self.path_file[key], exist_ok=True) 

    
        key=f'result_appr'
        self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths(['result', 'appr'])
        os.makedirs(self.path_file[key], exist_ok=True) 
 



    def get_dash_pages_name(self,index,data_studied):
        self.get_dend_name(data_studied=data_studied,
                            index=index,  ) 
        
        dash_path_dend=os.path.join(self.dash_pages_path,self.data_studied,self.model_type, self.model_sufix, self.dend_path_inits[index])  
        os.makedirs(dash_path_dend, exist_ok=True)
        self.dash_pages_name=os.path.join(dash_path_dend,f'{self.dend_names[index]}.py')



    def clean_dend_name(self, 
                      path_head_clean,
                    dend_names=None,
                    dend_namess=None,
                    file_path_org=None, 
                    data_studied=None,
                    file_path_model_data=None,
                    name_path_fin=None,
                    name_path_fin_save=None, 
                    dend_path_inits=None,
                    model_sufix=None,
                    pinn_dir_data=None, 
                    pinn_dir_data_all=None,
                    model_sufix_all=None,
                    path_heads=None,
                    model_type=None,


        ):  
        model_type=model_type or self.model_type
        file_path_org =file_path_org or self.file_path_org 
        self.model_sufix=model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type,)
        dend_names = dend_names or self.dend_names
        file_path_model_data = file_path_model_data or self.file_path_model_data
        dend_namess = dend_namess or self.dend_namess
        name_path_fin = name_path_fin or self.name_path_fin
        dend_path_inits = dend_path_inits or self.dend_path_inits
        self.data_studied = data_studied or self.data_studied
        name_path_fin_save = name_path_fin_save or self.name_path_fin_save
        pinn_dir_data=pinn_dir_data or self.pinn_dir_data


        pinn_dir_data_all= pinn_dir_data_all or self.pinn_dir_data_all# or ['save','new','new_pre',pinn_dir_data,f'{pinn_dir_data}_pre']
        model_sufix_all= model_sufix_all or self.model_sufix_all# or ['pre','opt' ,model_sufix] 
        path_heads = path_heads or self.path_heads

        for path_head in path_head_clean:
            if path_head not in ['true','result']:
                for pa in pinn_dir_data_all:
                    if pa is not None:
                        for model_sufi in model_sufix_all: 
                            key=f'{path_head}_{model_sufi}_{pa}' 
                            if key in self.path_file:
                                remove_directory(self.path_file[key] )   

        if 'result' in path_head_clean:
            for pa in pinn_dir_data_all:
                if pa is not None:
                    for model_sufi in model_sufix_all:
                        for res in ['result']:
                            for path_head in path_heads:
                                key=f'{res}_{path_head}_{model_sufi}_{pa}'
                                self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([res,path_head, model_sufi, pa])
                                if key in self.path_file:
                                    remove_directory(self.path_file[key] )  

        if 'all' in path_head_clean:
            remove_directory(self.file_path_org_init)
            remove_directory(os.path.join(self.dash_pages_path,self.data_studied))



            path_headss=[pa for pa in path_heads if pa.startswith(('pinn','rpinn','gcn'))]
            for pa in pinn_dir_data_all:
                if pa is not None:
                    for model_sufi in model_sufix_all:
                        for res in ['data']:
                            for path_head in path_headss:
                                key=f'{path_head}_{model_sufi}_{pa}'
                                path= os.path.join(self.dend_path,  path_head)
                                # path= os.path.join(self.dend_path, res,path_head),'true'
                                #{res}_ self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([res,path_head, model_sufi, pa])
                                print('Removing ----------',path)
                                remove_directory(path) 

                        for res in ['result']:
                            for path_head in path_heads:
                                key=f'{res}_{path_head}_{model_sufi}_{pa}'
                                self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([res,path_head, model_sufi, pa])
                                if key in self.path_file:
                                    remove_directory(self.path_file[key] )  


            print('Removing ---------',path)
            remove_directory(self.file_path_feat)
            for nn in 'true_0':
                remove_directory(os.path.join(self.dend_path_org,  nn))

            if len(self.obj_org_path_dict)>0:
                for ii,(keys,val) in enumerate(self.obj_org_path_dict.items()): 
                    path_head ,pa,model_sufi=f'true_{ii}','save',f'save' 
                    key=f'{path_head}_{model_sufi}_{pa}'
                    path_heads.extend(path_head) 
                    self.dend_path_original_mm[key]=os.path.join(val, self.dend_name, 'data_org')
                    pasd=self.path_file_sub[self.inten_file_sub[0]][key]=self.path_file[key]=self.get_paths([path_head, model_sufi, pa])
                    remove_directory(self.path_file[key] )


            key=f'result_true'
            if key in path_head_clean:  
                remove_directory(self.path_file[key] )

            dash_path_dend=os.path.join(self.dash_pages_path,self.data_studied)
            os.makedirs(dash_path_dend, exist_ok=True)
            dash_path_dend=os.path.join(dash_path_dend,self.model_sufix)
            os.makedirs(dash_path_dend, exist_ok=True) 
            remove_directory(dash_path_dend) 
 

        for nn in 'true_0':
            remove_directory(os.path.join(self.dend_path_org,  nn))
            remove_directory(os.path.join(self.file_path_org, 'data', self.dend_path_init,'result',nn))

class dend_dataset:
    def __init__(self, dend_path_inits=None, dend_names=None, dend_namess=None, 
                 line_num_points_shaft_thre=8000,
                 line_num_points_inter_shaft_thre=800,
                 cts=6,
                 stoppage=3,
                 zoom_threshold=5000,
                 size_threshold=50):
        self.dend_path_inits = dend_path_inits if dend_path_inits else []
        self.dend_names = dend_names if dend_names else []
        self.dend_namess = dend_namess if dend_namess else []
        self.line_num_points_shaft_thre = line_num_points_shaft_thre
        self.line_num_points_inter_shaft_thre = line_num_points_inter_shaft_thre
        self.cts = cts
        self.stoppage = stoppage
        self.zoom_threshold = zoom_threshold
        self.size_threshold = size_threshold

    def add_data(self, dend_path_init, dend_name, dend_names_set):
        """Adds a new dendrite dataset."""
        self.dend_path_inits.append(dend_path_init)
        self.dend_names.append(dend_name)
        self.dend_namess.extend(dend_names_set)

    def __add__(self, other):
        """Combines two dend_dataset instances using the + operator."""
        if not isinstance(other, dend_dataset):
            raise TypeError("Can only add two dend_dataset instances.")

        return dend_dataset(
            dend_path_inits=self.dend_path_inits + other.dend_path_inits,
            dend_names=self.dend_names + other.dend_names,
            dend_namess=self.dend_namess + other.dend_namess,
            line_num_points_shaft_thre=self.line_num_points_shaft_thre,  # Keep the same params
            line_num_points_inter_shaft_thre=self.line_num_points_inter_shaft_thre,
            cts=self.cts,
            stoppage=self.stoppage,
            zoom_threshold=self.zoom_threshold,
            size_threshold=self.size_threshold
        )

    def __repr__(self):
        return (f"dend_dataset(dend_names={self.dend_names}, "
                f"dend_path_inits={self.dend_path_inits}, "
                f"dend_namess={self.dend_namess})")









def get_text_dash_train(user_input,file_path_org,dend_path_inits,   dend_name, dend_namess, data_studied, model_sufix,dash_pages_name,disp_infos=False,path=None):
    # Generate the Python script content
    code_content = f"""
import dash
from dash import callback  

import sys
import os

file_path_org = os.getcwd() 
     
sys.path.append(os.path.abspath(os.path.join(file_path_org,'dend_fun')))

from app_param_4 import app_param 


dend_names = ['{dend_name}']
dend_namess = {dend_namess}
dend_path_inits =['{dend_path_inits}']
data_studied = '{data_studied}'  
model_sufix = '{model_sufix}' 

mapp = app_param(
    file_path_org=file_path_org,
    model_sufix=model_sufix, 
    data_studied=data_studied,
    dend_names=dend_names,
    dend_namess=dend_namess, 
    dend_path_inits=dend_path_inits,  
    dropdow_path={path},
)

title_dend_name = f'{dend_name}'
dash.register_page(__name__, title=title_dend_name, name=title_dend_name, order={user_input}) 

def layout():
    return mapp.app_layout

@callback(
    mapp.Output,
    mapp.Input, 
    prevent_initial_call=mapp.prevent_initial_call
)
def update_output(mode, dendd, clusts, metric, intensity_type, width, height, iou_per, templ,index): 
    figure, txt = mapp.Get_output(mode, dendd, clusts, metric, intensity_type, width, height, iou_per, templ,index)
    return figure, txt
"""
 
    with open(dash_pages_name, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_name}")

 
def get_text_dash_test(user_input,file_path_org,dend_path_inits,   dend_name, dend_namess, data_studied, model_sufix,dash_pages_name,
                       disp_infos=False,
                       path_train=None, 
                            path_file=None, 
                    pinn_dir_data=None,
                    dend_data=None,
                    index=None,
                    model_type=None,
                    obj_org_path_dict=None,
                    model_sufix_dic=None,):
    # Generate the Python script content
    code_content = f""" 

import os, sys ,dash 
file_path_org=os.getcwd()
sys.path.append(file_path_org ) 
from  dend_fun_0.app_param_test import app_param
from dash import callback  


dend_names = ['{dend_name}']
dend_namess = {dend_namess}
dend_path_inits =['{dend_path_inits}']
data_studied = '{data_studied}'  
model_sufix = '{model_sufix}' 
path_train= {path_train}
pinn_dir_data='{pinn_dir_data}'
dend_data={dend_data}
index={index}
model_type='{model_type}'
obj_org_path_dict={obj_org_path_dict}
model_sufix_dic={model_sufix_dic}
mapp = app_param(
    file_path_org=file_path_org,
    model_sufix=model_sufix,
    path_train=path_train, 
    pinn_dir_data=pinn_dir_data,
    index=index, 
    data_studied=data_studied,  
    dend_data=dend_data,
    model_type=model_type,
    obj_org_path_dict=obj_org_path_dict,
    model_sufix_dic=model_sufix_dic,
)

title_dend_name = f'{dend_name}'
dash.register_page(__name__, title=title_dend_name, name=title_dend_name, order={user_input}) 

def layout():
    return mapp.app_layout

@callback(
    mapp.Output,
    mapp.Input, 
    prevent_initial_call=mapp.prevent_initial_call
)
def update_output( path_head,model_suf,path,mode,dendd, clusts,  intensity_type, width, height,  templ,nbin,index): 
    figure,txt = mapp.Get_output( path_head,model_suf,path,mode,dendd, clusts,   intensity_type, width, height, templ,nbin,index)
    return figure ,txt
""" 
    with open(dash_pages_name, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_name}")

 

 


def get_text_dash_all(    page_name, page_dir_txt,dash_pages_path,
                       disp_infos=False,
                       path_train=None, 
                            path_file=None, 
                    pinn_dir_data=None,
                    dend_data=None,
                    index=None,
                    ): 
    code_content = f"""

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from  dend_fun_0.side_bar import layout_1


page_dir= '/{page_dir_txt}' 
page_name='{page_name}'
dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return layout_1(page_dir)
""" 
    with open(dash_pages_path, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")



 

def get_text_dash_dnn(    page_name, page_dir_txt,dash_pages_path,
                       disp_infos=False, 
                       path_train=None, 
                            path_file=None, 
                    pinn_dir_data=None,
                    dend_data=None,
                    index=None,
                    ): 
    code_content = f"""

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from dend_fun_0.side_bar import sidebar ,get_dnn


page_dir= '/{page_dir_txt}' 
page_name='{page_name}'
dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return get_dnn(page_dir)
""" 
    with open(dash_pages_path, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")

 






import re

def safe_id(raw: str) -> str: 
    return re.sub(r'[.\{\}/\\ ]+', '_', raw)





class get_app_param(get_name ):
    def __init__(self,
                 dropdown_path_head_option=None,
                 dropdown_model_suf_option=None,
                 dropdown_path_option=None,
                 dropdown_true_keys_option=None,):
        self.dropdown_path_head_option=dropdown_path_head_option
        self.dropdown_model_suf_option=dropdown_model_suf_option
        self.dropdown_path_option=dropdown_path_option
        self.dropdown_true_keys_option=dropdown_true_keys_option
        pass

        get_name.__init__(self) 
        self.dropdown_options_style=dropdown_options_style = {'color': 'white', 'background-color': 'gray'} 
        self.styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }
        self.box_style = {
            'width': '100%',
            'padding': '3px',
            'font-size': '20px',
            'text-align-last': 'center',
            'margin': 'auto',  # Center-align the dropdown horizontally
            'background-color': 'black',
            'color': 'black'
        }



        self.template=template= [ "seaborn", "plotly_dark","plotly", "plotly_white", "ggplot2", "simple_white", "none"]
        # Color and style settings
        self.tcouleur = 'plotly_dark' 
        self.bcouleur = 'navy'
        self.fcouleur = ['white','black','grey']
        self.fsize = 16
        self.dropdown_template_option = []  
        for temp in template:
            self.dropdown_template_option.append({'label': f'{temp}', 
                                                    'value': f'{temp}', 
                                                    'style': dropdown_options_style})  




    def model_train(self):
        dropdown_options_style=self.dropdown_options_style
        self.dropdown_mode_option=dropdown_mode_option= [
                            {'label': 'Algorithm',      'value': 'algorithm', 'style': dropdown_options_style},
                            {'label': 'Comparison',     'value': 'comparison','style': dropdown_options_style}, 
                            {'label': 'Skeleton',      'value': 'skeleton', 'style': dropdown_options_style},
                            {'label': 'Results',        'value': 'result',    'style': dropdown_options_style}, 
                            {'label': 'Results Optimum','value': 'result_opt','style': dropdown_options_style}, 
                    ] 

        action_name='mode'
        self.dropdown_mode=dropdown_mode ={
                        'option':dropdown_mode_option,
                        'id'         :f'dropdown_{action_name}',
                        'value'      :'result',
                        'placeholder':f'Select {action_name}', 
        }


        ################################ METRIC #########################################


        self.dropdown_plot_option=[
                            # {'label': 'Distribution', 'value': 'dist',     'style': dropdown_options_style} ,
                            # {'label': 'Density',      'value': 'den',      'style': dropdown_options_style},
                            {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style},
                            {'label': 'IOU',                   'value': 'iou',     'style': dropdown_options_style},
                            {'label': 'Confusion Matrix',      'value': 'conf',    'style': dropdown_options_style},
                            {'label': 'Classification Report', 'value': 'report',  'style': dropdown_options_style},
                            {'label': 'Compare',               'value': 'compare', 'style': dropdown_options_style},
        ]
        ################################ INTENSITY #########################################
        self.dropdown_intensity_option= [
                            {'label': 'Clusterization',  'value': 'kmean',       'style': dropdown_options_style},
                            {'label': 'Segmentation',    'value': 'cluu_rad',    'style': dropdown_options_style},
                            {'label': 'Spines',          'value': 'spines',      'style': dropdown_options_style},
                            {'label': 'Spines True',     'value': 'spines true', 'style': dropdown_options_style},
                            {'label': 'Gauss Curvature', 'value': 'gauss',       'style': dropdown_options_style},
                            {'label': 'Mean Curvature',  'value': 'mean',        'style': dropdown_options_style},
                    ]


    def model_test(self):
        dropdown_options_style=self.dropdown_options_style
        dropdown_mode_option= [
                            {'label': 'Image',         'value': 'algorithm', 'style': dropdown_options_style},
                            # {'label': 'Comparison',        'value': 'comparison',    'style': dropdown_options_style},  
                            {'label': 'Skeleton',      'value': 'skeleton', 'style': dropdown_options_style},
                            # {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style}, 
                    ]  
        # dropdown_mode_option.append({'label': 'IOU',      'value': 'IOU', 'style': dropdown_options_style},)
        for mmnn in self.inten_file_model_head_neck:
            dropdown_mode_option.append({'label': mmnn,      'value': mmnn, 'style': dropdown_options_style},)
 
        # metric_name=self.metrics_keys
        metric_name=[]


        metric_name.extend(['heatmap_cylinder','heatmap_cylinder_surface'])
        metric_name.extend(self.metrics_keys)
        # metric_name.extend(['heatmap_iou','heatmap_iou_union','histogram_iou'])
        self.metric_mapping = {
            "name": metric_name,
            "title": [f"{mm} Histogram" for mm in metric_name],
            "index": {mm: ii for ii, mm in enumerate(metric_name)},
            "xtitle": "Length",
            "ytitle": "Count",
        }
        self.metrics_combine={'diam_head_neck_length'  :{'key':  ['head_diameter','neck_diameter','spine_length'],
                                                       'label':['Head Diameter','Neck Diameter','Spine Length']},
                              'vol_area_length_spine':{'key':['spine_vol','spine_area','spine_length'],
                                                        'label':['Spine Volume','Spine Area','Spine Length']},
                              'vol_head_neck_spine'    :{'key':['head_vol','neck_vol','spine_vol'],
                                                     'label':['Head Volume','Neck Volume','Spine Volume'],
                                                     },
                              'area_head_neck_spine'  :{'key':['head_area','neck_area','spine_area'],
                                                      'label':['Head Area','Neck Area','Spine Area']},
                              'length_head_neck_spine':{'key':['head_length','neck_length','spine_length'],
                                                        'label':['Head Length','Neck Length','Spine Length']},
                              }
        keys=list(self.metrics_combine.keys())
        self.metric_mapping_combine = {
            "name": keys,
            "title": [f"{self.metrics_combine[mm]['label'][0]} vs {self.metrics_combine[mm]['label'][1]} vs {self.metrics_combine[mm]['label'][2]}" for mm in keys],
            "index": {mm: ii for ii, mm in enumerate(keys)},
            "xtitle": [f"{self.metrics_combine[mm]['label'][0]}" for mm in keys],
            "ytitle":[f"{self.metrics_combine[mm]['label'][1]}" for mm in keys],
        }
        for mo in self.metric_mapping_combine['name']:
            dropdown_mode_option.append({'label': mo,        'value': mo,    'style': dropdown_options_style}) 
        for mo in self.metric_mapping['name']:
            dropdown_mode_option.append({'label': mo,        'value': mo,    'style': dropdown_options_style})
        action_name='mode'
        self.dropdown_mode ={
                        'option':dropdown_mode_option,
                        'id'         :f'dropdown_{action_name}',
                        'value'      :'algorithm',
                        'placeholder':f'Select {action_name}', 
        }
 
        ################################ METRIC #########################################


        self.dropdown_plot_option=[ 
                            {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style},
                            {'label': 'IOU',                   'value': 'iou',     'style': dropdown_options_style},
                            {'label': 'Confusion Matrix',      'value': 'conf',    'style': dropdown_options_style},
                            {'label': 'Classification Report', 'value': 'report',  'style': dropdown_options_style},
                            {'label': 'Compare',               'value': 'compare', 'style': dropdown_options_style},
        ] 

        self.dropdown_intensity_option=[]
        for intt,name in zip(self.inten_file_sub[1:],self.inten_file_sub_name[1:]):
            self.dropdown_intensity_option.append({'label': name,   'value': intt,       'style': dropdown_options_style})
        for intt in self.inten_file_train:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 
        # for intt in self.inten_pca:
        #     self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style})  
        for intt in self.inten_file:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style})  
        for intt in self.base_features_dict.keys():
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 





    def more_param(self,id_name_end,model_sufix,dend_name): 
        dropdown_options_style=self.dropdown_options_style 
        self.dend_name=dend_name 

        action_name='template' 
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_template={
            'option'     :self.dropdown_template_option,
            'id'         :id_name,
            'value'      :self.dropdown_template_option[1]['value'],
            'placeholder':f'Select {action_name}', 
        }
        action_name='intensity'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_intensity ={
        'option':self.dropdown_intensity_option,
        'id'         :id_name,
        'value'      :self.dropdown_intensity_option[0]['value'],
        'placeholder':f'Select {action_name}', 
        }

        action_name='metric'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_plot={
        'option':self.dropdown_plot_option,
        'id'          :id_name,
        'value'       :self.dropdown_plot_option[0]['value'],
        'placeholder' :f'Select {action_name}', 
        } 

        self.output_graph_1={
            'id':f'output-graph1_{action_name}_{id_name_end}', 
            'style':{'display': 'flex', 'justify-content': 'center'}
        }

        self.output_text_1={
            'id':f'output-text1_{action_name}_{id_name_end}', 
            # 'style':{'display': 'flex', 'justify-content': 'center'}
        } 

        self.height_slider={
                        'id':f'height-slider_{action_name}_{id_name_end}',
                        'min':300,
                        'max':900,
                        'step':50,
                        'value':750, 
                        'marks':{i: f'{i}px' for i in range(300, 1000, 200)}, 
        }

        self.width_slider={
                        'id':f'width-slider_{action_name}_{id_name_end}',
                        'min':400,
                        'max':1200,
                        'step':50,
                        'value':1000, 
                        'marks':{i: f'{i}px' for i in range(400, 1200, 200)}, 
        }

        self.iou_slider={
                        'id':f'iou-slider_{action_name}_{id_name_end}',
                        'min':0,
                        'max':100,
                        'step':1,
                        'value':70, 
                        'marks':{i: f'{i}%' for i in range(0, 100, 20)}, 
        } 
 

        self.hist_slider={
                        'id':f'iou-slider_{action_name}_{id_name_end}',
                        'min':0,
                        'max':1000,
                        'step':5,
                        'value':100, 
                        'marks':{i: f'{i}' for i in range(0, 1000, 200)}, 
        } 
 
   
        action_name='true_keys'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_true_keys={
            'option'     :self.dropdown_true_keys_option,
            'id'         :id_name,
            'value'      :self.dropdown_true_keys_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }

        action_name='path_head'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_path_head={
            'option'     :self.dropdown_path_head_option,
            'id'         :id_name,
            'value'      :self.dropdown_path_head_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }

        action_name='model_suf'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_model_suf={
            'option'     :self.dropdown_model_suf_option,
            'id'         :id_name,
            'value'      :f'{model_sufix}',
            'placeholder':f'Select {action_name}', 
        }
 
        action_name='path'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_path={
            'option'     :self.dropdown_path_option,
            'id'         :id_name,
            'value'      :self.dropdown_path_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        } 
        dropdown_dend_option = [{'label': 'Initial', 'value': 'init', 'style': dropdown_options_style} ,
                                {'label': f'Smoothed', 'value': 'smooth', 'style': dropdown_options_style}]
        action_name='dendrite curve state'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_dend={
            'option'     :dropdown_dend_option,
            'id'         :id_name,
            'value'      :dropdown_dend_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }
 

    def get_dropdown_cluster_test(self,id_name_end,count):
        dropdown_options_style=self.dropdown_options_style 
        dropdown_cluster_option = [{'label': 'Dendrite', 'value': 0, 'style': dropdown_options_style}] 
        dropdown_cluster_option.append({'label': f'Shaft', 'value': 1, 'style': dropdown_options_style}) 
        ik=0
        jj=0
        scatter=[]
        for iii in count:  
                dropdown_cluster_option.append({'label': f'sp({iii:02})', 'value': iii+2, 'style': dropdown_options_style}) 
        
        action_name='cluster'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_cluster={
            'option'     :dropdown_cluster_option,
            'id'         :id_name,
            'value'      :dropdown_cluster_option[1]['value'],
            'placeholder':f'Select {action_name}', 
        }


    def get_dropdown_cluster(self,id_name_end,count):
        dropdown_options_style=self.dropdown_options_style 
        dropdown_cluster_option =     [{'label': 'Dendrite', 'value': 0, 'style': dropdown_options_style}] 
        dropdown_cluster_option.append({'label': f'Shaft'  , 'value': 1, 'style': dropdown_options_style})
        jjjj=2 
        for iii in count: 
            dropdown_cluster_option.append({
                'label': f'sp({iii:02})' if isinstance(iii, int) else f'sp_ap({iii[0]:02})~sp_an({iii[1]:02})', 
                'value':jjjj,
                'style': dropdown_options_style
            })
            jjjj+=1
        action_name='cluster'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_cluster={
            'option'     :dropdown_cluster_option,
            'id'         :id_name,
            'value'      :dropdown_cluster_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }



    def get_dropdown_index(self,id_name_end,dend_names,index):
        dropdown_options_style=self.dropdown_options_style  
        dropdown_index_option=[{
                'label': f'{iii}', 
                'value':ii,
                'style': dropdown_options_style
            } for ii,iii in enumerate(dend_names)] 
        action_name='index'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_index={
            'option'     :dropdown_index_option,
            'id'         :id_name,
            'value'      :index, 
            'placeholder':f'Select {action_name}', 
        } 



 



def get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir): 
    model = {
        path: {'data': {}, 'dest': {}}
        for path in ['shaft_path','spine_path', 'spine_path_pre', ] 
                     
    }
     
    for md in model:
        model[md]['data']['head'] = data_head  
        model[md]['dest']['head'] = dest_head 
        model[md]['data']['sufix'] = data_sufix
        model[md]['dest']['sufix'] = dest_sufix
        if md in ['spine_path_pre','center_curv']:
            model[md]['data']['dir'] = f'{data_dir}_pre'
            model[md]['dest']['dir'] = f'{dest_dir}_pre'
        elif md == 'shaft_path': 
            model[md]['data']['dir'] = f'save'
            model[md]['dest']['dir'] = f'save'
        else:
            if not data_dir.startswith('save') and not dest_dir.startswith('save'):
                model[md]['data']['dir'] = data_dir
                model[md]['dest']['dir'] = dest_dir
     
    path_train = {} 
    path_train['data_true_iou'] = 'true_save_save'  
    path_train['pinn_save_save'] = 'pinn_save_save' 
    for md in model:
        path_train[f'data_{md}'] = f"{model[md]['data']['head']}_{model[md]['data']['sufix']}_{model[md]['data']['dir']}"
        path_train[f'dest_{md}'] = f"{model[md]['dest']['head']}_{model[md]['dest']['sufix']}_{model[md]['dest']['dir']}"


    path_train['data_shaft_vertices_center_path']=f"{model['shaft_path']['data']['head']}_{model['shaft_path']['data']['sufix']}_{model['shaft_path']['data']['dir']}"
    path_train['dest_shaft_vertices_center_path']=f"{model['shaft_path']['dest']['head']}_{model['shaft_path']['dest']['sufix']}_{model['shaft_path']['dest']['dir']}"
    return path_train



 



class get_data_mode(get_name,get_model_name):
    def __init__(self,mode_ids=[],
                pre_portion=None,
                data_mode={}):
        super().__init__() 
        self.data_mode=data_mode
        self.data_mode['model_sufix_all']=[]
        self.data_mode['pinn_dir_data_all']=[]
        self.model_sufix_all=[]
        self.pinn_dir_data_all=[]
        self.data_mode['mode_id']=[]
        self.mode_ids=mode_ids
        
        pinn_dir_dest= data_sufix=  data_dir= 'save' 


        self.model_sufix_all.extend([ data_sufix]) 
        self.model_sufix_all=list(set(self.model_sufix_all))
        self.pinn_dir_data_all.extend([data_dir, ])
        self.pinn_dir_data_all=list(set(self.pinn_dir_data_all))


    def train_pre(self,data_mode=None,
                pre_portion='head_neck',
                pre_opt='pre',
                train_test='train', 
                data_head='true',
                dest_head='true', 
                seg_dend='nfull',
                base_features_index=[0,1,2,3] ,
                ):
        mon=get_model_name(
            pre_opt=pre_opt,
            train_test=train_test, 
            seg_dend=seg_dend,
            dest_head=dest_head,
            pre_portion=pre_portion)
        data_mode=data_mode if data_mode is not None else self.data_mode 

        

        inten_pinn_index=[]
        mon.vals(base_features_index=base_features_index,
                 inten_pinn_index=inten_pinn_index)
        list_features=[]
        base_features_list=mon.base_features_list
        dest_sufix_pr= mon.dest_sufix 

 

 
        data_sufix_pr=pinn_dir_data='save'
        data_dir_pr='save'
        dest_dir_pr=mon.dest_dir# 
        
        data_sufix=data_sufix_pr
        dest_sufix=dest_sufix_pr
        data_dir=dest_dir_pr
        dest_dir=dest_dir_pr
 
        self.mode_id=mode_id=mon.mode_id#f'train_{seg_dend}_{pre_portion[:2]}_{dest_head}_{dest_sufix}'
        self.mode_ids.append(mode_id)
        data_mode['mode_id'].append(mode_id)
        data_mode[mode_id]={}
        path_train_pre=get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir) 
        data_mode[mode_id]['path_train']= path_train_pre 
        data_mode[mode_id]['model_sufix']=[dest_sufix ]
        data_mode[mode_id]['dest_dir']=dest_dir 
        data_mode[mode_id]['base_features_list']=base_features_list
        data_mode[mode_id]['list_features']=list_features
        data_mode[mode_id]['pre_portion']= pre_portion 
        data_mode[mode_id]['pinn_dir_data']= pinn_dir_data   
        data_mode[mode_id]['seg_dend']=seg_dend
        data_mode[mode_id]['dest_path']='dest_spine_path' 
        data_mode[mode_id]['get_training']=True 
        data_mode[mode_id]['get_shaft']=True  
        data_mode[mode_id]['get_segm']=True 
        data_mode[mode_id]['get_head_neck_segm']=True  
        data_mode[mode_id]['get_segss_group']=False
        data_mode[mode_id]['train_spines']=False  
        
        data_shaft_train_path=data_mode[mode_id]['path_train']['dest_spine_path_pre']
        data_shaft_vertices_center_path=data_mode[mode_id]['path_train']['data_shaft_path']
        data_mode[mode_id]['path_train']['data_train_path']=data_shaft_train_path
        data_mode[mode_id]['path_train']['data_shaft_path']=data_shaft_vertices_center_path
        dest_shaft_vertices_center_path=data_mode[mode_id]['path_train']['dest_shaft_path']
        data_shaft_vertices_center_path_opt=data_mode[mode_id]['path_train']['data_spine_path']
        dest_shaft_vertices_center_path_opt=data_mode[mode_id]['path_train']['dest_spine_path']

        self.dest_sufix_pr=dest_sufix_pr
        self.dest_dir_pr=dest_dir_pr
        self.pinn_dir_data=pinn_dir_data
        self.data_shaft_train_path=data_shaft_train_path
        self.data_shaft_vertices_center_path=data_shaft_vertices_center_path
        self.dest_shaft_vertices_center_path_opt=dest_shaft_vertices_center_path_opt 
 
         
        data_mode[mode_id]['path_train']['data_shaft_path']=data_shaft_vertices_center_path
        data_mode[mode_id]['path_train']['data_spine_path_center']=dest_shaft_vertices_center_path_opt
        data_mode[mode_id]['path_train']['dest_spine_path_center']=data_mode[mode_id]['path_train']['dest_spine_path'] 

        self.model_sufix_all.extend([dest_sufix,data_sufix]) 
        self.model_sufix_all=list(set(self.model_sufix_all))
        self.pinn_dir_data_all.extend([data_dir,f'{data_dir}_pre',dest_dir,f'{dest_dir}_pre'])
        self.pinn_dir_data_all=list(set(self.pinn_dir_data_all))

        return data_mode





    def train_opt(self,data_mode=None,
            pre_opt='opt',
            train_test='train', 
                pre_portion='head_neck',
                data_head='true',
                dest_head='true', 
                seg_dend='nfull',
                base_features_index=[0,1,2,3],
        inten_pinn_index=[0,1],
                ):
        mon=get_model_name(
            pre_opt=pre_opt,
            train_test=train_test, 
            seg_dend=seg_dend,
            dest_head=dest_head,
            pre_portion=pre_portion)
        data_mode=data_mode if data_mode is not None else self.data_mode

        data_sufix=dest_sufix_pr=self.dest_sufix_pr
        data_dir=dest_dir_pr=self.dest_dir_pr
        pinn_dir_data=self.pinn_dir_data
        data_shaft_train_path=self.data_shaft_train_path
        data_shaft_vertices_center_path=self.data_shaft_vertices_center_path
        dest_shaft_vertices_center_path_opt=self.dest_shaft_vertices_center_path_opt 
 
   
        mon.vals(base_features_index=base_features_index,
                 inten_pinn_index=inten_pinn_index) 
        base_features_list=mon.base_features_list
        dest_sufix=  mon.dest_sufix
        self.mode_id=mode_id=mon.mode_id
        self.mode_ids.append(mode_id)
        list_features=[[mon.inten_pinn_tmp[nam]['id_path'], mon.inten_pinn_tmp[nam]['name'] ] for nam in inten_pinn_index] 
        dest_dir=mon.dest_dir  

  
        data_mode['mode_id'].append(mode_id)
        data_mode[mode_id]={}


        path_train_opt=get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir) 
        path_train_opt['data_shaft_vertices_center_path']=path_train_opt['data_true_iou'] 
        data_mode[mode_id]['path_train']= path_train_opt#get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir)
        data_mode[mode_id]['model_sufix']=[dest_sufix ]
        data_mode[mode_id]['dest_dir']=dest_dir  
        data_mode[mode_id]['base_features_list']=base_features_list
        data_mode[mode_id]['list_features']=list_features
        data_mode[mode_id]['pre_portion']= pre_portion  
        data_mode[mode_id]['pinn_dir_data']= pinn_dir_data  
        data_mode[mode_id]['seg_dend']=seg_dend
        data_mode[mode_id]['get_training']=True  
        data_mode[mode_id]['get_shaft']=True 
        data_mode[mode_id]['get_segm']=True 
        data_mode[mode_id]['get_head_neck_segm']=True 
        data_mode[mode_id]['dest_path']='dest_spine_path' 
        data_mode[mode_id]['get_segss_group']=True
        data_mode[mode_id]['train_spines']=False  

        data_mode[mode_id]['path_train']['data_train_path']=data_shaft_train_path 
        data_mode[mode_id]['path_train']['data_shaft_path']=data_shaft_vertices_center_path
        data_mode[mode_id]['path_train']['data_spine_path_center']=dest_shaft_vertices_center_path_opt
        data_mode[mode_id]['path_train']['dest_spine_path_center']=data_mode[mode_id]['path_train']['dest_spine_path'] 
 
        self.model_sufix_all.extend([dest_sufix,data_sufix]) 
        self.model_sufix_all=list(set(self.model_sufix_all))
        self.pinn_dir_data_all.extend([data_dir,f'{data_dir}_pre',dest_dir,f'{dest_dir}_pre'])
        self.pinn_dir_data_all=list(set(self.pinn_dir_data_all))
        
        return data_mode




    def test_pre(self,data_mode=None,
                pre_opt='pre',
                train_test='test',
                 pre_portion='head_neck',
                data_head='pinn',
                dest_head='pinn', 
                seg_dend='nfull' ,
                base_features_index=[0,1,2,3] ,
                ):
        mon=get_model_name(
            pre_opt=pre_opt,
            train_test=train_test, 
            seg_dend=seg_dend,
            dest_head=dest_head,
            pre_portion=pre_portion)
        
        data_mode=data_mode if data_mode is not None else self.data_mode
  
        #____________________________________'pre_non_full'________________________________ 
    
        inten_pinn_index=[]
        mon.vals(base_features_index=base_features_index,
                 inten_pinn_index=inten_pinn_index) 
        base_features_list=mon.base_features_list
        dest_sufix=dest_sufix_pr=data_sufix_pr=mon.dest_sufix
        self.mode_id=mode_id=mon.mode_id
        self.mode_ids.append(mode_id)
        list_features=[[mon.inten_pinn_tmp[nam]['id_path'], mon.inten_pinn_tmp[nam]['name'] ] for nam in inten_pinn_index] 
 
        dest_dir=dest_dir_pr=mon.dest_dir 

        data_sufix=data_sufix_pr#'save'
        dest_sufix=dest_sufix_pr
        data_dir=dest_dir#data_dir_pr #'save'#
        # dest_dir=dest_dir#dest_dir_pr 
        # path_train_pre=get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir)
 
        # mode_id=f'test_{seg_dend}_{pre_portion[:2]}_{dest_head}_{dest_sufix}'
        data_mode['mode_id'].append(mode_id)
        data_mode[mode_id]={} 
        data_mode[mode_id]['path_train']= get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir) 
        data_mode[mode_id]['model_sufix']=[dest_sufix] 
        data_mode[mode_id]['dest_dir']=dest_dir 
        data_mode[mode_id]['base_features_list']=base_features_list
        data_mode[mode_id]['list_features']=[] 
        data_mode[mode_id]['pre_portion']= pre_portion 
        data_mode[mode_id]['pinn_dir_data']= seg_dend 
        data_mode[mode_id]['seg_dend']='full' 
        data_mode[mode_id]['get_training']=False 
        data_mode[mode_id]['get_shaft']=True  
        data_mode[mode_id]['get_segm']=True 
        data_mode[mode_id]['get_head_neck_segm']=True 
        data_mode[mode_id]['dest_path']='dest_spine_path' 
        data_mode[mode_id]['get_segss_group']=False
        data_mode[mode_id]['train_spines']=False 
 
        data_mode[mode_id]['shaft']={}
        data_mode[mode_id]['shaft']['path']='dest_spine_path_pre' 
        data_mode[mode_id]['shaft']['tf']=True

        data_mode[mode_id]['group']={}
        data_mode[mode_id]['group']['path']='dest_spine_path' 
        data_mode[mode_id]['group']['tf']=False
        data_mode[mode_id]['group']['seg_dend']='nfull'

        data_mode[mode_id]['process']={}
        data_mode[mode_id]['process']['path']='dest_spine_path_pre' 
        data_mode[mode_id]['process']['tf']=True
        data_mode[mode_id]['process']['seg_dend']='full'

        data_mode[mode_id]['sp_full']={}
        data_mode[mode_id]['sp_full']['path']='dest_spine_path_pre' 
        data_mode[mode_id]['sp_full']['tf']=True
        data_mode[mode_id]['sp_full']['seg_dend']='full'

        data_mode[mode_id]['sp_nfull']={}
        data_mode[mode_id]['sp_nfull']['path']='dest_spine_path' 
        data_mode[mode_id]['sp_nfull']['tf']=True
        data_mode[mode_id]['sp_nfull']['seg_dend']='nfull'


        data_mode[mode_id]['hn_full']={}
        data_mode[mode_id]['hn_full']['path']='dest_spine_path_pre' 
        data_mode[mode_id]['hn_full']['tf']=True
        data_mode[mode_id]['hn_full']['seg_dend']='full'

        data_mode[mode_id]['hn_nfull']={}
        data_mode[mode_id]['hn_nfull']['path']='dest_spine_path' 
        data_mode[mode_id]['hn_nfull']['tf']=True
        data_mode[mode_id]['hn_nfull']['seg_dend']='nfull'



        data_shaft_train_path=data_mode[mode_id]['path_train']['dest_spine_path_pre']
        data_shaft_vertices_center_path=data_mode[mode_id]['path_train']['data_shaft_path']
        dest_shaft_vertices_center_path=data_mode[mode_id]['path_train']['dest_shaft_path']
        data_shaft_vertices_center_path_opt=data_mode[mode_id]['path_train']['data_spine_path']
        dest_shaft_vertices_center_path_opt=data_mode[mode_id]['path_train']['dest_spine_path']


        self.dest_sufix_pr=dest_sufix_pr
        self.dest_dir_pr=dest_dir_pr
        self.pinn_dir_data=seg_dend
        self.data_shaft_train_path=data_shaft_train_path
        self.data_shaft_vertices_center_path=data_shaft_vertices_center_path
        self.dest_shaft_vertices_center_path_opt=dest_shaft_vertices_center_path_opt  
        self.dest_spine_path_pre_init=data_mode[mode_id]['path_train']['pinn_save_save']

 

        data_mode[mode_id]['path_train']['data_spine_path_center']=data_mode[mode_id]['path_train']['data_shaft_path']
        data_mode[mode_id]['path_train']['dest_spine_path_center']=data_mode[mode_id]['path_train']['dest_shaft_path']
        data_mode[mode_id]['path_train']['dest_spine_path_pre_init']=self.dest_spine_path_pre_init
        data_mode[mode_id]['path_train']['data_train_path']=data_shaft_train_path
 
        self.model_sufix_all.extend([dest_sufix,data_sufix]) 
        self.model_sufix_all=list(set(self.model_sufix_all))
        self.pinn_dir_data_all.extend([data_dir,f'{data_dir}_pre',dest_dir,f'{dest_dir}_pre'])
        self.pinn_dir_data_all=list(set(self.pinn_dir_data_all))

        return data_mode




    def test_opt(self,data_mode=None,
                pre_opt='opt',
                train_test='test',
                pre_portion='head_neck',
                data_head='pinn',
                dest_head='pinn', 
                seg_dend='nfull' ,
                base_features_index=[0,1,2,3] ,
                inten_pinn_index=[0],
                 ):
        mon=get_model_name(
            pre_opt=pre_opt,
            train_test=train_test, 
            seg_dend=seg_dend,
            dest_head=dest_head,
            pre_portion=pre_portion)
        data_mode=data_mode if data_mode is not None else self.data_mode 
        data_sufix= self.dest_sufix_pr
        data_dir=self.dest_dir_pr
        pinn_dir_data=self.pinn_dir_data
        data_shaft_train_path=self.data_shaft_train_path
        data_shaft_vertices_center_path=self.data_shaft_vertices_center_path
        dest_shaft_vertices_center_path_opt=self.dest_shaft_vertices_center_path_opt
        pinn_dir_dest=seg_dend  


 
        #--------------------------- OPT opt_non_full_gmg2m2_sh_vcv_di_sid_pca123________________________________ 
  

        # model_name='vcv'
        # inten_pca=['shaft_vcv_length']
        # list_features=[['data_spine_path_center',f'{nam}.txt'] for nam in inten_pca] 

        
        mon.vals(base_features_index=base_features_index,
                 inten_pinn_index=inten_pinn_index) 
        base_features_list=mon.base_features_list
        dest_sufix=mon.dest_sufix
        dest_dir=mon.dest_dir
        list_features=[[mon.inten_pinn_tmp[nam]['id_path'], mon.inten_pinn_tmp[nam]['name'] ] for nam in inten_pinn_index] 
   
        self.mode_id=mode_id=mon.mode_id  
        self.mode_ids.append(mode_id)
 

        data_mode['mode_id'].append(mode_id)
        data_mode[mode_id]={}
 
        
        data_mode[mode_id]['path_train']= get_path_train(data_head, dest_head, data_sufix, dest_sufix, data_dir, dest_dir)
        data_mode[mode_id]['model_sufix']=[dest_sufix] 
        data_mode[mode_id]['base_features_list']=base_features_list
        data_mode[mode_id]['list_features']=list_features
        data_mode[mode_id]['pre_portion']= pre_portion  
        data_mode[mode_id]['pinn_dir_data']= pinn_dir_data  
        data_mode[mode_id]['seg_dend']='nfull'
        data_mode[mode_id]['dest_path']='dest_spine_path' 
        data_mode[mode_id]['get_training']=False  
        data_mode[mode_id]['get_shaft']=True 
        data_mode[mode_id]['get_segm']=True 
        data_mode[mode_id]['get_segm']=True 
        data_mode[mode_id]['get_head_neck_segm']=True 
        data_mode[mode_id]['get_segss_group']=False
        data_mode[mode_id]['train_spines']=False   

        data_mode[mode_id]['shaft']={}
        data_mode[mode_id]['shaft']['path']='dest_spine_path_pre' 
        data_mode[mode_id]['shaft']['tf']=True

        data_mode[mode_id]['group']={}
        data_mode[mode_id]['group']['path']='dest_spine_path' 
        data_mode[mode_id]['group']['tf']=False
        data_mode[mode_id]['group']['seg_dend']='nfull'

        data_mode[mode_id]['process']={}
        data_mode[mode_id]['process']['path']='dest_spine_path_pre' 
        data_mode[mode_id]['process']['tf']=False
        data_mode[mode_id]['process']['seg_dend']='full'

        data_mode[mode_id]['sp_full']={}
        data_mode[mode_id]['sp_full']['path']='dest_spine_path_pre' 
        data_mode[mode_id]['sp_full']['tf']=True
        data_mode[mode_id]['sp_full']['seg_dend']='full'

        data_mode[mode_id]['sp_nfull']={}
        data_mode[mode_id]['sp_nfull']['path']='dest_spine_path' 
        data_mode[mode_id]['sp_nfull']['tf']=True
        data_mode[mode_id]['sp_nfull']['seg_dend']='nfull'


        data_mode[mode_id]['hn_full']={}
        data_mode[mode_id]['hn_full']['path']='dest_spine_path_pre' 
        data_mode[mode_id]['hn_full']['tf']=True
        data_mode[mode_id]['hn_full']['seg_dend']='full'

        data_mode[mode_id]['hn_nfull']={}
        data_mode[mode_id]['hn_nfull']['path']='dest_spine_path' 
        data_mode[mode_id]['hn_nfull']['tf']=True
        data_mode[mode_id]['hn_nfull']['seg_dend']='nfull'



        data_mode[mode_id]['path_train']['dest_spine_path_pre_init']=self.dest_spine_path_pre_init
        data_mode[mode_id]['path_train']['data_train_path']=data_shaft_train_path
        data_mode[mode_id]['path_train']['data_shaft_path']=data_shaft_vertices_center_path
        data_mode[mode_id]['path_train']['data_spine_path_center']=dest_shaft_vertices_center_path_opt
        data_mode[mode_id]['path_train']['dest_spine_path_center']=data_mode[mode_id]['path_train']['dest_spine_path'] 
 


        # data_mode['pinn_dir_data_all']=list(set(data_mode['pinn_dir_data_all'])) 
        # data_mode['model_sufix_all']=list(set(data_mode['model_sufix_all'])) 
        self.model_sufix_all.extend([dest_sufix,data_sufix]) 
        self.model_sufix_all=list(set(self.model_sufix_all))
        self.pinn_dir_data_all.extend([data_dir,f'{data_dir}_pre',dest_dir,f'{dest_dir}_pre'])
        self.pinn_dir_data_all=list(set(self.pinn_dir_data_all))

        return data_mode

 

def get_model_test(path_heads,pre_portion,dnn_modes):
    modes={val:{} for val in path_heads}
    configs=get_configs()
    dmode = get_data_mode(pre_portion=pre_portion) 

    for val in path_heads:
        ids,name=0,'mode0'  
        data_mode=dmode.test_pre(pre_portion=pre_portion,
                                    data_head=val,
                                    dest_head=val, )
        modes[val][name]=[dmode.mode_id] 
        print(f"Finished {name}, mode_id={dmode.mode_id}") 

        for ids, name in enumerate(dnn_modes):
            cfg=configs[name]
            data_mode = dmode.test_opt(
                                    data_mode=data_mode,
                                    pre_portion=pre_portion,
                                    train_test='test',
                                    data_head=val,
                                    dest_head=val, 
                                    **cfg
                                )
            modes[val][name]=[dmode.mode_id] 
            print(f"Finished {name}, mode_id={dmode.mode_id}")

    return modes,data_mode,dmode





