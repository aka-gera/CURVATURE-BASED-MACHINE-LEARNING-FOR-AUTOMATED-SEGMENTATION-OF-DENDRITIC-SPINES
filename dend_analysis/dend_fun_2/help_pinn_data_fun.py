

import sys
import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np 
import tensorflow as tf  
tf.config.run_functions_eagerly(True)  
DTYPE='float32' 
import pickle 
from dend_fun_0.curvature import curv_mesh as curv_mesh
import dend_fun_0.help_funn as hff   
from dend_fun_0.help_funn import dendrite,get_unique_class,  cluster_class,label_cluster,Branch_division,get_intensity,Threshold_curv,Impute_intensity, order_points_along_pca
from dend_fun_0.help_funn import mappings_vertices,clust_pca,dendrite_io,volume,closest_distances_group,find_min_max_no_cross,Curve_length   
 
DTYPE = tf.float32

from sklearn.neighbors import KDTree  

from dend_fun_0.obj_get import Obj_to_vertices,get_obj_filenames_with_indices_2  
from dend_fun_2.metric import center_curvature, get_kmean,get_kmean_mean ,get_center_lines,get_kmean_mode
from dend_fun_0.help_pinn_data_fun import get_model,model_shaft 
from dend_fun_0.help_pinn_data_fun import pinn_data as pdata
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0" 
 
from dend_fun_0.help_spine_division import region_branch 


import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

 

from collections import defaultdict
import trimesh 
from scipy.spatial import KDTree 


from skimage.measure import label 
from skimage.morphology import skeletonize  
from scipy.ndimage import label 








def get_contraction(vertices,   skeleton_points, alpha=0.5):   
    return  (1 - alpha) * vertices + alpha * skeleton_points[ KDTree(skeleton_points).query(vertices )[1] ]


import open3d as o3d 
class mesh_resize():
    def __init__(self,vertices,faces,target_number_of_triangles=6000, ):
        self.vertices=vertices 
        self.faces=faces
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces) 
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles) 
        self.mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=False,
        ) 
    def get_index(self,):
        idx = KDTree(self.vertices).query(self.mesh.vertices)[1].flatten()

        unique, counts = np.unique(idx, return_counts=True)
        duplicates = unique[counts > 1]
        idd={va:ke for ke,va in enumerate(idx)}
        tree = KDTree(self.vertices)
        duplicates = set(duplicates)

        replacement = {}
        idxx=list(idx)
        for d in duplicates:
            dist, nn = tree.query(self.vertices[d], k=min(40,self.vertices.shape[0]))  # search 20 neighbors
            for n in nn:
                if n not in idxx:
                    replacement[d] = n
                    idxx.append(n) 
                    break
        for key,val in replacement.items():
            idx[idd[key]]=val  
        self.index=idx 

    def get_mesh(self,vertices=None,):
        vertices=vertices if vertices is not None else self.vertices
        self.vertices_simplify=vertices[self.index]
        self.faces_simplify=self.mesh.faces
        self.dend = curv_mesh(vertices=self.vertices_simplify, faces=self.mesh.faces,)
        if not hasattr(self, "vertex_neighbor")  :
            self.dend.Vertices_neighbor() 
            self.vertex_neighbor=self.dend.vertex_neighbor
        self.mapp=mappings_vertices(vertices_0=vertices)
        # self.vertex_neighbor=self.dend.vertex_neighbor
 
    # def reduce_rhs(self, labels): 
    #     self.get_index_simplify()
    #     return labels[self.idx]
    




    def reduce_rhs(self, label, label_gen=None,size_threshold=3,tf_mesh=False):
        if not hasattr(self, "index"):
            self.get_index()  
        if len(label)==len(self.vertices):
            if (np.array(label).ndim<3) and not tf_mesh:
                return label[self.index] 
            else:
                return trimesh.Trimesh(vertices=label[self.index] ,
                                 faces=self.mesh.faces,process=False )
        label_gen=label_gen if label_gen is not None else self.vertices
        self.get_mesh(vertices=label_gen) 
        vertices_index_=self.mapp.Mapping_inverse(label) 
        intensity=np.zeros_like(label_gen[:,0:1],dtype=int)
        intensity[vertices_index_]=1
        vertices_index_=np.where(intensity[self.index].astype(int)==1)[0] 
        if (np.array(label).ndim<3) and not tf_mesh:
            return label[vertices_index_] 

        cclu=cluster_class(faces_neighbor_index=self.vertex_neighbor)  
        node = Branch_division(
            cclu=cclu,
            dend=self.dend, 
            vertices_index=vertices_index_,
            size_threshold=size_threshold,
            stop_index=1,
        )    
        if len(node.children)>0:  
            return trimesh.Trimesh(vertices=self.vertices_simplify[node.children[0].vertices_index],
                                 faces=node.children[0].faces,process=False) 
        else:
            faces=self.faces_simplify 
            orig_idx = np.asarray(vertices_index_, dtype=int)  
            max_orig = orig_idx.max()
  
            faces = np.asarray(faces, dtype=int)
 
            map_dict = {orig: new for new, orig in enumerate(orig_idx)}
 
            flat = faces.ravel()
            mapped_flat = np.empty_like(flat, dtype=int)
            missing = set()
            for i, idx in enumerate(flat):
                new = map_dict.get(int(idx))
                if new is None:
                    missing.add(int(idx))
                    mapped_flat[i] = -1
                else:
                    mapped_flat[i] = new
 
            faces_mapped = mapped_flat.reshape(faces.shape) 
            return  trimesh.Trimesh(vertices=self.vertices_simplify[vertices_index_], 
                                    faces=faces_mapped, process=False)


 

        return None
    
 
import trimesh
import open3d as o3d 
from scipy.ndimage import gaussian_filter1d




class smooth_curvature:

    def __init__(self, mesh, iterations=None, sigma=3, radius=None):

        # Convert to Open3D
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        iterations =iterations if iterations is not None else max(10, min(40,len(mesh.edges_unique) // 500))
        # Volume-preserving Taubin smoothing
        o3d_mesh = o3d_mesh.filter_smooth_taubin(
            number_of_iterations=iterations
        )
        o3d_mesh.compute_vertex_normals()
 
        mesh_smooth = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            # faces=np.asarray(o3d_mesh.triangles),
            faces=mesh.faces,
            process=False
        )
        if radius is None:
            radius = np.median(mesh_smooth.edges_unique_length) * 0.25

        curv = trimesh.curvature.discrete_mean_curvature_measure(
            mesh_smooth,
            mesh_smooth.vertices,
            radius=radius
        )
 
        curv_smooth = gaussian_filter1d(curv, sigma=sigma)
 
        self.mesh_smooth = mesh_smooth
        self.curv_smooth = curv_smooth






class mesh_to_skeleton():
    def __init__(self,vertices,faces,target_number_of_triangles=6000,voxel_resolution = 128,tf_largest=False,
                 disp_infos=True,
                 tf_resize=True,
                 skl_method='medial_axis',
                 fill_method="orthographic",
                 ):
        self.vertices=vertices
         
        from skimage.morphology import medial_axis
        if tf_resize:
            mrs=mesh_resize(vertices,faces,target_number_of_triangles=target_number_of_triangles, )
            self.mesh=mrs.mesh
        else: 
            self.mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices),
                faces=np.asarray(faces)
            ) 

        edges = np.median(self.mesh.edges_unique_length) * 0.25
        print('mesh_to_skeleton-------edge----------',edges)
        voxelized = self.mesh.voxelized(pitch=(edges)) 

        filled = voxelized.fill(method=fill_method)   
        voxels = filled.matrix.astype(bool)   

        if disp_infos:
            print("Voxel grid shape:", voxels.shape,)

        # skeleton_voxels = skeletonize_3d(voxels)
        if skl_method=='kimimaro':
            import kimimaro
            skeleton_voxels = kimimaro.skeletonize(
                voxels.astype(np.uint8),
                teasar_params={
                    'scale': 4,
                    'const': 500,
                }
            )
        # elif skl_method=='medial_axis':
        #     skeleton, dist = medial_axis(voxels, return_distance=True)

        else:
            skeleton_voxels = skeletonize(voxels)
        skeleton_coords = np.argwhere(skeleton_voxels)
        if tf_largest:
            labeled = label(skeleton_voxels)
            largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            skeleton_voxels = labeled == largest_label 

        if skeleton_coords.size == 0:
            print('--------------------',voxel_resolution,target_number_of_triangles)
            raise ValueError("No skeleton found — something went wrong in voxelization.")
 
        min_bound = self.mesh.bounds[0]
        pitch = filled.pitch
        self.skeleton_points = skeleton_coords * pitch + min_bound
 
        tree = KDTree(self.skeleton_points) 
        distances, self.skl_index = tree.query(vertices)
        self.distances=distances
        self.dist_norm = (distances - distances.min()) / (distances.max() - distances.min())

    def mapping(self):
        self.skeleton_to_vertices = defaultdict(list) 
        for vertex, skl_idx in zip(self.vertices, self.skl_index):
            if skl_idx not in self.skeleton_to_vertices:
                self.skeleton_to_vertices[skl_idx] = []
            self.skeleton_to_vertices[skl_idx].append(vertex) 

    def mapping_inv(self, skeleton_vec):
        if not hasattr(self, 'skeleton_to_vertices'): 
            self.mapping()
        vertices_mapped=[]
        for vec in skeleton_vec:
            vertices_mapped.extend(self.skeleton_to_vertices[vec])
        return vertices_mapped
    

    def reduce_rhs(self, labels): return labels[KDTree(self.vertices).query(self.mesh.vertices)[1]]

    def reduce_to_full_rhs(self, labels): return labels[KDTree(self.mesh.vertices).query(self.vertices)[0]]

 






def def_mesh_to_skeleton_finder(vertices,faces,
                                interval_voxel_resolution=None,
                                interval_target_number_of_triangles=None,
                                tf_largest=False,
                                disp_infos=True,
                                min_voxel_resolution=20,
                                min_target_number_of_triangles=100,
                                tf_division=False ,
                                alpha_fraction=.5,
                                offset_fraction=0.500,
                                wrap_method='alpha_wrap',
                 tf_resize=True,
                 skl_method='medial_axis',

                                ):
    ktrr=KDTree(vertices) 
    skss={}
    cv=1e100
    n_vert=len(vertices) 
    nskl=None 
    interval_target_number_of_triangles=interval_target_number_of_triangles if interval_target_number_of_triangles is not None else [1]
    if disp_infos:
        print('Start mesh_to_skeleton')
        print('------------------------------------')
        print('interval_target_number_of_triangles =',interval_target_number_of_triangles)
        # print('interval_voxel_resolution           =',interval_voxel_resolution)
        print('len(vertices)                       =',n_vert)
    for vv in interval_target_number_of_triangles:
        # for uu in interval_voxel_resolution:
        #     n_voxel_resolution =uu if not tf_division else max(n_vert//uu,min_voxel_resolution)
            target_number_of_triangles =min(vv,vertices.shape[0]) #if not tf_division else max(n_vert//vv,min_target_number_of_triangles)
            # print('n_voxel_resolution,target_number_of_triangles  =',n_voxel_resolution,target_number_of_triangles)
            nskl=mesh_to_skeleton(vertices=vertices, 
                                  faces=faces, 
                                #   voxel_resolution =n_voxel_resolution,
                                    target_number_of_triangles=target_number_of_triangles, 
                                  tf_largest=tf_largest,
                                  disp_infos=disp_infos,
                                    tf_resize=tf_resize,
                                    skl_method=skl_method,
                 ) 
            distan = ktrr.query(nskl.skeleton_points)[0] 
            cvtmp= np.std(distan) / np.mean(distan)
            if cvtmp<cv:
                cv=cvtmp 
                # uutmp=uu 
                skl=nskl
                best_target_number_of_triangles=target_number_of_triangles,vv
                # best_voxel_resolution=n_voxel_resolution,uu
    if disp_infos:
        print('------------------------------------')
        print('The best target_number_of_triangles =',best_target_number_of_triangles)
        # print('The best voxel_resolution           =',best_voxel_resolution)
    return skl



def get_model(base_features ,vcv_length, model_sufix, add_param=None ):  
    vcv_length =np.asarray(vcv_length).reshape(-1, 1) 
    if model_sufix.startswith("opt"):
        base_features.append(hff.normalize(vcv_length)  )   
    return base_features 




class mapping_skl():
    def __init__(self,vertices,skeleton_points, ):
        self.vertices=vertices 
        tree = KDTree( skeleton_points) 
        _,  skl_index = tree.query(vertices)  
        self.mappk={} 
        for fb,vf in zip(skl_index,vertices): 
            fbv=tuple(skeleton_points[fb])
            if fbv not in self.mappk:
                self.mappk[fbv]=[]
            self.mappk[fbv].append(vf)
 


    def mapping_inv(self, skeleton_points):
        vertices_mapped=[]
        for fb in skeleton_points:
            fbv=tuple(fb) 
            vertices_mapped.extend(self.mappk[fbv]) 
       
        return  np.array(vertices_mapped) 






 
def get_head_neck_mectric(self,metric_save,metrics,dname, name,spine_index,spine_faces,vertices_index_unique,vertices_center, vertices_head_index_set,vertices_neck_index_set,
                            subsample_thre=None,
                            f=None,
                            N=None,
                            num_chunks=None,
                            num_points=None, 
                            line_num_points= None,
                            line_num_points_inter= None,
                            spline_smooth= None,
                            ctl_run_thre=None,
                             node_neck=None,
                              node_head=None, ):
 
        metrics['head_diameter'][dname]= 2*metric_save[name]['limit'][0]
        metrics['neck_diameter'][dname]= 2*metric_save[name]['limit'][2] 


        ctl_tmp=get_center_lines(
                                vertices =self.vertices_00[spine_index], 
                                vertices_center=  vertices_center,
                                subsample_thre=subsample_thre,
                                f=f,
                                N=N,
                                num_chunks=num_chunks,
                                num_points=num_points, 
                                line_num_points= line_num_points,
                                line_num_points_inter= line_num_points_inter,
                                spline_smooth= spline_smooth,
                                ctl_run_thre=ctl_run_thre, 
                                ) 
          
        metrics['spine_length'][dname] = Curve_length(ctl_tmp.vertices_center ) 
        metrics['head_length'][dname]=metrics['spine_length'][dname] 


        head_index=list(set(spine_index).intersection(vertices_head_index_set))
        neck_index=list(set(spine_index).intersection(vertices_neck_index_set)) 


        metrics['spine_vol'][dname]=vol=volume(vertices=self.vertices_00[spine_index],faces=spine_faces)
        metrics['head_vol'][dname]= vol
        mesh=curv_mesh(vertices=self.vertices_00[spine_index],faces=spine_faces)
        mesh.Curvature()
        metrics['spine_area'][dname]=mesh.areas
        metrics['head_area'][dname]=mesh.areas


        vertices_index=spine_index
        facess=spine_faces 
        neck_vertices_index=vertices_index_unique 
        metrics['neck_vol'][dname]=0.
        metrics['neck_area'][dname]=0.
        metrics['neck_length'][dname]=0.
        if node_head is None:
            node_head=Branch_division(
                            cclu=self.cclu,
                            dend=self.dend, 
                            vertices_index=head_index,
                            size_threshold= 2, 
                            )
        if len(node_head.children)<1:
            vertices_index=spine_index
            facess=spine_faces  

            neck_vertices_index=vertices_index_unique
            neck_facess=facess 

        else:
            if node_neck is None:
                node_neck=Branch_division(
                                cclu=self.cclu,
                                dend=self.dend, 
                                vertices_index=neck_index,
                                size_threshold= 2, 
                                )
            if (node_neck.children is not None) and len(node_neck.children)>0:
                neck_vertices_index=node_neck.children[0].vertices_index  
                neck_facess=node_neck.children[0].faces
                neck_vertices_index_unique=node_neck.children[0].vertices_index_unique  
                metrics['neck_vol'][dname]=volume(vertices=self.vertices_00[neck_vertices_index],faces=neck_facess) 
                mesh=curv_mesh(vertices=self.vertices_00[neck_vertices_index],faces=neck_facess)
                mesh.Curvature()
                metrics['neck_area'][dname]=mesh.areas

                ctl_tmp=get_center_lines(
                                        vertices =self.vertices_00[neck_vertices_index], 
                                        vertices_center=  vertices_center,
                                        subsample_thre=subsample_thre,
                                        f=f,
                                        N=N,
                                        num_chunks=num_chunks,
                                        num_points=num_points, 
                                        line_num_points= line_num_points,
                                        line_num_points_inter= line_num_points_inter,
                                        spline_smooth= spline_smooth,
                                        ctl_run_thre=ctl_run_thre, 
                                        ) 
                

                metrics['neck_length'][dname] = Curve_length(ctl_tmp.vertices_center ) 
            else:
                neck_vertices_index=  vertices_index_unique
                neck_facess=vertices_index_unique
                neck_vertices_index_unique=  vertices_index_unique 
            if (node_head.children is not None) and len(node_head.children)>0:
                vertices_index=node_head.children[0].vertices_index
                facess=node_head.children[0].faces
                vertices_index_unique=node_head.children[0].vertices_index_unique 
                metrics['head_vol'][dname]=volume(vertices=self.vertices_00[vertices_index],faces=facess)
                mesh=curv_mesh(vertices=self.vertices_00[vertices_index],faces=facess)
                mesh.Curvature()
                metrics['head_area'][dname]=mesh.areas
                ctl_tmp=get_center_lines(
                                        vertices =self.vertices_00[vertices_index], 
                                        vertices_center=  vertices_center,
                                        subsample_thre=subsample_thre,
                                        f=f,
                                        N=N,
                                        num_chunks=num_chunks,
                                        num_points=num_points, 
                                        line_num_points= line_num_points,
                                        line_num_points_inter= line_num_points_inter,
                                        spline_smooth= spline_smooth,
                                        ctl_run_thre=ctl_run_thre, 
                                        ) 
                

                metrics['head_length'][dname] = Curve_length(ctl_tmp.vertices_center ) 





class pinn_data(pdata):
    def __init__(self, file_path_feat=None,
                 dict_mesh_to_skeleton_finder_mesh=None,
                   **kwargs):
        super().__init__(**kwargs)  
        self.file_path_feat=file_path_feat 
        self.dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh 

 



    def get_pinn_features(self, 
                         feat_paths= None,
                          base_features_list=None, 
                          file_path=None,
                          file_path_feat=None, 
						thre_gauss=None,
						thre_mean=None,
                        head_neck=False,
						shaft_path_init=None,
                        spine_path_init=None,
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                          ): 
        file_path = file_path or self.file_path
        file_path_feat = file_path_feat or self.file_path_feat 
        thre_gauss=thre_gauss or self.thre_gauss
        thre_mean=thre_mean or self.thre_mean  
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter

 
        self.mean_curv = np.loadtxt(os.path.join(file_path_feat, self.txt_mean_curv_smooth), dtype=float)
        self.gauss_curv= np.loadtxt(os.path.join(file_path_feat, self.txt_gauss_curv_smooth), dtype=float)
        self.skl_curv  = np.loadtxt(os.path.join(file_path_feat, self.txt_skl_distance), dtype=float)
        # self.skl_curv_con= np.loadtxt(os.path.join(file_path_feat, self.txt_skl_distance_con), dtype=float)
        # self.skl_curv_imp_con = Impute_intensity(self.skl_curv_con)  
        self.skl_curv_imp =skl_curv_imp= Impute_intensity(self.skl_curv) 
        self.mean_curv_imp=mean_curv_imp=Threshold_curv(curv=Impute_intensity(self.mean_curv) , thre=thre_mean) 
        self.gauss_curv_imp=gauss_curv_imp=Threshold_curv(curv=Impute_intensity(self.gauss_curv) ,thre=thre_gauss) 


        self.mean_sq_curv =np.loadtxt(os.path.join(file_path_feat, self.txt_mean_sq_curv_smooth), dtype=float)
        self.gauss_sq_curv =np.loadtxt(os.path.join(file_path_feat, self.txt_gauss_sq_curv_smooth), dtype=float) 
        self.mean_sq_curv_imp=Threshold_curv(curv=Impute_intensity(self.mean_sq_curv), thre=thre_mean) 
        self.gauss_sq_curv_imp=Threshold_curv(curv=Impute_intensity(self.gauss_sq_curv),thre=thre_gauss)  


        self.mean_curv_init   =np.loadtxt(os.path.join(file_path_feat, self.txt_mean_curv_init), dtype=float)
        self.gauss_curv_init=np.loadtxt(os.path.join(file_path_feat, self.txt_gauss_curv_init), dtype=float)
        self.mean_curv_init_imp=Threshold_curv(curv=Impute_intensity(self.mean_curv_init) , thre=thre_mean) 
        self.gauss_curv_init_imp=Threshold_curv(curv=Impute_intensity(self.gauss_curv_init) , thre=thre_mean) 

        mean_sq_curv_init =np.loadtxt(os.path.join(file_path_feat, self.txt_mean_sq_curv_init), dtype=float)
        gauss_sq_curv_init =np.loadtxt(os.path.join(file_path_feat, self.txt_gauss_sq_curv_init), dtype=float) 
        self.mean_sq_curv_init_imp=Threshold_curv(curv=Impute_intensity(mean_sq_curv_init), thre=thre_mean) 
        self.gauss_sq_curv_init_imp=Threshold_curv(curv=Impute_intensity(gauss_sq_curv_init),thre=thre_gauss)  
 
        curv_k=mean_curv_imp+np.abs(mean_curv_imp**2-gauss_curv_imp)**(1/2)
        curv_v=mean_curv_imp-np.abs(mean_curv_imp**2-gauss_curv_imp)**(1/2)
 
        self.base_features_dict['curv_k']['values']=curv_k
        self.base_features_dict['curv_v']['values']=curv_v
        self.base_features_dict['curv_k2']['values']=curv_k**2
        self.base_features_dict['curv_v2']['values']=curv_v**2
        self.base_features_dict['curv_kv']['values']=curv_v*curv_k  
        self.base_features_dict['curv_kv22']['values']=(curv_v*curv_k)**2
 


        self.base_features_dict['gauss']['values']=gauss_curv_imp
        self.base_features_dict['mean']['values']=mean_curv_imp
        self.base_features_dict['gauss_sq']['values']=self.gauss_sq_curv_imp #gauss_curv_imp**2#
        self.base_features_dict['mean_sq']['values']=self.mean_sq_curv_imp  #mean_curv_imp**2#
        self.base_features_dict['gauss_qd']['values']=gauss_curv_imp**4
        self.base_features_dict['mean_qd']['values']=mean_curv_imp**4

        self.base_features_dict['igauss']['values']=self.gauss_curv_init_imp
        self.base_features_dict['imean']['values']=self.mean_curv_init_imp
        self.base_features_dict['igauss_sq']['values']=self.gauss_sq_curv_init_imp
        self.base_features_dict['imean_sq']['values']=self.mean_sq_curv_init_imp
        # print('[[[[[[[]]]]]]]',self.base_features_dict.keys()) 
        self.base_features_dict['skl']['values']=hff.normalize( (skl_curv_imp)) 
        for nam,pat in zip(['kmean_smooth','kmean_init'],[self.txt_skl_distance,self.txt_skl_distance_org]):
            path_org=os.path.join(file_path_feat, pat)
            # print('[[[[[[[path_org]]]]]]]================================',kmean_n_run, path_org) 
            if os.path.exists(path_org): 
                ipath_org = Impute_intensity(np.loadtxt(path_org, dtype=float) ) 
                for rf in self.kmean_list: 
                    path_ex = os.path.join(file_path_feat, f'{nam}_{rf}.txt')
                    get_feat = param_dic['tf_restart']['get_pinn_features'] 
                    if os.path.exists(path_ex) and not get_feat: 
                        # print(' ============LOADING========',f'{nam}_{rf}', )
                        self.base_features_dict[f'{nam}_{rf}']['values']=np.loadtxt(path_ex,dtype=float).reshape(-1,1)
                    else:
                        # print(' ====================',f'{nam}_{rf}', ) 
                        self.base_features_dict[f'{nam}_{rf}']['values']=  get_kmean_mode(ipath_org,n_clusters=rf,
                                                                                        kmean_max_iter=kmean_max_iter,
                                                                                        n_runs=kmean_n_run).reshape(-1,1) 
             
        path_org=os.path.join(file_path_feat, 'skl_shaft_distance.txt') 
        if os.path.exists(path_org):  
            # print('[[[[[[[]]]]]]]',path_org)
            self.base_features_dict['skl_shaft_distance']['values']=np.loadtxt(path_org, dtype=float).reshape(-1,1)
            
        if base_features_list is not None:
            base_featuress=base_features_list 
        else:
            base_featuress=self.base_features_dict.keys() 
        base_features=[]
        for ii in base_featuress: 
            ngn=self.base_features_dict[ii]['values']
            base_features.append(ngn) 

        for ii in self.base_features_dict.keys():
            if 'values' not in self.base_features_dict[ii]:
                continue
            ngn=self.base_features_dict[ii]['values'] 
            patr=os.path.join(self.file_path_feat,f'{ii}.txt') 
            ff='%d' if ii.startswith(('kmean_init','kmean_smooth')) else '%f'
            np.savetxt(patr,ngn,fmt=ff) 
        
        for add_feat in feat_paths:
            if os.path.exists(add_feat):   
                vcv_length =np.asarray(np.loadtxt(add_feat)).reshape(-1, 1) 
                if self.model_sufix.startswith("opt"):
                    base_features.append(hff.normalize(vcv_length)  )  
            else: 
                print(f'I activated {self.model_sufix}')  
        print('[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]',[len(nn) for nn in base_features])
        print('[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]',feat_paths)
        return base_features


    def get_central_data(
                        self,
                        line_num_points_shaft=None,
                        line_num_points_inter_shaft=None,
                        spline_smooth_shaft=None,
                        shaft_path=None,
                        ctl_run_thre=1,
                        smooth_tf=False,
                        vertices_center=None,
                        ): 
        dend=self.dend
        shaft_path=shaft_path or self.shaft_path_pre or self.shaft_path
        line_num_points_shaft=line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft=line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft=spline_smooth_shaft or self.spline_smooth_shaft

        # vertices_index_shaft=shaft_index=spn.children[0].vertices_index  
        print('shaft_path----------->>>>>>. get_central_data----------->>>>>>.',shaft_path) 
        print('spline_smooth_shaft',spline_smooth_shaft)
        print('line_num_points_inter_shaft',line_num_points_inter_shaft)
        print('line_num_points_shaft',line_num_points_shaft)
        vertices_0=dend.vertices 
        vcv_length_path=os.path.join( shaft_path, self.txt_shaft_vcv_length)
        vertices_center_path=os.path.join( shaft_path, self.txt_shaft_vcv_vertices_center) 
        # print('vcv destination----------->>>>>>.',vcv_length_path) 
        # if not (os.path.exists(vertices_center_path) and os.path.exists(vcv_length_path)): 
        vertices_00 = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_old), dtype=float)  
        self.mapp=mappings_vertices(vertices_0=dend.vertices) 
        self.kdtree=KDTree(vertices_0) 
        self.cclu=cluster_class(faces_neighbor_index=dend.vertex_neighbor) 
        if os.path.exists(os.path.join(shaft_path, self.txt_shaft_index)):
            shaft_index = np.loadtxt(os.path.join(shaft_path, self.txt_shaft_index), dtype=int)    
            # faces = np.loadtxt(os.path.join(shaft_path, self.txt_faces), dtype=int)

            # if not os.path.exists(vertices_center_path):
            #     return
            line_num_points_shaft = max(vertices_00.shape[0] // line_num_points_shaft, 10) 
            # print('line_num_points_shaft----------->>>>>>.[[]]',line_num_points_shaft,line_num_points_inter_shaft,spline_smooth_shaft) 
            if vertices_center is not None:  
                # if not (os.path.exists(vcv_length_path) and os.path.exists(vertices_center_path)):  
                    ctl = center_curvature(vertices=vertices_center,  
                                        line_num_points=line_num_points_shaft,
                                        line_num_points_inter=line_num_points_inter_shaft,
                                        spline_smooth=spline_smooth_shaft,
                                        smooth_tf=smooth_tf,)             
                    dist = KDTree(ctl.vertices_center).query(vertices_00 )[0] 
                    np.savetxt(vcv_length_path, dist, fmt='%f')  
                    # np.savetxt(vcv_length_path, ctl.vcv_length, fmt='%f') 
                    self.vcv_length= np.loadtxt(vcv_length_path,ndmin=1) 
                    np.savetxt(vertices_center_path, ctl.vertices_center, fmt='%f') 
                    # print('=========== im in',vcv_length_path)
            # else:
                    
            #     ctl = center_curvature(vertices=vertices_00, 
            #                         vertices_index=shaft_index,
            #                         line_num_points=line_num_points_shaft,
            #                         line_num_points_inter=line_num_points_inter_shaft,
            #                         spline_smooth=spline_smooth_shaft,
            #                         smooth_tf=smooth_tf,)  

            self.shaft_index=shaft_index=np.loadtxt(os.path.join( shaft_path, self.txt_shaft_index),dtype=int)
            self.vertices_center = np.loadtxt(vertices_center_path) 
            self.vcv_length= np.loadtxt(vcv_length_path,ndmin=1)  
            self.clu_pca=clust_pca(vertices_0,shaft_index,
                            vertices_center= self.vertices_center,
                            vcv_length=self.vcv_length)   

        # print('shaft_path----------->>>>>>. get_central_data----------->>>>>>.DONE' ) 
  
    def save_pinn_data(self,
                        file_path=None,
                        file_path_feat=None,
                        spine_path=None,
                        shaft_path=None,
                        thre_gen=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution =None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                                      ):
        file_path = file_path or self.file_path
        file_path_feat = file_path_feat or self.file_path_feat
        shaft_path=shaft_path or self.shaft_path
        spine_path=spine_path or self.spine_path 
        thre_gen=thre_gen or self.thre_gen
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles 
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh

        print('file_path',file_path)
        voxel_resolution=voxel_resolution or self.voxel_resolution
        vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_old), dtype=float) 
        faces =       np.loadtxt(os.path.join(file_path, self.txt_faces), dtype=int)  
        dend = curv_mesh(vertices=vertices_00, faces=faces)  

        dend.Gauss_curv()
        dend.Mean_curv() 
        
        gauss_curv=Threshold_curv(curv=dend.gauss_curv,thre=thre_gen)
        mean_curv=Threshold_curv(curv=dend.mean_curv,thre=thre_gen)  
        gauss_sq_curv=Threshold_curv(curv=dend.gauss_curv*dend.gauss_curv,thre=thre_gen)
        mean_sq_curv=Threshold_curv(curv=dend.mean_curv*dend.mean_curv,thre=thre_gen)  
        np.savetxt(os.path.join(file_path_feat, self.txt_gauss_curv_init),gauss_curv, fmt='%f') 
        np.savetxt(os.path.join(file_path_feat, self.txt_mean_curv_init),mean_curv, fmt='%f') 
        gauss_sq_curv=Threshold_curv(curv=dend.gauss_curv*dend.gauss_curv,thre=thre_gen)
        mean_sq_curv=Threshold_curv(curv=dend.mean_curv*dend.mean_curv,thre=thre_gen)  
        np.savetxt(os.path.join(file_path_feat, self.txt_gauss_sq_curv_init),gauss_sq_curv, fmt='%f') 
        np.savetxt(os.path.join(file_path_feat, self.txt_mean_sq_curv_init),mean_sq_curv, fmt='%f') 
        # np.savetxt(os.path.join(file_path, self.txt_faces_class_faces), faces_class_faces, fmt='%d') 
        # np.savetxt(os.path.join(file_path, self.txt_vertex_neighbor),vertex_neighbor, fmt='%d')  
        # print('[[[[[[[[[[[]]]]]]]]]]]',[len(mm) for mm in [gauss_sq_curv,mean_sq_curv,]])

        vertices_0  = np.loadtxt(os.path.join(file_path, self.txt_vertices_0), dtype=float)
        # vertices_0 -= np.mean(vertices_0, axis=0)
        dend = curv_mesh(vertices=vertices_0, faces=faces ) 
        if not os.path.exists(os.path.join(file_path_feat,self.pkl_vertex_neighbor)): 
            dend.Vertices_neighbor() 
            with open(os.path.join(file_path_feat,self.pkl_vertex_neighbor), "wb") as file:
                pickle.dump(dend.vertex_neighbor, file)
        else:        
            with open(os.path.join(file_path_feat,self.pkl_vertex_neighbor) , "rb") as file: 
                dend.vertex_neighbor=pickle.load(file)
        # if not os.path.exists(os.path.join(file_path, self.txt_gauss_curv_smooth)):
        #     if os.path.exists(os.path.join(file_path, self.txt_vertices_0)):
        dend.Gauss_curv()
        dend.Mean_curv() 
        self.dend=dend
        gauss_curv=Threshold_curv(curv=dend.gauss_curv,thre=thre_gen)
        mean_curv=Threshold_curv(curv=dend.mean_curv,thre=thre_gen) 
        np.savetxt(os.path.join(file_path_feat, self.txt_gauss_curv_smooth),gauss_curv, fmt='%f') 
        np.savetxt(os.path.join(file_path_feat, self.txt_mean_curv_smooth),mean_curv, fmt='%f') 
        gauss_sq_curv=Threshold_curv(curv=dend.gauss_curv*dend.gauss_curv,thre=thre_gen)
        mean_sq_curv=Threshold_curv(curv=dend.mean_curv*dend.mean_curv,thre=thre_gen)  
        np.savetxt(os.path.join(file_path_feat, self.txt_gauss_sq_curv_smooth),gauss_sq_curv, fmt='%f') 
        np.savetxt(os.path.join(file_path_feat, self.txt_mean_sq_curv_smooth),mean_sq_curv, fmt='%f')  




    def get_shaft_pred(self,   
                        rhs, 
                        path_train,
                        pre_portion=None, 
                        weight=None,
                        weights=None,
                        seg_dend='full',
                        zoom_thre=25,
                        skip_first_n=1,
                        skip_mid_n=None,
                        skip_end_n=10,
                        subdivision_thre=3, 
                        subsample_thre=.02,
                        f=0.99,
                        N=10,
                        num_chunks=100, 
                        line_num_points=None,
                        line_num_points_inter=None,
                        spline_smooth=None,
                        num_points=50,
                        ctl_run_thre=0,
                        size_threshold=None ,
                        end_thre=40,
                        get_refine_=False,
                        dest_path='dest_spine_path',
                        spine_fraction=3,
                        shaft_thre=1/4, 
                        gauss_threshold=10, 
                        smooth_tf=False,
                        neck_lim=0,
                        get_data_txt=True,
                        reconstruction_tf=False,
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        file_path_feat=None,
                        file_path=None,
                        tf_skl_shaft_distance=False, 
                        ):             
        from scipy.spatial import cKDTree
        file_path_feat = file_path_feat or self.file_path_feat
        file_path = file_path or self.file_path 
        line_num_points=line_num_points or self.line_num_points_shaft
        line_num_points_inter=line_num_points_inter or self.line_num_points_inter_shaft
        spline_smooth=spline_smooth or self.spline_smooth_shaft    
        shaft_path = self.path_file[path_train['dest_shaft_path']]   
        spine_path = self.path_file[path_train['dest_shaft_path']]    
        size_threshold=size_threshold or self.size_threshold 
 
  
        nng=np.loadtxt(os.path.join(self.file_path,'vertices_0.txt'),dtype=float)
        skl_vertices=np.loadtxt(os.path.join(file_path_feat, self.txt_skl_vertices),dtype=float) 
        skl_index = cKDTree(skl_vertices).query(nng)[1].flatten()
        # skl_index=dtree[1]#np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_index),dtype=int)  
        inten=-1*np.ones_like(rhs[:,0]) 
        if weight is None:
            siz=rhs.shape[1]
            weight=1/siz*np.ones(siz)  
        dend=self.dend  
        if len(nng) != len(rhs):
            hh,jj=len(nng),len(rhs)
            print(f'====================={hh}        {jj}') 
        weig=np.array([weight,0.81])
        rhs0=weig*rhs.copy() #rhs.copy()#
        msh_tmp=model_shaft(dend,rhs0, stop_index=1,size_threshold=size_threshold,shaft_thre=shaft_thre,uniq_rem_lim=neck_lim) 
        msh_tmp.get_spine_node() 
        masa_tmp=len(msh_tmp.node_spine.children) 
        msh=msh_tmp
        print(weig)
        print(f'get_shaft_pred: -- Weight {weig}| Spine Total Number = {masa_tmp}| size_threshold = {size_threshold}') 
        self.msh=msh
        shaft_index=msh.shaft_index
        shaft_faces=msh.shaft_faces
        shaft_index_unique=msh.shaft_index_unique 


        if get_data_txt: 
            predicted_labels=msh.rhs_vals
            for label in range(rhs.shape[1]): 
                self.vertices_approx_index=vertices_approx_index = np.where(predicted_labels == label)[0] 
                inten[vertices_approx_index]=label 
    
            cluster = np.ones(dend.n_vert )
            cluster[msh.shaft_index]=0
            intensity_spines_cluster=msh.rhs_vals 


 
            np.savetxt(os.path.join(spine_path,f'intensity_{pre_portion}_segm.txt'), inten, fmt='%d')
            np.savetxt(os.path.join(shaft_path,f'intensity_{pre_portion}_segm.txt'), inten, fmt='%d') 
            # dest_spine_path_pre_init=self.path_file[path_train['dest_spine_path_pre_init']]
            # np.savetxt(os.path.join( dest_spine_path_pre_init,f'intensity_{pre_portion}_segm.txt'),inten, fmt='%d')

    

            np.savetxt(os.path.join(spine_path, 'intensity_spines_segment_shaft.txt'), cluster, fmt='%d') 
            np.savetxt(os.path.join(shaft_path, 'intensity_spines_segment_shaft.txt'), cluster, fmt='%d') 
            # for kk,yuy in enumerate(['sh','sp']):
            #     np.savetxt(os.path.join(shaft_path, f'intensity_spines_logit_{yuy}.txt'), rhs[:,kk], fmt='%f') 

 
            np.savetxt(os.path.join(spine_path, self.txt_shaft_index), shaft_index, fmt='%d') 
            np.savetxt(os.path.join(spine_path, self.txt_shaft_faces), shaft_faces, fmt='%d') 
            np.savetxt(os.path.join(spine_path, self.txt_shaft_index_unique), shaft_index_unique, fmt='%d')

            np.savetxt(os.path.join(shaft_path, self.txt_shaft_index), shaft_index, fmt='%d') 
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_faces), shaft_faces, fmt='%d') 
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_index_unique), shaft_index_unique, fmt='%d')   
            skl_vertices=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_vertices),dtype=float) 
            skl_index=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_index),dtype=int)  
      
            nodevv=msh.node_spine 

            node_io=dendrite_io(txt_save_file=shaft_path,
                                shaft_index=shaft_index ,
                                name=self,
                                itera_start=0,
                                vertices_index=np.arange(self.dend.n_vert,dtype=int),
                                faces=self.dend.faces,
                                skl_vertices=skl_vertices,
                                skl_index=skl_index,
                                size_threshold=size_threshold,)
            node_io.node_to_txt(node=nodevv, intensity_len=dend.n_vert, part=self.name_spine,tf_mesh=False) 
    

            np.savetxt(os.path.join(shaft_path, self.txt_intensity_spines_segment), intensity_spines_cluster, fmt='%d') 
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_index), shaft_index, fmt='%d') 
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_faces), shaft_faces, fmt='%d')   
            skl_vert=skl_vertices[skl_index[shaft_index]]

            nng=np.loadtxt(os.path.join(self.file_path,'vertices_old.txt'),dtype=float)
            distance = cKDTree(skl_vert).query(nng)[0].flatten()
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_vertices_center), skl_vert, fmt='%f')
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_vcv_length), distance, fmt='%f')
            # np.savetxt(os.path.join(shaft_vertices_center_path_dest, self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')

            # np.savetxt(os.path.join(shaft_vertices_center_path_dest, self.txt_shaft_vcv_length), self.vcv_length, fmt='%f') 
            '''
            self.get_central_data(shaft_path=shaft_path,
                                smooth_tf=smooth_tf,
                                vertices_center=skl_vertices[skl_index[shaft_index]])     
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_vcv_length), self.vcv_length, fmt='%f')
            np.savetxt(os.path.join(shaft_path , self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_vcv_length), self.vcv_length, fmt='%f') 
            # np.savetxt(os.path.join(file_path_feat , self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')
            # np.savetxt(os.path.join(spine_path_new , 'shaft_vcv_length.txt'), self.vcv_length, fmt='%f')  

            '''



    def get_wrap_shaft(self,    
                        shaft_path,
                        path_train=None,
                        pre_portion=None, 
                        weight=None,
                        weights=None,
                        seg_dend='full',
                        zoom_thre=25,
                        skip_first_n=1,
                        skip_mid_n=None,
                        skip_end_n=10,
                        subdivision_thre=3, 
                        subsample_thre=.02,
                        f=0.99,
                        N=10,
                        num_chunks=100, 
                        line_num_points=None,
                        line_num_points_inter=None,
                        spline_smooth=None,
                        num_points=50,
                        ctl_run_thre=0,
                        size_threshold=None ,
                        end_thre=40,
                        get_refine_=False,
                        dest_path='data_shaft_path',
                        spine_fraction=3,
                        shaft_thre=1/4, 
                        gauss_threshold=10, 
                        smooth_tf=False,
                        neck_lim=0,
                        get_data_txt=True,
                        reconstruction_tf=False,
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        file_path_feat=None,
                        tf_skl_shaft_distance=False, 
                        ):             
        file_path_feat = file_path_feat or self.file_path_feat
        line_num_points=line_num_points or self.line_num_points_shaft
        line_num_points_inter=line_num_points_inter or self.line_num_points_inter_shaft
        spline_smooth=spline_smooth or self.spline_smooth_shaft
        size_threshold=size_threshold or self.size_threshold 
        from dend_fun_0.get_wrap import get_wrap_o3d,build_mesh
        from scipy.spatial import cKDTree  
        shaft_path=  self.path_file[path_train['data_shaft_path']]  
        dest_path = self.path_file[path_train['dest_shaft_path']] 
  
        # skl_vertices=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_vertices),dtype=float) 
        # skl_index=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_index),dtype=int)  
        shaft_index=np.loadtxt(os.path.join(shaft_path, self.txt_shaft_index), dtype=int)
        shaft_faces=np.loadtxt(os.path.join(shaft_path, self.txt_shaft_faces), dtype=int,ndmin=2)   
        vertices_00 = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_old), dtype=float) 
        faces_00 = np.loadtxt(os.path.join(self.file_path, self.txt_faces), dtype=int) 
        # if tf_skl_shaft_distance and (dict_mesh_to_skeleton_finder is not None):
            # if os.path.exists(os.path.join(dest_path, self.txt_shaft_vcv_length)) and os.path.exists(os.path.join(dest_path, self.txt_shaft_vcv_length)): 
            #     return
        
        # path_train=self.path_train 

        # shaft_index=np.loadtxt(os.path.join(shaft_path, self.txt_shaft_index), dtype=int)
        # shaft_faces=np.loadtxt(os.path.join(shaft_path, self.txt_shaft_faces), dtype=int,ndmin=2)   
        # vertices_00 = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_old), dtype=float) 
        # faces_00 = np.loadtxt(os.path.join(self.file_path, self.txt_faces), dtype=int) 

        dend = curv_mesh(
                        vertices=vertices_00 ,
                        faces=faces_00
                        ) 
        dend.Vertices_neighbor()
        cclu=cluster_class(faces_neighbor_index=dend.vertex_neighbor)  
        node = Branch_division(
            cclu=cclu,
            dend=dend, 
            vertices_index=shaft_index,
            size_threshold=size_threshold,
            stop_index=1,
        ) 
        vertices_index_unique=node.children[0].vertices_index_unique
        sh_index=node.children[0].vertices_index
        sh_faces=node.children[0].faces
        # print('[[[[[[vertices_index_unique]]]]]]',vertices_index_unique) 
        coll=[] if len(vertices_index_unique)==0  else get_unique_class(index_unique=vertices_index_unique,
                                vertex_neighbor=dend.vertex_neighbor,)
            

        mesh_cop=[]
        for idx in coll:
            loop = vertices_00[idx] 
            builder = build_mesh(loop)
            mesh = builder.build_loft_all(num_steps=200, offset=0.00, iterations=10, chunk_size=10)
            mesh_cop.append(mesh) 
        mesh_sh=trimesh.Trimesh(vertices=vertices_00[sh_index],faces=sh_faces)
        mesh_sh+=sum(mesh_cop)
        if len(coll)>0: 
            mesh_sh.update_faces(builder.nondegenerate_faces(mesh_sh))
        mesh_sh.remove_unreferenced_vertices()
        mesh_sh.merge_vertices()  
        mesh_sh.export( os.path.join(shaft_path, 'shaft_index.obj'))  
        # np.savetxt(os.path.join(shaft_path, self.txt_shaft_index),dtree[0], fmt='%f')
        # np.savetxt(os.path.join(shaft_path, self.txt_shaft_faces),nskl.skeleton_points, fmt='%f') 


    def get_skl_shaft_pred(self,    
                        shaft_path,
                        path_train=None,
                        pre_portion=None, 
                        weight=None,
                        weights=None,
                        seg_dend='full',
                        zoom_thre=25,
                        skip_first_n=1,
                        skip_mid_n=None,
                        skip_end_n=10,
                        subdivision_thre=3, 
                        subsample_thre=.02,
                        f=0.99,
                        N=10,
                        num_chunks=100, 
                        line_num_points=None,
                        line_num_points_inter=None,
                        spline_smooth=None,
                        num_points=50,
                        ctl_run_thre=0,
                        size_threshold=None ,
                        end_thre=40,
                        get_refine_=False,
                        dest_path='data_shaft_path',
                        spine_fraction=3,
                        shaft_thre=1/4, 
                        gauss_threshold=10, 
                        smooth_tf=False,
                        neck_lim=0,
                        get_data_txt=True,
                        reconstruction_tf=False,
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        file_path_feat=None,
                        tf_skl_shaft_distance=False, 
                        ):             
        file_path_feat = file_path_feat or self.file_path_feat
        line_num_points=line_num_points or self.line_num_points_shaft
        line_num_points_inter=line_num_points_inter or self.line_num_points_inter_shaft
        spline_smooth=spline_smooth or self.spline_smooth_shaft
        size_threshold=size_threshold or self.size_threshold 
        from dend_fun_0.get_wrap import get_wrap_o3d,get_alpha_wrap
        number_of_points=dict_wrap['number_of_points']
  
        # skl_vertices=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_vertices),dtype=float) 
        # skl_index=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_index),dtype=int)  
        shaft_index=np.loadtxt(os.path.join(shaft_path, self.txt_shaft_index), dtype=int)
        shaft_faces=np.loadtxt(os.path.join(shaft_path, self.txt_shaft_faces), dtype=int,ndmin=2)   
        vertices_00 = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_old), dtype=float) 
        faces_00 = np.loadtxt(os.path.join(self.file_path, self.txt_faces), dtype=float) 
        if tf_skl_shaft_distance and (dict_mesh_to_skeleton_finder is not None):
            # if os.path.exists(os.path.join(dest_path, self.txt_shaft_vcv_length)) and os.path.exists(os.path.join(dest_path, self.txt_shaft_vcv_length)): 
            #     return
             
            self.get_wrap_shaft(shaft_path=shaft_path,
                                size_threshold=size_threshold,
                                tf_skl_shaft_distance=tf_skl_shaft_distance,
                                dict_mesh_to_skeleton_finder=dict_mesh_to_skeleton_finder,
                                path_train=path_train,
                                )
            vertices_0 = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_0), dtype=float)
            # dict_wrap['number_of_points'] = int(max( number_of_points, min(5000,len(shaft_index)//25)))
            mesh_sh=trimesh.load_mesh( os.path.join(shaft_path, 'shaft_index.obj'),process=False)
            
            wrap_method=dict_mesh_to_skeleton_finder['wrap_method']
            if wrap_method=='alpha_wrap':
                simplified_vertices,simplified_faces=get_alpha_wrap(vertices=mesh_sh.vertices,
                                                            faces=mesh_sh.faces,
                                                            **dict_wrap,
                                                            )
            else:
                # vertices_00[shaft_index]
                simplified_vertices,simplified_faces=get_wrap_o3d(vertices=mesh_sh.vertices,
                                                            faces=mesh_sh.faces,
                                                            **dict_wrap,
                                                            ) 
            mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces, process=False)  
            mesh.export(os.path.join(dest_path,'mesh_shaft_wrap.obj'))
            print('---------------i m doing get_shaft_pred----> shaft skeleton',dict_wrap['number_of_points'])
            nskl=def_mesh_to_skeleton_finder(vertices=simplified_vertices,
                                        faces=simplified_faces,
                                        ** dict_mesh_to_skeleton_finder,
                                            )
            # dict_wrap['number_of_points']=int(number_of_points) 
            

            skeleton_p,_,_=order_points_along_pca(nskl.skeleton_points)
            np.savetxt(os.path.join(dest_path , self.txt_skl_shaft_vertices),skeleton_p, fmt='%f')  

            tree = KDTree(skeleton_p)
            dist, indices= tree.query(vertices_00 )
            np.savetxt(os.path.join(dest_path, self.txt_skl_shaft_distance),dist, fmt='%f') 
            mesh_sk=trimesh.Trimesh(vertices=skeleton_p[indices.flatten()],faces=faces_00, process=False ) 
            mesh_sk.export( os.path.join(dest_path ,'skl_shaft_vertices.obj')) 


            self.get_central_data(shaft_path=shaft_path,
                                smooth_tf=smooth_tf,
                                vertices_center=skeleton_p)     
            np.savetxt(os.path.join(dest_path, self.txt_shaft_vcv_vertices_center), self.vertices_center, fmt='%f')
            tree = KDTree( self.vertices_center)
            dist, indices= tree.query(vertices_00 )
            np.savetxt(os.path.join(dest_path, self.txt_shaft_vcv_length), dist, fmt='%f')
            mesh_sk=trimesh.Trimesh(vertices=self.vertices_center[indices.flatten()],faces=faces_00, process=False ) 
            mesh_sk.export( os.path.join(dest_path ,'vcv_shaft_vertices.obj')) 

            # np.savetxt(os.path.join(file_path_feat , self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')
            # np.savetxt(os.path.join(spine_path_new , 'shaft_vcv_length.txt'), self.vcv_length, fmt='%f')  




    def get_head_neck_segss(self,   
                         spine_shaft_txt,
                         path_train,
                         metrics,
                         seg_dend='full',
                        pre_portion=None,
                        zoom_thre=25,
                        skip_first_n=1,
                        skip_end_n=10,
                        subdivision_thre=3, 
                        subsample_thre=.02,
                        f=0.99,
                        N=10,
                        num_chunks=100, 
                        line_num_points=None,
                        line_num_points_inter=None,
                        spline_smooth=None,
                        num_points=50,
                        ctl_run_thre=1,
                        size_threshold=None ,
                        end_thre=40,
                        get_refine_=False,
                        head_neck_path='dest_spine_path', 
                        smooth_tf=False,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        kmean_max_iter=600,
                        n_runs=100,  
                        ):             
        line_num_points=line_num_points or self.line_num_points_shaft
        line_num_points_inter=line_num_points_inter or self.line_num_points_inter_shaft
        spline_smooth=spline_smooth or self.spline_smooth_shaft   
        spine_path_new=self.path_file[path_train[head_neck_path]] 
 
        self.vertices_00=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_old))

        self.get_dend_data()     
        dist_norm=dist_norm_gen=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_distance))
        n_vert=len(dist_norm_gen) 
        pathi=os.path.join(self.file_path_feat, f'kmean_init_4.txt')
        if os.path.exists(pathi):
            dist_norm=np.loadtxt(pathi).astype(int) 
            inten=np.loadtxt(os.path.join(self.file_path_feat, f'kmean_init_10.txt')).astype(int)
        else:
            path_org=os.path.join(self.file_path_feat, self.txt_skl_distance_org)
            dist_norm=get_kmean_mode(Impute_intensity(np.loadtxt(path_org, dtype=float) ) ,n_clusters=4, 
                                                                                  kmean_max_iter=kmean_max_iter,
                                                                                  n_runs=n_runs)  
            inten=get_kmean_mode(Impute_intensity(np.loadtxt(path_org, dtype=float) ) ,n_clusters=10, 
                                                                                  kmean_max_iter=kmean_max_iter,
                                                                                  n_runs=n_runs) 
        
        intensity_org_neck_head =   np.zeros(n_vert)
        intensity_pca={}
        self.metric_save={}
        for key in self.inten_pca:
            intensity_pca[key]=np.zeros(n_vert)
        skl_vectices= np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_vertices_org),dtype=float)
        reg=region_branch(region_index=inten,dend=self.dend,skl_vectices=skl_vectices)
        reg.get_skl_smooth(
                line_num_points=10,
                line_num_points_inter=30,
                spline_smooth=0.75) 
        count= hff.loadtxt_count(os.path.join(spine_path_new,self.txt_spine_count)) 
        mmm=count.ndim
        count=count if mmm==2 else count.reshape(-1,1)
        for idx in range(count.shape[0]): 
            ii=count[idx,0]
            name=f'{ii}_{count[idx,1]}' if mmm==2 else f'{ii}'
            dname=f'{self.dend_first_name}_sy{name}'

            if ii<0: 
                name=f'0_0' 
                dname=f'{self.dend_first_name}_sy{name}' 
                spine_index=self.dend.vertices
                spine_faces=self.dend.faces 
                np.savetxt(os.path.join(spine_path_new, f'{self.name_spine}_{self.name_index}_{name}.txt'), spine_index, fmt='%d') 
                np.savetxt(os.path.join(spine_path_new, f'{self.name_spine}_{self.name_faces}_{name}.txt'), spine_faces, fmt='%d')
                np.savetxt(os.path.join(spine_path_new, f'spine_{self.name_count}.txt'), [[0,0]], fmt='%d')  
            else:
                spine_index = np.loadtxt(os.path.join(spine_path_new, f'{self.name_spine}_{self.name_index}_{name}.txt'),dtype=int)
                spine_faces  = np.loadtxt(os.path.join(spine_path_new, f'{self.name_spine}_{self.name_faces}_{name}.txt'),dtype=int) 
                vertices_index_unique=np.loadtxt(os.path.join(spine_path_new, f'{self.name_spine_index_unique}_{name}.txt'),dtype=int)
                loo=os.path.join( spine_path_new,f'{self.name_spine}_{self.name_center}_curv_{name}.txt')
                if os.path.exists(loo):
                    vertices_center=np.loadtxt(loo,dtype=float) 
                else:
                    continue 
                vecty=dist_norm[spine_index]
                vectyy=reg.intensity_thickness[spine_index]
                arg_ver=np.argsort(vecty)
                bvg=2*np.ones(len(spine_index)) 
                bvg[vecty<=vecty[arg_ver[1]]]=1 
                intensity_org_neck_head[spine_index]=bvg

                self.metric_save[name]={} 
                self.metric_save[name]['limit']=(0,0,0,0) if len(vectyy)<=0 else find_min_max_no_cross(vectyy) 
                vertices_head_index_set = set(np.where(intensity_org_neck_head == 2)[0])
                vertices_neck_index_set = set(np.where(intensity_org_neck_head == 1)[0]) 
                if (len(vertices_center)>3) and (len(spine_index)>0):
                    get_head_neck_mectric(self,self.metric_save,metrics,dname, name,spine_index,spine_faces,vertices_index_unique,vertices_center, vertices_head_index_set,vertices_neck_index_set,
                            subsample_thre=subsample_thre,
                            f=f,
                            N=N,
                            num_chunks=num_chunks,
                            num_points=num_points, 
                            line_num_points= line_num_points,
                            line_num_points_inter= line_num_points_inter,
                            spline_smooth= spline_smooth,
                            ctl_run_thre=ctl_run_thre,
                            #  node_head=node_head,
                            #   node_neck=node_neck,
                                )
            np.savetxt(os.path.join(spine_path_new, 'intensity_head_neck_segm.txt'),intensity_org_neck_head,fmt='%d') 
 




    def get_diane_spines(self,dend_name,dend_first_name,name_spine_id,mesh_org_path_file,shaft_path_file,spines_true_path,k_query,size_threshold=5):
        mesh_shaft=trimesh.load(os.path.join(shaft_path_file,f'{dend_name}_shaft.obj',)) 
        mesh_org=trimesh.load(os.path.join(mesh_org_path_file,f'{dend_name}.obj',)) 

        tree=KDTree(mesh_org.vertices)
        idx=tree.query(mesh_shaft.vertices,k=k_query)[1].flatten()
        vertices_index=list(set(np.arange(len(mesh_org.vertices)))-set(idx))
        vertices_0,faces_0=mesh_org.vertices,mesh_org.faces
        intensity=-20*np.ones_like(vertices_0[:,0])

        shaft_inv=[]
        dend = curv_mesh(
        vertices=mesh_org.vertices,
        faces=mesh_org.faces
        )
        dend.Vertices_neighbor()
        cclu=cluster_class(faces_neighbor_index=dend.vertex_neighbor)  
        node = Branch_division(
            cclu=cclu,
            dend=dend, 
            vertices_index=vertices_index,
            size_threshold=size_threshold,
            # stop_index=1,
        ) 
        dd=0
        count=[]
        if len(node.children)>0:
            for nhn in node.children: 
                if len(nhn.vertices_index)>size_threshold: 
                    shaft_inv.extend(nhn.vertices_index)
                    intensity[nhn.vertices_index]=dd
                    dend_path_original_m=os.path.join(spines_true_path,f'{dend_first_name}{name_spine_id}_{dd}.obj')
                    mesh_shaft_wrap = trimesh.Trimesh(vertices=vertices_0[nhn.vertices_index],faces=nhn.faces)
                    mesh_shaft_wrap.export(dend_path_original_m)
                    count.append(dd)
                    dd+=1  








