

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
from dend_fun_0.help_funn import dendrite,  cluster_class,label_cluster,Branch_division,get_intensity,Threshold_curv,Impute_intensity, order_points_along_pca
from dend_fun_0.help_funn import mappings_vertices,clust_pca,dendrite_io,volume,closest_distances_group,find_min_max_no_cross,Curve_length   
 
DTYPE = tf.float32

from sklearn.neighbors import KDTree  

from dend_fun_0.obj_get import Obj_to_vertices,get_obj_filenames_with_indices_2  
from dend_fun_2.metric import center_curvature, get_kmean,get_kmean_mean ,get_center_lines
from dend_fun_0.help_pinn_data_fun import get_model,model_shaft,pinn_data_init
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0" 
 
from dend_fun_0.help_spine_division import region_branch 


import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

 

from collections import defaultdict
import trimesh 
from scipy.spatial import KDTree 
import open3d as o3d 

from skimage.measure import label 
from skimage.morphology import skeletonize  
from scipy.ndimage import label

def get_wrap(vertices,faces,number_of_points=8000,radius=0.9, max_nn=30):

    mesh = trimesh.Trimesh(vertices=vertices , faces=faces)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces) 
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=number_of_points)

    # Estimate and orient normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=10)
 
    mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(    pcd,
                                                                                depth=12,      # adjust octree depth as needed
                                                                                width=0,       # default
                                                                                scale=1.1,     # default scaling
                                                                                linear_fit=False,
                                                                                n_threads=1   ,
                                                                            )


    return np.asarray(mesh_poisson.vertices), np.asarray(mesh_poisson.triangles)

 
 
def get_contraction(vertices,   skeleton_points, alpha=0.5):  
    tree = KDTree(skeleton_points)
    idx = tree.query(vertices )[1]
    nearest_skel = skeleton_points[idx ] 
    return  (1 - alpha) * vertices + alpha * nearest_skel
 

class mesh_resize():
    def __init__(self,vertices,faces,target_number_of_triangles=6000, ):
        self.vertices=vertices 
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces) 
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles) 
        self.mesh = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles)
        ) 

    def reduce_rhs(self, labels): return labels[KDTree(self.vertices).query(self.mesh.vertices)[1]]

    def reduce_to_full_rhs(self, labels): return labels[KDTree(self.mesh.vertices).query(self.vertices)[0]]

 


class mesh_to_skeleton():
    def __init__(self,vertices,faces,target_number_of_triangles=6000,voxel_resolution = 128,tf_largest=False,disp_infos=True,tf_resize=True):
        self.vertices=vertices 
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

        filled = voxelized.fill()   
        voxels = filled.matrix.astype(bool)   

        if disp_infos:
            print("Voxel grid shape:", voxels.shape,)

        # skeleton_voxels = skeletonize_3d(voxels)
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
                                tf_division=False

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
        print('interval_voxel_resolution           =',interval_voxel_resolution)
        print('len(vertices)                       =',n_vert)
    for vv in interval_target_number_of_triangles:
        for uu in interval_voxel_resolution:
            n_voxel_resolution =uu if not tf_division else max(n_vert//uu,min_voxel_resolution)
            target_number_of_triangles =vv if not tf_division else max(n_vert//vv,min_target_number_of_triangles)
            print('n_voxel_resolution,target_number_of_triangles  =',n_voxel_resolution,target_number_of_triangles)
            nskl=mesh_to_skeleton(vertices=vertices, 
                                  faces=faces, 
                                  voxel_resolution =n_voxel_resolution,
                                    target_number_of_triangles=target_number_of_triangles, 
                                  tf_largest=tf_largest,
                                  disp_infos=disp_infos) 
            distan = ktrr.query(nskl.skeleton_points)[0] 
            cvtmp= np.std(distan) / np.mean(distan)
            if cvtmp<cv:
                cv=cvtmp 
                uutmp=uu 
                skl=nskl
                best_target_number_of_triangles=target_number_of_triangles,vv
                best_voxel_resolution=n_voxel_resolution,uu
    if disp_infos:
        print('The best target_number_of_triangles =',best_target_number_of_triangles)
        print('The best voxel_resolution           =',best_voxel_resolution)
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





class pinn_data(pinn_data_init):
    def __init__(self, file_path_feat=None,
                 dict_mesh_to_skeleton_finder_mesh=None,
                   **kwargs):
        super().__init__(**kwargs)  
        self.file_path_feat=file_path_feat 
        self.dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh 


    def get_annotation(self,
						dend_first_name=None,
						spine_path=None,
                        shaft_path=None,
						file_path=None,
						dend_path_original_m=None, 
						radius_threshold=None, 
						disp_infos=None,
						size_threshold=None,
                        file_path_feat=None,
                        dend_path_original_new_smooth=None,
                        dend_path_org_new=None,
                        dend_name=None,
						):
        disp_infos=disp_infos or self.disp_infos
        file_path  = file_path or self.file_path
        file_path_feat = file_path_feat or self.file_path_feat
        spine_path = spine_path or self.spine_path
        shaft_path = shaft_path or self.shaft_path
        dend_path_original_m=dend_path_original_m or self.dend_path_original_m 
        radius_threshold = radius_threshold or self.radius_threshold
        size_threshold=size_threshold or self.size_threshold
        dend_first_name=dend_first_name or self.dend_first_name
        if disp_infos:
            print(f"Annotation path original: {file_path}") 
            print(f"Annotation path original dend_path_original_m: {dend_path_original_m}") 
        from sklearn.neighbors import KDTree
         
        vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_1), dtype=float)
        vertices_0 = np.loadtxt(os.path.join(file_path, self.txt_vertices_0), dtype=float) 
        faces = np.loadtxt(os.path.join(file_path, self.txt_faces), dtype=int) 



        os.makedirs( dend_path_org_new, exist_ok=True)

        mesh = trimesh.Trimesh(vertices=vertices_0, faces=faces) 
        dend_nameu=dend_name or 'mesh'
        mesh.export(os.path.join(dend_path_original_new_smooth,f'{dend_nameu}.obj'))


        mesh = trimesh.Trimesh(vertices=vertices_00, faces=faces) 
        dend_nameu=dend_name or 'mesh'
        mesh.export(os.path.join(dend_path_org_new,f'{dend_nameu}.obj'))




        self.mapp=mappings_vertices(vertices_0=vertices_00)    
        if not os.path.exists(os.path.join(file_path_feat,self.pkl_vertex_neighbor)): 
            dend = curv_mesh(vertices=vertices_00, 
                                    faces=faces,   )
            dend.Vertices_neighbor()  

            with open(os.path.join(file_path_feat,self.pkl_vertex_neighbor), "wb") as file:
                pickle.dump(dend.vertex_neighbor, file)
        else:
            with open(os.path.join(file_path_feat,self.pkl_vertex_neighbor), 'rb') as f:
                vertex_neighbor = pickle.load(f)#             
            dend = curv_mesh(vertices=vertices_00, 
                                    faces=faces, 
                                      vertex_neighbor=vertex_neighbor,  )
        kdtree_00=KDTree(vertices_00)   
        shaft=[]
        intensity_org=-20*np.ones_like(vertices_00[:,0:1])  
        cclu=cluster_class(faces_neighbor_index= dend.vertex_neighbor) 
        obj_indices = get_obj_filenames_with_indices_2(directory=dend_path_original_m,
                                startwith=f'{dend_first_name}{self.name_spine_id}')   
        if len(obj_indices)==0:
            return
        np.savetxt(os.path.join(spine_path,'spine_count_org.txt'),obj_indices, fmt='%s') 
        vertices_index_appr_all=[]
        count=[]
        for ip,nam in enumerate(obj_indices):    
            vertices_sp, _ = Obj_to_vertices(
                            file_path_original= dend_path_original_m,
                            mesh_name=nam, 
                            faces_new_mesh=False, 
                            save=False
            )  
            vertices_index_appr=np.array(list(set(np.concatenate(kdtree_00.query_radius(vertices_sp,radius_threshold))))) 
            print('vertices_index_appr',vertices_sp.shape,len(vertices_index_appr)) 
            if len(vertices_index_appr)> size_threshold:
                nodee=Branch_division(
                                cclu=cclu,
                                dend=dend, 
                                vertices_index=vertices_index_appr,
                                size_threshold= size_threshold,
                                stop_index=1 )
                vertices_index_appr_all.extend(nodee.children[0].vertices_index)
                if len(nodee.children)>0:
                    vertices_index=nodee.children[0].vertices_index
                    faces=nodee.children[0].faces
                    vertices_index_unique=nodee.children[0].vertices_index_unique
                    intensity_org[vertices_index ]=ip  
                    np.savetxt(os.path.join(spine_path, f'{self.name_spine_index}_{ip}.txt'), vertices_index, fmt='%d') 
                    np.savetxt(os.path.join(spine_path, f'{self.name_spine_faces}_{ip}.txt'), faces, fmt='%d') 
                    np.savetxt(os.path.join(spine_path, f'{self.name_spine_index_unique}_{ip}.txt'), vertices_index_unique, fmt='%d')
                    count.append(ip) 
                    mesh=trimesh.Trimesh(vertices=vertices_00[vertices_index],faces=faces) 
                    mesh.export(os.path.join(dend_path_org_new,f'{dend_first_name}{self.name_spine_id}_{ip}.obj') )    
                    mesh=trimesh.Trimesh(vertices=vertices_0[vertices_index],faces=faces) 
                    mesh.export(os.path.join(dend_path_original_new_smooth,f'{dend_first_name}{self.name_spine_id}_{ip}.obj') )  
                else:
                    if disp_infos:
                        print(obj_indices[ip],vertices_index_appr.shape) 

        np.savetxt(os.path.join(spine_path,'spine_count.txt'),np.array(count), fmt='%d') 
        vertices_index_shaft=np.array(list(set(np.arange(vertices_00.shape[0]))-set(vertices_index_appr_all)))
        np.savetxt(os.path.join(spine_path, self.txt_spine_intensity ), intensity_org, fmt='%d')  
        if len(vertices_index_shaft)> size_threshold:
            nodee=Branch_division(
                            cclu=cclu,
                            dend=dend, 
                            vertices_index=vertices_index_shaft,
                            size_threshold= size_threshold,
                            stop_index=1 )
            if len(nodee.children)>0:
                vertices_index=nodee.children[0].vertices_index
                faces=nodee.children[0].faces
                vertices_index_unique=nodee.children[0].vertices_index_unique 
                np.savetxt(os.path.join(shaft_path, self.txt_shaft_index), vertices_index, fmt='%d') 
                np.savetxt(os.path.join(shaft_path, self.txt_shaft_faces), faces, fmt='%d') 
                np.savetxt(os.path.join(shaft_path, self.txt_shaft_index_unique), vertices_index_unique, fmt='%d') 
  
                mesh=trimesh.Trimesh(vertices=vertices_00[vertices_index],faces=faces) 
                mesh.export(os.path.join(dend_path_org_new,f'{dend_first_name}{self.name_spine_id}_shaft.obj') )    

                mesh=trimesh.Trimesh(vertices=vertices_0[vertices_index],faces=faces) 
                mesh.export(os.path.join(dend_path_original_new_smooth,f'{dend_first_name}{self.name_spine_id}_shaft.obj') )    

 

 
    def get_intensity_rhs(self, 
						dend_first_name=None,
						spine_path=None,
                        shaft_path=None,
						file_path=None,
						dend_path_original_m=None, 
						radius_threshold=None, 
						disp_infos=None,
						size_threshold=None,
                        file_path_feat=None,
						):
        disp_infos=disp_infos or self.disp_infos
        file_path  = file_path or self.file_path
        file_path_feat=file_path_feat or self.file_path_feat
        spine_path = spine_path or self.spine_path
        shaft_path = shaft_path or self.shaft_path
        dend_path_original_m=dend_path_original_m or self.dend_path_original_m 
        radius_threshold = radius_threshold or self.radius_threshold
        size_threshold=size_threshold or self.size_threshold
        dend_first_name=dend_first_name or self.dend_first_name
        if disp_infos:
            print(f"get_intensity_head_neck: {file_path}") 

        vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_1), dtype=float)

        intensity  =  np.zeros(vertices_00.shape[0]) 
        intensity_1hot=np.zeros_like(vertices_00[:,:-1],dtype=int)
        count= hff.loadtxt_count(os.path.join(spine_path,self.txt_spine_count))  
        mmm=count.ndim 
        spine_index_all=[]
        count=count if mmm==2 else count.reshape(-1,1) 
        for i in range(count.shape[0]): 
            ii=count[i,0]
            if ii <0:
                continue
            name=f'{ii}_{count[i,1]}' if mmm==2 else f'{count[i,0]}'  
            spine_index = np.loadtxt(os.path.join(spine_path, f'{self.name_spine_index}_{name}.txt'),dtype=int) 
            intensity[spine_index]=1
            spine_index_all.extend(spine_index) 
        intensity_1hot[:,1:2][spine_index_all]=1 
        intensity_1hot[:,0:1][list(set(np.arange(vertices_00.shape[0]))-set(spine_index_all))]=1
        np.savetxt(os.path.join(self.file_path_feat,'intensity_shaft_spine.txt'), intensity, fmt='%d')
        np.savetxt(os.path.join(self.file_path_feat,'intensity_1hot_shaft_spine.txt'), intensity_1hot, fmt='%d')


 

    def get_annotation_resized(self,
                                file_path_resized,
                                shaft_path_resized,
                                dend_path_org_resized,
                                dend_path_org_smooth_resized,
                                dend_first_name=None,
                                spine_path=None,
                                shaft_path=None,
                                file_path=None,
                                file_path_feat=None,
                                dend_path_original_m=None, 
                                radius_threshold=None, 
                                disp_infos=None,
                                size_threshold=None,
                                train_tf=None,
                                thre_target_number_of_triangles=40000, 
                                min_target_number_of_triangles_faction=600000,
                                target_number_of_triangles_faction=1000,
                                voxel_resolution=2064,
                                dict_mesh_to_skeleton_finder_mesh=None,
                                dend_name=None,
						):
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        disp_infos=disp_infos or self.disp_infos
        file_path  = file_path or self.file_path
        file_path_feat = file_path_feat or self.file_path_feat
        spine_path = spine_path or self.spine_path
        shaft_path = shaft_path or self.shaft_path
        dend_path_original_m=dend_path_original_m or self.dend_path_original_m 
        radius_threshold = radius_threshold or self.radius_threshold
        size_threshold=size_threshold or self.size_threshold
        dend_first_name=dend_first_name or self.dend_first_name 
        os.makedirs(file_path_resized, exist_ok=True)
        os.makedirs(shaft_path_resized, exist_ok=True) 
        os.makedirs(os.path.join(file_path_resized,'feat'), exist_ok=True)
        os.makedirs( dend_path_org_resized, exist_ok=True)
        os.makedirs( dend_path_org_smooth_resized, exist_ok=True) 
        if disp_infos:
            print(f"Annotation path original: {file_path}") 
            print(f"================================= get_annotation_resized ====================================") 
 



        vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_1), dtype=float)
        n_vert=vertices_00.shape[0]
        vertices_0 = np.loadtxt(os.path.join(file_path, self.txt_vertices_0), dtype=float) 
        faces = np.loadtxt(os.path.join(file_path, self.txt_faces), dtype=int) 
        # target_number_of_triangles_faction=max(vertices_0.shape[0]//dict_mesh_to_skeleton_finder_mesh['target_number_of_triangles_faction'],
        #                                        dict_mesh_to_skeleton_finder_mesh['min_target_number_of_triangles_faction'],)
        target_number_of_triangles_faction=max(vertices_0.shape[0]// target_number_of_triangles_faction, 2*min_target_number_of_triangles_faction)
        self.skl=skl=mesh_resize(vertices=vertices_00,
                                 faces=faces,
                                 target_number_of_triangles=target_number_of_triangles_faction ,)
 
        vertices_00 = skl.mesh.vertices
        vertices_0 = skl.reduce_rhs(vertices_0)
        faces = skl.mesh.faces  

        mesh = trimesh.Trimesh(vertices=vertices_00, faces=faces) 
        dend_nameu=dend_name or 'mesh'
        mesh.export(os.path.join(dend_path_org_resized,f'{dend_nameu}.obj'))

        np.savetxt(os.path.join(file_path_resized, self.txt_vertices_1), skl.mesh.vertices, fmt='%f') 
        np.savetxt(os.path.join(file_path_resized, self.txt_faces), skl.mesh.faces, fmt='%d') 
        np.savetxt(os.path.join(file_path_resized, self.txt_vertices_0),   vertices_0 , fmt='%f')  
 
        mesh=trimesh.Trimesh(vertices=vertices_0,faces=faces) 
        mesh.export(os.path.join(dend_path_org_smooth_resized,f'{dend_name}.obj') )  
 
        if os.path.exists(os.path.join(spine_path,'spine_count_org.txt')):
            if not os.path.exists(os.path.join(file_path_resized,self.pkl_vertex_neighbor)): 
                dend = curv_mesh(vertices=vertices_00, 
                                        faces=faces,   )
                dend.Vertices_neighbor() 
                vertex_neighbor=dend.vertex_neighbor

                with open(os.path.join(file_path_resized,self.pkl_vertex_neighbor), "wb") as file:
                    pickle.dump(dend.vertex_neighbor, file)
            else:
                with open(os.path.join(file_path_resized,self.pkl_vertex_neighbor), 'rb') as f:
                    vertex_neighbor = pickle.load(f)#             
                dend = curv_mesh(vertices=vertices_00, 
                                        faces=faces, 
                                        vertex_neighbor=vertex_neighbor,  ) 
            cclu=cluster_class(faces_neighbor_index= vertex_neighbor)  
            shaft=[]
            intensity_org=-20*np.ones_like(vertices_00[:,0:1])   
            obj_indices=np.loadtxt(os.path.join(spine_path,'spine_count_org.txt'),dtype=str,ndmin=1) 
            np.savetxt(os.path.join(shaft_path_resized,'spine_count_org.txt'), obj_indices, fmt='%s')  
            vertices_index_appr_all=[]
            count=[]
            for ip,nam in enumerate(obj_indices):       
                if nam.endswith('shaft'):
                    continue
                intensity_sp= np.zeros(n_vert)
                vertices_index=np.loadtxt(os.path.join(spine_path, f'{self.name_spine_index}_{ip}.txt'), dtype=int) 
                intensity_sp[vertices_index]=1
                vertices_index= skl.reduce_rhs(  intensity_sp).astype(int)
                vertices_index=np.where(vertices_index==1)[0]


                cclu.Cluster_index(ln_elm= vertices_index)
                cclu.Cluster_faces()
                cclu.Cluster_faces_unique()  
                if len(cclu.cluster_index)>0:
                    intensity_org[cclu.cluster_index ]=ip  
                np.savetxt(os.path.join(shaft_path_resized, f'{self.name_spine_index}_{ip}.txt'), cclu.cluster_index, fmt='%d') 
                np.savetxt(os.path.join(shaft_path_resized, f'{self.name_spine_faces}_{ip}.txt'), cclu.cluster_faces, fmt='%d') 
                np.savetxt(os.path.join(shaft_path_resized, f'{self.name_spine_index_unique}_{ip}.txt'), cclu.cluster_faces_unique, fmt='%d')

                mesh=trimesh.Trimesh(vertices=vertices_00[cclu.cluster_index],faces=cclu.cluster_faces) 
                mesh.export(os.path.join(dend_path_org_resized,f'{dend_first_name}{self.name_spine_id}_{ip}.obj') )    
                mesh=trimesh.Trimesh(vertices=vertices_0[cclu.cluster_index],faces=cclu.cluster_faces) 
                mesh.export(os.path.join(dend_path_org_smooth_resized,f'{dend_first_name}{self.name_spine_id}_{ip}.obj') )     
 
                
                count.append(ip) 
                vertices_index_appr_all.extend(cclu.cluster_index)


            np.savetxt(os.path.join(shaft_path_resized,'spine_count.txt'),np.array(count), fmt='%d') 
            sett=set(np.arange(vertices_0.shape[0]))
            vertices_index_shaft=list(sett-set(vertices_index_appr_all).intersection(sett)) 
            np.savetxt(os.path.join(shaft_path_resized, self.txt_spine_intensity ), intensity_org, fmt='%d') 
 
            if len(vertices_index_shaft)> size_threshold:  
                cclu.Cluster_index(ln_elm= vertices_index_shaft)
                cclu.Cluster_faces()
                cclu.Cluster_faces_unique() 
                np.savetxt(os.path.join(shaft_path_resized, self.txt_shaft_index), cclu.cluster_index, fmt='%d') 
                np.savetxt(os.path.join(shaft_path_resized, self.txt_shaft_faces), cclu.cluster_faces, fmt='%d') 
                np.savetxt(os.path.join(shaft_path_resized, self.txt_shaft_index_unique), cclu.cluster_faces_unique, fmt='%d')
 
                mesh=trimesh.Trimesh(vertices=vertices_00[cclu.cluster_index],faces=cclu.cluster_faces) 
                mesh.export(os.path.join(dend_path_org_resized,f'{dend_first_name}_shaft.obj') )  
                mesh=trimesh.Trimesh(vertices=vertices_0[cclu.cluster_index],faces=cclu.cluster_faces) 
                mesh.export(os.path.join(dend_path_org_smooth_resized,f'{dend_first_name}_shaft.obj') )    

  
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
                          ): 
        add_feats=[]  
        for vcv in feat_paths:
            if os.path.exists(vcv): 
                add_feat=np.loadtxt(vcv)  
            else:
                add_feat=None 
            add_feats.append(add_feat)
        file_path = file_path or self.file_path
        file_path_feat = file_path_feat or self.file_path_feat 
        thre_gauss=thre_gauss or self.thre_gauss
        thre_mean=thre_mean or self.thre_mean  
 
 
        self.mean_curv= np.loadtxt(os.path.join(file_path_feat, self.txt_mean_curv_smooth), dtype=float)
        self.gauss_curv= np.loadtxt(os.path.join(file_path_feat, self.txt_gauss_curv_smooth), dtype=float)
        self.skl_curv= np.loadtxt(os.path.join(file_path_feat, self.txt_skl_distance), dtype=float)
        # self.skl_curv_con= np.loadtxt(os.path.join(file_path_feat, self.txt_skl_distance_con), dtype=float)
        # self.skl_curv_imp_con = Impute_intensity(self.skl_curv_con)  
        self.skl_curv_imp =skl_curv_imp= Impute_intensity(self.skl_curv) 
        self.mean_curv_imp=mean_curv_imp=Threshold_curv(curv=Impute_intensity(self.mean_curv) , thre=thre_mean) 
        self.gauss_curv_imp=gauss_curv_imp=Threshold_curv(curv=Impute_intensity(self.gauss_curv) ,thre=thre_gauss) 


        self.mean_sq_curv =np.loadtxt(os.path.join(file_path_feat, self.txt_mean_sq_curv_smooth), dtype=float)
        self.gauss_sq_curv =np.loadtxt(os.path.join(file_path_feat, self.txt_gauss_sq_curv_smooth), dtype=float) 
        self.mean_sq_curv_imp=Threshold_curv(curv=Impute_intensity(self.mean_sq_curv), thre=thre_mean) 
        self.gauss_sq_curv_imp=Threshold_curv(curv=Impute_intensity(self.gauss_sq_curv),thre=thre_gauss)  

        self.skl_curv_org= np.loadtxt(os.path.join(file_path_feat, 'skl_distance_org.txt'), dtype=float)
        self.skl_curv_imp_org = Impute_intensity(self.skl_curv_org) 


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
 
        self.base_features_dict['skl']['values']=hff.normalize( (skl_curv_imp)) 
        for rf in self.kmean_list:
            self.base_features_dict[f'kmean_{rf}']['values']= get_kmean(skl_curv_imp,n_clusters=rf,kmean_max_iter=600).reshape(-1,1) 
        for rf in self.kmean_list:
            self.base_features_dict[f'kmean_mean_{rf}']['values']= get_kmean(self.skl_curv_imp_org,n_clusters=rf,kmean_max_iter=600).reshape(-1,1)
            
        if base_features_list is not None:
            base_featuress=base_features_list 
        else:
            base_featuress=self.base_features_dict.keys() 
        base_features=[]
        for ii in base_featuress:
            ngn=self.base_features_dict[ii]['values']
            base_features.append(ngn)
            patr=os.path.join(self.file_path_feat,f'{ii}.txt')  

        for ii in self.base_features_dict.keys():
            ngn=self.base_features_dict[ii]['values'] 
            patr=os.path.join(self.file_path_feat,f'{ii}.txt') 
            np.savetxt(patr,ngn,fmt='%f')

         
        for add_feat in add_feats:
            if add_feat is not None:
                base_features=  get_model(base_features= base_features, 
                                vcv_length=add_feat, 
                                model_sufix=self.model_sufix)  
            else: 
                print(f'I activated {self.model_sufix}')  
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
        # print('spline_smooth_shaft',spline_smooth_shaft)
        vertices_0=dend.vertices 
        vcv_length_path=os.path.join( shaft_path, self.txt_shaft_vcv_length)
        vertices_center_path=os.path.join( shaft_path, self.txt_shaft_vertices_center) 
        # print('vcv destination----------->>>>>>.',vcv_length_path) 
        # if not (os.path.exists(vertices_center_path) and os.path.exists(vcv_length_path)): 
        vertices_00 = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_1), dtype=float)  
        self.mapp=mappings_vertices(vertices_0=dend.vertices) 
        self.kdtree=KDTree(vertices_0) 
        self.cclu=cluster_class(faces_neighbor_index=dend.vertex_neighbor) 
        if os.path.exists(os.path.join(shaft_path, self.txt_shaft_index)):
            shaft_index = np.loadtxt(os.path.join(shaft_path, self.txt_shaft_index), dtype=int)    
            # faces = np.loadtxt(os.path.join(shaft_path, self.txt_faces), dtype=int)

            # if not os.path.exists(vertices_center_path):
            #     return
            line_num_points_shaft = max(vertices_00.shape[0] // line_num_points_shaft, 10) 
            print('line_num_points_shaft----------->>>>>>.[[]]',line_num_points_shaft,line_num_points_inter_shaft,spline_smooth_shaft) 
            if vertices_center is not None:  
                # if not (os.path.exists(vcv_length_path) and os.path.exists(vertices_center_path)):  
                    ctl = center_curvature(vertices=vertices_center,  
                                        line_num_points=line_num_points_shaft,
                                        line_num_points_inter=line_num_points_inter_shaft,
                                        spline_smooth=spline_smooth_shaft,
                                        smooth_tf=smooth_tf,) 
                    np.savetxt(vcv_length_path, ctl.vcv_length, fmt='%f') 
                    self.vcv_length= np.loadtxt(vcv_length_path,ndmin=1)   
                    np.savetxt(vertices_center_path, ctl.vertices_center, fmt='%f') 
                    print('=========== im in',vcv_length_path)
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

        print('shaft_path----------->>>>>>. get_central_data----------->>>>>>.DONE' ) 
  
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

        print('self.file_path',self.file_path)
        voxel_resolution=voxel_resolution or self.voxel_resolution
        self.vertices_00 =vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_1), dtype=float) 
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
        if not os.path.exists(os.path.join(file_path_feat, self.txt_skl_distance)): 
            self.skl=skl=def_mesh_to_skeleton_finder(vertices=vertices_0,
                                                        faces=faces,
                                                ** dict_mesh_to_skeleton_finder_mesh,
                                                )
    # 
            np.savetxt(os.path.join(file_path_feat, self.txt_skl_vectices),skl.skeleton_points, fmt='%f')
            np.savetxt(os.path.join(file_path_feat, self.txt_skl_distance),skl.distances, fmt='%f')  
            np.savetxt(os.path.join(file_path_feat, self.txt_skl_index),skl.skl_index, fmt='%d') 

        if not os.path.exists(os.path.join(file_path_feat, self.txt_skl_distance_org)): 
            vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_1), dtype=float)
            skl=def_mesh_to_skeleton_finder(vertices=vertices_00,
                                            faces=faces,
                                            ** dict_mesh_to_skeleton_finder_mesh,
                                            ) 
            np.savetxt(os.path.join(file_path_feat, self.txt_skl_vectices_org),skl.skeleton_points, fmt='%f')
            np.savetxt(os.path.join(file_path_feat, self.txt_skl_distance_org),skl.distances, fmt='%f')  
            np.savetxt(os.path.join(file_path_feat, self.txt_skl_index_org),skl.skl_index, fmt='%d') 

 

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
                        tf_skl_shaft_distance=False, 
                        ):             
        file_path_feat = file_path_feat or self.file_path_feat
        line_num_points=line_num_points or self.line_num_points_shaft
        line_num_points_inter=line_num_points_inter or self.line_num_points_inter_shaft
        spline_smooth=spline_smooth or self.spline_smooth_shaft  
        data_shaft_path=self.path_file[path_train['data_spine_path']]  
        dest_shaft_path=self.path_file[path_train['dest_shaft_path']]
        shaft_path = self.path_file[path_train['dest_shaft_path']]  
        spine_path_new=self.path_file[path_train['dest_spine_path']] 
        spine_path = self.path_file[path_train['dest_spine_path']] 
        spine_path_new_pre=self.path_file[path_train['dest_spine_path_pre']]  
        shaft_vertices_center_path=self.path_file[self.path_train['data_shaft_path']] 
        shaft_vertices_center_path_dest=self.path_file[path_train['dest_spine_path_center']] 
        dest_spine_path_new=self.path_file[path_train[dest_path]] 
        pre_save_txt= False if dest_path=='dest_spine_path_pre' else True
        size_threshold=size_threshold or self.size_threshold 
 
  
        skl_vertices=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_vectices),dtype=float) 
        skl_index=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_index),dtype=int)  
        inten=-1*np.ones_like(rhs[:,0]) 
        if weight is None:
            siz=rhs.shape[1]
            weight=1/siz*np.ones(siz)  
        dend=self.dend  
        rhs0=np.array([weight,0.81])*rhs.copy()
        msh_tmp=model_shaft(dend,rhs0, stop_index=1,size_threshold=size_threshold,shaft_thre=shaft_thre,uniq_rem_lim=neck_lim) 
        msh_tmp.get_spine_node() 
        masa_tmp=len(msh_tmp.node_spine.children) 
        msh=msh_tmp
        print(f'get_shaft_pred: -- Weight {weight}| Spine Total Number = {masa_tmp}| size_threshold = {size_threshold}') 
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



            dest_spine_path_pre_init=self.path_file[path_train['dest_spine_path_pre_init']] 
            np.savetxt(os.path.join(spine_path,f'intensity_{pre_portion}_segm.txt'), inten, fmt='%d')
            np.savetxt(os.path.join(shaft_path,f'intensity_{pre_portion}_segm.txt'), inten, fmt='%d') 
            np.savetxt(os.path.join( dest_spine_path_pre_init,f'intensity_{pre_portion}_segm.txt'),inten, fmt='%d')

    

            np.savetxt(os.path.join(spine_path, 'intensity_spines_segment_shaft.txt'), cluster, fmt='%d') 
            np.savetxt(os.path.join(shaft_path, 'intensity_spines_segment_shaft.txt'), cluster, fmt='%d') 

 
            np.savetxt(os.path.join(spine_path, self.txt_shaft_index), shaft_index, fmt='%d') 
            np.savetxt(os.path.join(spine_path, self.txt_shaft_faces), shaft_faces, fmt='%d') 
            np.savetxt(os.path.join(spine_path, self.txt_shaft_index_unique), shaft_index_unique, fmt='%d')

            np.savetxt(os.path.join(shaft_path, self.txt_shaft_index), shaft_index, fmt='%d') 
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_faces), shaft_faces, fmt='%d') 
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_index_unique), shaft_index_unique, fmt='%d')   
            skl_vertices=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_vectices),dtype=float) 
            skl_index=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_index),dtype=int)  
      
            self.get_central_data(shaft_path=shaft_path,
                                smooth_tf=smooth_tf,
                                vertices_center=skl_vertices[skl_index[shaft_index]])    
            nodevv=msh.node_spine
            node_io=dendrite_io(txt_save_file=spine_path,
                                shaft_index=shaft_index, 
                                name=self,
                                itera_start=0,
                                vertices_index=np.arange(self.dend.n_vert,dtype=int),
                                faces=self.dend.faces,
                                skl_vertices=skl_vertices,
                                skl_index=skl_index,
                                size_threshold=size_threshold,)
            node_io.node_to_txt(node=nodevv, intensity_len=dend.n_vert, part=self.name_spine) 

            node_io=dendrite_io(txt_save_file=shaft_path,
                                shaft_index=shaft_index ,
                                name=self,
                                itera_start=0,
                                vertices_index=np.arange(self.dend.n_vert,dtype=int),
                                faces=self.dend.faces,
                                skl_vertices=skl_vertices,
                                skl_index=skl_index,
                                size_threshold=size_threshold,)
            node_io.node_to_txt(node=nodevv, intensity_len=dend.n_vert, part=self.name_spine) 
    

            np.savetxt(os.path.join(shaft_path, self.txt_intensity_spines_segment), intensity_spines_cluster, fmt='%d') 
            np.savetxt(os.path.join(shaft_vertices_center_path, self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')
            np.savetxt(os.path.join(shaft_vertices_center_path, self.txt_shaft_index), shaft_index, fmt='%d') 
            np.savetxt(os.path.join(shaft_vertices_center_path, self.txt_shaft_faces), shaft_faces, fmt='%d') 
            np.savetxt(os.path.join(shaft_vertices_center_path, self.txt_shaft_index), shaft_index, fmt='%d') 
            np.savetxt(os.path.join(shaft_vertices_center_path, self.txt_shaft_faces), shaft_faces, fmt='%d')
            np.savetxt(os.path.join(shaft_vertices_center_path_dest, self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')

            np.savetxt(os.path.join(shaft_vertices_center_path_dest, self.txt_shaft_vcv_length), self.vcv_length, fmt='%f') 
            np.savetxt(os.path.join(shaft_path, self.txt_shaft_vcv_length), self.vcv_length, fmt='%f') 
            np.savetxt(os.path.join(shaft_path , self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')
            np.savetxt(os.path.join(file_path_feat, self.txt_shaft_vcv_length), self.vcv_length, fmt='%f') 
            np.savetxt(os.path.join(file_path_feat , self.txt_shaft_vertices_center), self.vertices_center, fmt='%f')
            np.savetxt(os.path.join(spine_path_new , 'shaft_vcv_length.txt'), self.vcv_length, fmt='%f')  
            
            if tf_skl_shaft_distance and (dict_mesh_to_skeleton_finder is not None):
                if not os.path.exists(os.path.join(file_path_feat, self.txt_skl_shaft_vectices)): 
                    vertices_00 = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_1), dtype=float) 
                    vertices_0 = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_0), dtype=float)   
                    simplified_vertices,simplified_faces=get_wrap(vertices=vertices_0[shaft_index],
                                                                faces=shaft_faces,
                                                                **dict_wrap,
                                                                )
                    mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces)  
                    mesh.export(os.path.join(file_path_feat,'mesh_wrap.obj'))

                    nskl=def_mesh_to_skeleton_finder(vertices=simplified_vertices,
                                                faces=simplified_faces,
                                                ** dict_mesh_to_skeleton_finder,
                                                    ) 
                    

                    skeleton_p,_,_=order_points_along_pca(nskl.skeleton_points)
                    np.savetxt(os.path.join(file_path_feat, self.txt_skl_shaft_vectices),skeleton_p, fmt='%f')  
 
                    tree = KDTree(skeleton_p)
                    dist, _= tree.query(vertices_00 )
                    np.savetxt(os.path.join(file_path_feat, self.txt_skl_shaft_distance),dist, fmt='%f') 

  

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
                        ):             
        line_num_points=line_num_points or self.line_num_points_shaft
        line_num_points_inter=line_num_points_inter or self.line_num_points_inter_shaft
        spline_smooth=spline_smooth or self.spline_smooth_shaft  
        data_shaft_path=self.path_file[path_train['data_spine_path']]  
        dest_shaft_path=self.path_file[path_train['dest_shaft_path']]  
        spine_path_new_pre=self.path_file[path_train['dest_spine_path_pre']] 
        shaft_vertices_center_path=self.path_file[path_train['data_spine_path_center']] 
        shaft_vertices_center_path_dest=self.path_file[path_train['dest_spine_path_center']] 
        dest_spine_path_pre_init=self.path_file[path_train['dest_spine_path_pre_init']] 
        spine_path_new=dest_spine_path_new=self.path_file[path_train[head_neck_path]] 
        pre_save_txt= False if head_neck_path=='dest_spine_path_pre' else True
 

        self.get_dend_data()     
        dist_norm=dist_norm_gen=np.loadtxt(os.path.join(self.file_path_feat, self.txt_skl_distance))
        n_vert=len(dist_norm_gen) 
        dist_norm=np.loadtxt(os.path.join(self.file_path_feat, f'kmean_4.txt')).astype(int) 
        
        intensity_org_neck_head =   np.zeros(n_vert)
        intensity_pca={}
        self.metric_save={}
        for key in self.inten_pca:
            intensity_pca[key]=np.zeros(n_vert)
        skl_vectices= np.loadtxt(os.path.join(self.file_path_feat, f'skl_vectices.txt'),dtype=float)
        inten=np.loadtxt(os.path.join(self.file_path_feat, f'kmean_10.txt')).astype(int)
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
 
 