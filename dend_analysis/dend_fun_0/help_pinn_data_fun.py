

import sys
import os
import numpy as np
import dend_fun_0.help_funn as hff  
from scipy.interpolate import splprep, splev 

from scipy.spatial import  distance_matrix
import networkx as nx 

from sklearn.neighbors import KDTree 

from dend_fun_0.curvature import curv_mesh as curv_mesh

from dend_fun_0.help_funn import   cluster_class,label_cluster,Branch_division,get_intensity,Threshold_curv,Impute_intensity,remove_file,loadtxt,loadtxt_count
from dend_fun_0.help_funn import mappings_vertices,clust_pca,dendrite_io,get_color,Refine_vertices_index, volume,closest_distances_group,find_min_max_no_cross
from dend_fun_0.obj_get import get_obj_filenames_with_indices_2,Obj_to_vertices


def get_model(base_features ,vcv_length, model_sufix, add_param=None ): 
    # Ensure inputs are numpy arrays
    vcv_length =np.asarray(vcv_length).reshape(-1, 1)

    if model_sufix.startswith("opt"):
        base_features.append( hff.normalize(vcv_length) )

    if model_sufix == "opt_mean":
        base_features.append(vcv_length - np.mean(vcv_length)) 

    if model_sufix == "opt_std":
        base_features.append((vcv_length - np.mean(vcv_length)) / np.std(vcv_length))

    if model_sufix == "opt_add_param" and add_param is not None:
        add_param = np.asarray(add_param).reshape(-1, 1)
        base_features.append(add_param)
 
    print(f'I activated {model_sufix}')
    return base_features 



class model_shaft:
    def __init__(self,dend,rhs, stop_index=1,size_threshold=10,shaft_thre=1/2,uniq_rem_lim=0):
        # predicted_labelss = np.argmax(rhs0, axis=1)  
        self.dend=dend
        self.rhs=rhs 
        self.rhs_vals=np.argmax(self.rhs, axis=1)
        self.rhs_shaft_index=np.where((self.rhs_vals == 0).astype(int)==1)[0]
        self.size_threshold=size_threshold
        self.stop_index=stop_index
        self.shaft_thre,self.uniq_rem_lim=shaft_thre,uniq_rem_lim

    def get_shaft_index(self, stop_index=None,size_threshold=None):
        self.stop_index=stop_index or self.stop_index 
        self.size_threshold=size_threshold or self.size_threshold
        self.cclu=cluster_class(faces_neighbor_index=self.dend.vertex_neighbor)  
        self.node = Branch_division(
            cclu=self.cclu,
            dend=self.dend, 
            vertices_index=self.rhs_shaft_index,
            size_threshold=self.size_threshold,
            stop_index=self.stop_index,
        )  
        if len(self.node.children)==0: 
            return []  
        else:
            hhhjjj=set()
            for id in self.node.children:
                hhhjjj.update(id.vertices_index)
            return list(hhhjjj)
        
    def get_shaft_node(self, stop_index=None,size_threshold=None,shaft_thre=None,uniq_rem_lim=None):
        # self.stop_index=stop_index or self.stop_index 
        # self.size_threshold=size_threshold or self.size_threshold
        
        self.shaft_thre=shaft_thre or self.shaft_thre
        self.uniq_rem_lim=uniq_rem_lim or self.uniq_rem_lim
        self.get_shaft_index(stop_index=stop_index,size_threshold=size_threshold)
        sz=[]
        cclu=self.cclu
        node=self.node
        if len(node.children)==0:
            cclu.Cluster_index(ln_elm=np.arange(self.dend.n_vert)) 
            cclu.Cluster_faces()
            cclu.Cluster_faces_unique() 
        else:
            szz=self.node.children[0].vertices_index
            lszz=len(szz)*self.shaft_thre
            for id in node.children:
                bvc=list(set(id.vertices_index)-set(id.vertices_index_unique))
                if len(id.vertices_index)>lszz:
                    sz.extend(bvc) 
            cclu.Cluster_index(ln_elm=np.array(sz)) 
            cclu.Cluster_faces()
            cclu.Cluster_faces_unique()    
        self.shaft_index=cclu.cluster_index 
        self.shaft_faces=cclu.cluster_faces 
        self.shaft_index_unique =cclu.cluster_faces_unique 
        print('ppiiuu',[len(nb) for nb in [cclu.cluster_index,self.shaft_index,set(cclu.cluster_index).intersection(set(cclu.cluster_faces_unique ))]])

    def get_spine_node(self, stop_index=None,size_threshold=None,shaft_thre=None,uniq_rem_lim=None):
        self.get_shaft_node(stop_index=stop_index,
                            size_threshold=size_threshold,
                            shaft_thre=shaft_thre,
                            uniq_rem_lim=uniq_rem_lim) 
        dend=self.dend
        cclu=cluster_class(faces_neighbor_index=dend.vertex_neighbor)
        vert_ind=np.array(list(set(np.arange(dend.n_vert,dtype=int))-set(self.shaft_index)))

        cclu.Cluster_index(ln_elm=np.array(vert_ind)) 
        cclu.Cluster_faces()
        cclu.Cluster_faces_unique()    
        uiu=0
        while uiu<self.uniq_rem_lim:
            cclu.Cluster_deleted_index(ln_elm=cclu.cluster_index,ln_rm=cclu.cluster_faces_unique) 
            cclu.Cluster_faces()
            cclu.Cluster_faces_unique()  
            uiu+=1  
        spine_index=cclu.cluster_index  


        self.node_spine = Branch_division(
            cclu=cclu,
            dend=dend, 
            vertices_index=spine_index,
            size_threshold=self.size_threshold,
        )
        shaft_index= list(set(np.arange(self.dend.n_vert))-set(spine_index))

        cclu.Cluster_index(ln_elm=np.array(shaft_index)) 
        cclu.Cluster_faces()
        cclu.Cluster_faces_unique() 
        self.shaft_index=cclu.cluster_index 
        self.shaft_faces=cclu.cluster_faces 
        self.shaft_index_unique =cclu.cluster_faces_unique
        return spine_index


    def get_spine_node_recon(self,pid, stop_index=None,size_threshold=None,shaft_thre=None,uniq_rem_lim=None):
        if not hasattr(self, 'node_spine'): 
            self.get_spine_node( stop_index=stop_index,size_threshold=size_threshold,shaft_thre=shaft_thre,uniq_rem_lim=uniq_rem_lim)
        vertices_recon=[]
        mapp=mappings_vertices(vertices_0=pid.skl.vertices)
        for i,nb in enumerate(self.node_spine.children): 
            index=mapp.Mapping_inverse(pid.skl.mapping_inv( pid.skl.skl_index[nb.vertices_index]))
            print('index',len(index)) 
            self.node_spine.children[i].vertices_index=index
            vertices_recon.extend(index)  

        dend=self.dend
        cclu=cluster_class(faces_neighbor_index=dend.vertex_neighbor)
        self.node_spine = Branch_division(
            cclu=cclu,
            dend=dend, 
            vertices_index=vertices_recon,
            size_threshold=self.size_threshold,
        )
        shaft_index= list(set(np.arange(self.dend.n_vert))-set(vertices_recon))
        print('==----------',len(shaft_index))
        if len(shaft_index)>0:
            cclu.Cluster_index(ln_elm=np.array(shaft_index)) 
            cclu.Cluster_faces()
            cclu.Cluster_faces_unique() 
            self.shaft_index=cclu.cluster_index 
            self.shaft_faces=cclu.cluster_faces 
            self.shaft_index_unique =cclu.cluster_faces_unique
        else:
            self.shaft_index=[0]
            self.shaft_faces=[[0,0,0]] 
            self.shaft_index_unique =[0]

        return vertices_recon

 


def get_model_one_hot(dend,rhs0,num_classes=2,size_threshold=10):
    # predicted_labelss = np.argmax(rhs0, axis=1)  
    cclu=cluster_class(faces_neighbor_index=dend.vertex_neighbor)  
    nodev = Branch_division(
        cclu=cclu,
        dend=dend, 
        vertices_index=np.where((np.argmax(rhs0, axis=1) == 0).astype(int)==1)[0],
        size_threshold=size_threshold,
        stop_index=1,
    ) 
    cluster = np.ones(dend.n_vert )
    if len(nodev.children)>0: 
        cluster[nodev.children[0].vertices_index]=0
    return np.eye(num_classes)[cluster.astype(int)] 


def Curve_length(points):  
    return np.sum(np.linalg.norm(np.diff(np.array(points) , axis=0), axis=1) )  

 
class Sorted_distance_matrix:
    def __init__(self, vert_0: np.ndarray, vert_1: np.ndarray) -> None:
        self.vert_0, self.vert_1 = vert_0, vert_1
        self.distances = distance_matrix(self.vert_0, self.vert_1).flatten()
        self.sorted_indices = np.argsort(self.distances)
        
        ind_y, ind_x = np.meshgrid(range(self.vert_0.shape[0]), range(self.vert_1.shape[0]))
        self.ind_x = ind_x.flatten()[self.sorted_indices]
        self.ind_y = ind_y.flatten()[self.sorted_indices]
 
        self.vert_sorted_0 = self.vert_0[self.ind_x, :]
        self.vert_sorted_1 = self.vert_1[self.ind_y, :]
 
    def Min(self):
        self.vert_min_0,self.vert_min_1=self.vert_0[self.ind_x[0],:],self.vert_1[self.ind_y[0],:]
 
    def Max(self):
        self.vert_max_0,self.vert_max_1=self.vert_0[self.ind_x[-1],:],self.vert_1[self.ind_y[-1],:]
        

def get_aligned_points(vertices, line_num_points=100, spline_smooth=0): 

    G = nx.Graph() 
    for i, pos in enumerate(vertices):
        G.add_node(i, pos=pos)
     
    for i in range(len(vertices)-1):
        dist = np.linalg.norm(vertices[i] - vertices[i+1])
        G.add_edge(i, i+1, weight=dist)
     
    path = list(nx.dfs_preorder_nodes(G, 0))
    path_points = vertices[path]
     
    if vertices.shape[0] > 2: 
        k = max(1, min(3, vertices.shape[0] - 1)) 
        tck, _ = splprep([path_points[:, 0],
                         path_points[:, 1],
                         path_points[:, 2]],
                        s=spline_smooth,
                        k=k,
                        )
         
        u = np.linspace(0, 1, line_num_points)
        aligned_points = np.array(splev(u, tck)).T
         
        curve_length = Curve_length(aligned_points)
        
        return aligned_points, curve_length
     
    else:
        aligned_points = np.linspace(vertices[0], vertices[-1], line_num_points)
        curve_length = Curve_length(aligned_points)
        return aligned_points, curve_length
 

def get_cluster_length_using_center_curve(vertices,vertices_center,
                                          num_chunks=100,
                                          spline_smooth=0,
                                          line_num_points_inter=100,
                                          max_iter=100,): 
    _,gf= closest_distances_group(vertices, vertices_center,num_chunks=num_chunks)
  
    align_pointss,length=get_aligned_points(vertices=vertices_center[np.sort(list(set(gf.reshape(-1))))],
                                    spline_smooth=spline_smooth,
                                    line_num_points= line_num_points_inter,) 
    return length,align_pointss 
 



def get_head_neck_mectric(self,metric_save,metrics,dname, name,spine_index,spine_faces,vertices_index_unique,vertices_center, vertices_head_index_set,vertices_neck_index_set):
 
        metrics['head_diameter'][dname]= metric_save[name]['limit'][0]
        metrics['neck_diameter'][dname]= metric_save[name]['limit'][2]  

        metrics['spine_length'][dname],_=get_cluster_length_using_center_curve(self.vertices_00[spine_index],vertices_center,num_chunks=100)
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

        node=Branch_division(
                        cclu=self.cclu,
                        dend=self.dend, 
                        vertices_index=head_index,
                        size_threshold= 2, 
                        )
        if len(node.children)<1:
            vertices_index=spine_index
            facess=spine_faces  

            neck_vertices_index=vertices_index_unique
            neck_facess=facess 

        else:
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
                metrics['neck_length'][dname],_=get_cluster_length_using_center_curve(self.vertices_00[neck_vertices_index],vertices_center,max_iter=100,num_chunks=100)

            else:
                neck_vertices_index=  vertices_index_unique
                neck_facess=vertices_index_unique 
            node_head=Branch_division(
                            cclu=self.cclu,
                            dend=self.dend, 
                            vertices_index=np.array(list(set(spine_index)-set(neck_vertices_index))),
                            size_threshold= 2, 
                            )
            if (node_head.children is not None) and len(node_head.children)>0:
                vertices_index=node_head.children[0].vertices_index
                facess=node_head.children[0].faces
                vertices_index_unique=node_head.children[0].vertices_index_unique 
                metrics['head_vol'][dname]=volume(vertices=self.vertices_00[vertices_index],faces=facess)
                mesh=curv_mesh(vertices=self.vertices_00[vertices_index],faces=facess)
                mesh.Curvature()
                metrics['head_area'][dname]=mesh.areas
                metrics['head_length'][dname],_=get_cluster_length_using_center_curve(self.vertices_00[vertices_index],vertices_center,max_iter=100,num_chunks=100)
 
 


from sklearn.cluster import KMeans
def get_kmean(dist_norm,n_clusters=4,kmean_max_iter=300):
    pca_radius=np.array(dist_norm).reshape(-1,1)
 
    cluster1 = KMeans(n_clusters=n_clusters, max_iter=kmean_max_iter, n_init='auto')
    cluster1.fit(pca_radius)
    y_pred = cluster1.labels_
    pca_rad=[float(np.mean(pca_radius[:,0][y_pred==i])) for i in range(n_clusters)]
    y_pred_1=np.argsort(pca_rad)

    cluu_rad_ = np.zeros_like(y_pred)  
    for ii,iip in enumerate(y_pred_1):
        cluu_rad_[(y_pred==iip)  ]=ii

    return cluu_rad_




import pickle
from dend_fun_0.get_path import get_name,get_param


class pinn_data_init(get_name,get_param):
    def __init__(self,file_path,
                #  txt_save_file,
                shaft_path,
                spine_path,
                dend_path_original_m,
                dend_first_name,
                name_spine_id=None,
                model_sufix=None,
                shaft_path_pre=None,
                spine_path_pre=None,
                path_mapping=None,
                path_pre=None,
                path_train=None,
                path_file=None,
                pre_portion=None,
                line_num_points_shaft=200,
                line_num_points_inter_shaft=300, 
                spline_smooth_shaft=1, 
                thre_target_number_of_triangles=None,
                voxel_resolution=None,
                        ):
        get_name.__init__(self)  
        get_param.__init__(self,
                         line_num_points_shaft=line_num_points_shaft,
                         line_num_points_inter_shaft=line_num_points_inter_shaft, 
                         spline_smooth_shaft=spline_smooth_shaft,
                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                        voxel_resolution=voxel_resolution, ) 
        # self.txt_save_file=txt_save_file
        self.shaft_path=shaft_path
        self.spine_path=spine_path
        self.file_path=file_path
        self.shaft_path_pre=shaft_path_pre
        self.spine_path_pre=spine_path_pre
        self.dend_path_original_m=dend_path_original_m
        self.dend_first_name=dend_first_name
        self.name_spine_id=name_spine_id
        self.model_sufix=model_sufix
        self.path_mapping=path_mapping
        self.path_pre=path_pre
        self.path_train=path_train
        self.path_file=path_file
        self.pre_portion=pre_portion
        pass
 
    def get_intensity_head_neck(self, 
						dend_first_name=None,
						spine_path=None,
                        shaft_path=None,
						file_path=None,
						dend_path_original_m=None, 
						radius_threshold=None, 
						disp_infos=None,
						size_threshold=None,
						):
        disp_infos=disp_infos or self.disp_infos
        file_path  = file_path or self.file_path
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
        skeleton_points=[]
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
 





    def get_spine_group(self,spine_path):  
        self.get_dend_data()    
        count= hff.loadtxt_count( os.path.join(spine_path,self.txt_spine_count))
        spine_group=[]
        for clustss in count:
            spine_index = np.loadtxt(os.path.join(spine_path, f'{self.name_spine}_{self.name_index}_{clustss}.txt'),dtype=int) 
            spine_group.extend(spine_index)

        self.cclu.Cluster_index(ln_elm= spine_group)
        self.cclu.Cluster_faces()
        self.cclu.Cluster_faces_unique()   
        if len(self.cclu.cluster_index)>self.size_threshold:
            np.savetxt(os.path.join(spine_path,f'{self.name_spine}_group_{self.name_index}.txt'),self.cclu.cluster_index, fmt='%d')
            np.savetxt(os.path.join(spine_path,f'{self.name_spine}_group_{self.name_faces}.txt'),self.cclu.cluster_faces, fmt='%d')
            np.savetxt(os.path.join(spine_path,f'{self.name_spine}_group_index_unique.txt'),self.cclu.cluster_faces_unique, fmt='%d')




    def get_annotation(self,
						dend_first_name=None,
						spine_path=None,
                        shaft_path=None,
						file_path=None,
						dend_path_original_m=None, 
						radius_threshold=None, 
						disp_infos=None,
						size_threshold=None,
						):
        disp_infos=disp_infos or self.disp_infos
        file_path  = file_path or self.file_path
        spine_path = spine_path or self.spine_path
        shaft_path = shaft_path or self.shaft_path
        dend_path_original_m=dend_path_original_m or self.dend_path_original_m 
        radius_threshold = radius_threshold or self.radius_threshold
        size_threshold=size_threshold or self.size_threshold
        dend_first_name=dend_first_name or self.dend_first_name
        file_path_feat=self.file_path_feat
        if disp_infos:
            print(f"Annotation path original: {file_path}") 

        vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_1), dtype=float) 
        faces = np.loadtxt(os.path.join(file_path, self.txt_faces), dtype=int) 
        self.mapp=mappings_vertices(vertices_0=vertices_00)    
        if not os.path.exists(os.path.join(file_path_feat,self.pkl_vertex_neighbor)): 
            dend = curv_mesh(vertices=vertices_00, 
                                    faces=faces,   )
            dend.Vertices_neighbor()  
            vertex_neighbor=dend.vertex_neighbor
            with open(os.path.join(file_path_feat,self.pkl_vertex_neighbor), "wb") as file:
                pickle.dump(dend.vertex_neighbor, file)
        else:
            with open(os.path.join(file_path_feat,self.pkl_vertex_neighbor), 'rb') as f:
                vertex_neighbor = pickle.load(f)#             
            dend = curv_mesh(vertices=vertices_00, 
                                    faces=faces, 
                                      vertex_neighbor=vertex_neighbor,  )
        kdtree_00=KDTree(vertices_00)  
        print('im in',radius_threshold,spine_path)
        shaft=[]
        intensity_org=-20*np.ones_like(vertices_00[:,0:1])  
        cclu=cluster_class(faces_neighbor_index= vertex_neighbor)  
        obj_indices = get_obj_filenames_with_indices_2(directory=dend_path_original_m,
                                startwith=f'{dend_first_name}{self.name_spine_id}')  
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
 


    def get_pinn_rhs(self,
                        pre_portion,
						file_path=None, 
                        radius_threshold=.03): 
        file_path = file_path or self.file_path 
 
        if os.path.exists(os.path.join(file_path, self.txt_vertices_0)):
            vertices_0  = np.loadtxt(os.path.join(file_path, self.txt_vertices_0), dtype=float)
            vertices_0 -= np.mean(vertices_0, axis=0)
            self.vertices_0=vertices_0
 
        if pre_portion=='head_neck': 
            shaft_path_tmp=os.path.join(self.file_path_feat,'intensity_1hot_shaft_neck_head.txt')
            if not os.path.exists(shaft_path_tmp):
                self.get_intensity_head_neck(radius_threshold=radius_threshold) 
        elif pre_portion=='spine': 
            shaft_path_tmp=os.path.join(self.file_path_feat,'intensity_1hot_shaft_spine.txt')
            if not os.path.exists(shaft_path_tmp):
                self.get_intensity_head_neck(radius_threshold=radius_threshold)

        return np.loadtxt(shaft_path_tmp,dtype=int) 

 


    def get_pinn_data(self,
						file_path=None,
						shaft_path=None,
                        spine_path=None,
						weight_positive=.5,
						thre_gauss=None,
						thre_mean=None,
                        head_neck=False,
						shaft_path_init=None,
                        spine_path_init=None, 
                        pre_portion=None,):
        pre_portion=pre_portion or self.pre_portion
        file_path = file_path or self.file_path
        shaft_path=shaft_path or self.shaft_path
        spine_path=spine_path or self.spine_path 
        thre_gauss=thre_gauss or self.thre_gauss
        thre_mean=thre_mean or self.thre_mean   
        vertices_path=os.path.join(file_path, self.txt_vertices_1)
        if not os.path.exists(vertices_path):
            raise FileNotFoundError(f"File {self.txt_vertices_1} not found in {file_path}.")
        self.vertices_00=  np.loadtxt(vertices_path, dtype=float)   
        if os.path.exists(os.path.join(file_path, self.txt_vertices_0)): 
            self.vertices_0=  np.loadtxt(os.path.join(file_path, self.txt_vertices_0), dtype=float)
 
        if pre_portion=='head_neck': 
            shaft_path_tmp=os.path.join(self.file_path_feat,'intensity_1hot_shaft_neck_head.txt')
            if not os.path.exists(shaft_path_tmp):
                self.get_intensity_head_neck(radius_threshold=.03) 
        elif pre_portion=='spine': 
            shaft_path_tmp=os.path.join(self.file_path_feat,'intensity_1hot_shaft_spine.txt')
            if not os.path.exists(shaft_path_tmp):
                self.get_intensity_head_neck(radius_threshold=.03)
        self.rhs_shaft=np.loadtxt(shaft_path_tmp,dtype=int)  
        print(shaft_path_tmp,vertices_path)
        self.get_curv_data() 
 

    def get_dend_data(self,
                    file_path=None,
                    shaft_path=None,
                    spine_path=None,
                    disp_infos=None,
                    file_path_feat=None, ):
        file_path = file_path or self.file_path
        file_path_feat = file_path_feat or self.file_path_feat
        shaft_path=shaft_path or self.shaft_path
        spine_path=spine_path or self.spine_path 
        disp_infos=disp_infos or self.disp_infos

        self.vertices_00=vertices_00 = np.loadtxt(os.path.join(file_path, self.txt_vertices_1), dtype=float)
        if os.path.exists(os.path.join(file_path, self.txt_vertices_0)):
            vertices_0 = np.loadtxt(os.path.join(file_path, self.txt_vertices_0), dtype=float) 
        else:
            print("No smoothed mesh detected. Using the inital data")
            vertices_0=vertices_00
        faces = np.loadtxt(os.path.join(file_path, self.txt_faces), dtype=int) 
        if not os.path.exists(os.path.join(file_path_feat, self.txt_mean_curv_smooth)):
            self.save_pinn_data()
        self.mean_curv=mean_curv  =np.loadtxt(os.path.join(file_path_feat, self.txt_mean_curv_smooth), dtype=float)
        self.gauss_curv=gauss_curv=np.loadtxt(os.path.join(file_path_feat, self.txt_gauss_curv_smooth), dtype=float)   

        if not os.path.exists(os.path.join(file_path_feat, self.txt_skl_distance)):
            self.save_pinn_data()
            self.skl_distance=np.loadtxt(os.path.join(file_path_feat, self.txt_skl_distance), dtype=float)   


        if not os.path.exists(os.path.join(file_path_feat,self.pkl_vertex_neighbor)): 
            self.dend = curv_mesh(vertices=vertices_0, 
                                    faces=faces, 
                                    mean_curv=mean_curv,
                                    gauss_curv=gauss_curv, )
            self.dend.Vertices_neighbor() 
            self.lab_dend =  label_cluster(dend=self.dend,disp=disp_infos,)

            with open(os.path.join(file_path_feat,self.pkl_vertex_neighbor), "wb") as file:
                pickle.dump(self.dend.vertex_neighbor, file)
        else:
            with open(os.path.join(file_path_feat,self.pkl_vertex_neighbor), 'rb') as f:
                vertex_neighbor = pickle.load(f)# 
            self.dend = curv_mesh(vertices=vertices_0, 
                                    faces=faces,
                                    vertex_neighbor=vertex_neighbor,
                                    mean_curv=mean_curv,
                                    gauss_curv=gauss_curv, )
            self.lab_dend =  label_cluster(dend=self.dend,disp=disp_infos,)
        self.cclu=cluster_class(faces_neighbor_index=vertex_neighbor)

 
 
  