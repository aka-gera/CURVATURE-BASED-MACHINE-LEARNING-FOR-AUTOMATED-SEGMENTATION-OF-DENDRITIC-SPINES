import numpy as np 
# from IPython.display import Image 
from dend_fun_0.curvature import curv_mesh as curv_mesh
from tqdm import tqdm  
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KDTree  
from sklearn.decomposition import PCA 





import networkx as nx
import trimesh
from scipy.sparse import coo_matrix

def adjoint(vertices, faces, num_sub_nodes=None,random_tf=True):
    num_vertices = vertices.shape[0]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
 
    edges = mesh.edges_unique
    edges = np.sort(edges, axis=1)
 
    row = edges[:, 0]
    col = edges[:, 1]
    rows = np.concatenate([row, col])
    cols = np.concatenate([col, row])
    data = np.ones(len(rows))

    adj_sparse = coo_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices))
 
    G = nx.from_scipy_sparse_array(adj_sparse) 
    if num_sub_nodes is None:
        sub_nodes = list(G.nodes) 
        subgraph = G.subgraph(sub_nodes) 
        sub_adj_sparse = nx.to_scipy_sparse_array(subgraph, nodelist=sub_nodes)
        sub_adj_dense = sub_adj_sparse.toarray()

        return np.array(sub_adj_dense,dtype=int),sub_nodes
    else: 
        sub_nodes = list(G.nodes)[:num_sub_nodes]

        subgraph = G.subgraph(sub_nodes)
 
        sub_adj_sparse = nx.to_scipy_sparse_array(subgraph, nodelist=sub_nodes)
        sub_adj_dense = sub_adj_sparse.toarray()

        return np.array(sub_adj_dense,dtype=int),sub_nodes 


 

def normalize(vec):
    min_val = np.min(vec)
    max_val = np.max(vec)
    range_val = max_val - min_val

    if range_val == 0:
        return vec 
    else:
        return (vec - min_val) / range_val
    

class mappings_vertices:
    def __init__(self,vertices_0=None,index=None,Dict=None): 
        pass
        if Dict is not None:
            self.mappings=Dict
        else:
            if index is None:
                index=np.arange(vertices_0.shape[0])
            self.mappings={tuple(k): v for k, v in zip(vertices_0.tolist(), index.tolist())}

    def Mapping_inverse(self,vertices): 
        result = []
        for jj in vertices:
            key = tuple(jj)
            if key in self.mappings:
                result.append(self.mappings[key]) 
        return np.array(result)

 
def mmean(vertices, axis=0):
    return  np.mean(vertices, axis=axis) if len(vertices)>0 else np.array([ ])

def loadtxt_2d(file_path, dtype=float): 
    data = np.loadtxt(file_path,dtype=dtype) 
    return data.reshape(1, -1) if data.ndim == 1 else data

def Closest_point_on_line(line_start, line_end, point): 
    line_vec, point_vec= line_end - line_start, point - line_start  
    line_vec_norm = line_vec / np.linalg.norm(line_vec)#np.repmat(,point.shape[0],point[1])  
    return line_start  + np.outer( np.dot(point_vec, line_vec_norm),line_vec_norm)

 
def sigmoid(val):
    return 1/(1+np.exp(-val))
 
def curv_kmean_fun(curv, aa, bb, fun=sigmoid, max_iter=500, n_cluster=2):
    curv = fun(aa * curv + bb)  # Apply the transformation using the provided function
     
    cluster = KMeans(n_clusters=n_cluster, max_iter=max_iter, n_init='auto')
    cluster.fit(curv.reshape(-1, 1))  # Reshaping needed if curv is 1D
    
    # Get the labels and sort them based on the curvature values
    labels = cluster.labels_
    cluster_centers = cluster.cluster_centers_.flatten() 
    min_label = np.argmin(cluster_centers)
     
    new_labels = np.where(labels == min_label, 0, labels + 1)
    new_labels[new_labels == 1] = min_label 
    
    return new_labels



def curv_kmean(curv,max_iter=1000,n_cluster=2):   
    cluster = KMeans(n_clusters=n_cluster, max_iter=max_iter, n_init='auto')
    cluster.fit(curv.reshape(-1, 1))    
    labels = cluster.labels_
    cluster_centers = cluster.cluster_centers_.flatten()
    
    # Find the label corresponding to the minimum cluster center
    min_label = np.argmin(cluster_centers)
    
    # Set the minimum cluster to label 0 and adjust other labels
    new_labels = np.where(labels == min_label, 0, labels + 1)
    new_labels[new_labels == 1] = min_label   
    
    return new_labels


def Impute_intensity(intensity ):  
    if intensity.ndim == 1:
        intensity = intensity.reshape(-1, 1)

    if np.isnan(intensity).any():
        print("Data contains NaNs. Imputing missing values...")
        imputer = SimpleImputer(strategy='mean')
        intensity = imputer.fit_transform(intensity)
    return intensity

def Curv_gauss_mean( dend,gauss_thre=100):
    if dend.gauss_curv is None:
        dend.Gauss_curv() 
        gauss_curv=dend.gauss_curv
        gauss_curv[  (gauss_curv)>gauss_thre]=gauss_thre
        gauss_curv[gauss_curv<=-gauss_thre]=-gauss_thre
    dend.gauss_curv = gauss_curv

    if dend.mean_curv is None:
        dend.Mean_curv() 
    dend.mean_curv =  dend.mean_curv 


def Thre_kmean(gauss_curv,mean_curv,aa_gauss=-1,bb_gauss=-3.45,aa_mean=1,bb_mean=3.6, max_iter=500,n_cluster=2 ): 
    kmean_gauss= curv_kmean_fun(curv= gauss_curv,aa=aa_gauss,bb=bb_gauss, max_iter=max_iter,n_cluster=n_cluster) 
    kmean_mean= curv_kmean_fun(curv= mean_curv,aa=aa_mean,bb=bb_mean, max_iter=max_iter,n_cluster=n_cluster)  

    return  (kmean_mean==1) |(kmean_gauss==0) 

class label_cluster:
    def __init__(self,dend,gauss_curv=None,mean_curv=None,disp=False  ) -> None:
        self.dend=dend
        self.disp=disp
        self.gauss_curv = gauss_curv
        self.mean_curv = mean_curv  
        self.index=[0,1]
 

    def Threshold_kmean(self,vertices_index=None,
                            aa_gauss=-1,
                            bb_gauss=-3.45,
                            aa_mean=1,
                            bb_mean=-3.45, 
                            max_iter=500,
                            n_cluster=2,
                            lab_dend_iter=10,
                            gauss_thre=100,
                            mean_thre=40,
                            curv_abs=False,
                            Imputed_intensity_thre=True):
        if (self.dend.gauss_curv is None) or (self.dend.mean_curv is None):
            self.dend.Threshold_gauss_mean(gauss_thre=gauss_thre,mean_thre=mean_thre)

        if Imputed_intensity_thre:
            self.gauss_curv= Impute_intensity(self.dend.gauss_curv )
            self.mean_curv= Impute_intensity(self.dend.mean_curv)


        if vertices_index is not None:     
            gauss_curv=self.gauss_curv[vertices_index]
            mean_curv=self.mean_curv[vertices_index]
            cluster = np.zeros(len(vertices_index) )
        else:
            gauss_curv=self.gauss_curv 
            mean_curv=self.mean_curv  

            cluster = np.zeros(self.dend.n_vert )
            
        if curv_abs:
            gauss_curv[  (gauss_curv)>gauss_thre]=gauss_thre
            gauss_curv[gauss_curv<=-gauss_thre]=-gauss_thre
            mean_curv[  (mean_curv)>mean_thre]=mean_thre
            mean_curv[mean_curv<=-mean_thre]=mean_thre

        lab_dend = label_cluster(dend=self.dend,disp=False) 
        lab_dend.threshold = Thre_kmean( 
                               gauss_curv=gauss_curv,
                               mean_curv=mean_curv,
                               aa_gauss=aa_gauss,
                               bb_gauss=bb_gauss,
                               aa_mean=aa_mean,
                               bb_mean=bb_mean,
                               n_cluster=n_cluster, 
                               max_iter=max_iter) 
        lab_min=np.array( lab_dend.threshold).sum() 
        for _ in range(lab_dend_iter): 
            lab_dend_tmp=label_cluster(dend=self.dend,disp=False) 
            lab_dend_tmp.threshold = Thre_kmean( 
                               gauss_curv=gauss_curv,
                               mean_curv=mean_curv,
                               aa_gauss=aa_gauss,
                               bb_gauss=bb_gauss,
                               aa_mean=aa_mean,
                               bb_mean=bb_mean,
                               n_cluster=n_cluster, 
                               max_iter=max_iter)
            lab_min_tmp=np.array( lab_dend_tmp.threshold).sum()  
            if lab_min_tmp > lab_min:
                lab_min=lab_min_tmp
                lab_dend = lab_dend_tmp
                 
        self.index=lab_dend.index
        threshold = lab_dend.threshold

        cluster[threshold]=self.index[0]
        cluster[~threshold]=self.index[1]

        self.cluster=cluster 
        self.threshold= threshold
        # for i in range(nn):
        #     clus_mean[(y_pred==i) ]=1-i 
        if self.disp:
            print(f'Number of vertices on the main branch : {np.array(~threshold).sum()}')
            print(f'Number of vertices on the spines cluster : {np.array(threshold).sum()}')


    def Label_branch(self,vertices_index=None,
                     unique_vertices=None,
                     branch_threshold=1):

        if vertices_index is not None:
            ind_cluster=np.where(~self.threshold.reshape(-1))[0][vertices_index]
            self.cluster_label=np.zeros_like(self.dend.vertices[:,0] )
        else:
            ind_cluster=np.where(~self.threshold.reshape(-1))[0]
            self.cluster_label=self.cluster.copy() 

            
        self.dend.Cluster_class(ind_cluster )
        self.cluster_branch=self.dend.cluster_class_sorted
        if self.disp:
            print(f'Number of clusters on the main branch: {len(self.cluster_branch)}')
 

        if unique_vertices is None:
            for cl in self.cluster_branch[branch_threshold:]:
                self.cluster_label[cl]=self.index[0]
            cll =[]
            for cl in self.cluster_branch[:branch_threshold+1]:
                cll.extend(cl)
            self.cluster_branch_main=[cll]
        else:
            for cl in self.cluster_branch: 
                if   (set(unique_vertices).intersection(set(cl))) :
                    self.cluster_label[cl]=50   
                    self.cluster_branch_main=[cl] 
                    # unique_vertices = None
                    # print('yessssss!')
                else:
                    self.cluster_label[cl]=10
        
 
    def Label_spines(self):
        self.dend.Cluster_class(ind_cluster=np.where(self.cluster_label==self.index[0])[0] )
        self.cluster_spines=self.dend.cluster_class_sorted
        if self.disp:
            print(f'Number of spine clusters labelled: {len(self.cluster_spines)}') 
 

def KMeans_intensity(intensity, n_cluster=20, max_iter=300): 
    if intensity.ndim == 1:
        intensity = intensity.reshape(-1, 1)

    if np.isnan(intensity).any():
        # print("Data contains NaNs. Imputing missing values...")
        imputer = SimpleImputer(strategy='mean')
        intensity = imputer.fit_transform(intensity) 
    cluster1 = KMeans(n_clusters=n_cluster, max_iter=max_iter, n_init='auto')
    cluster1.fit(intensity)
    y_pred = cluster1.labels_ 
    pca_rad = np.array([intensity[y_pred == i].mean() for i in range(n_cluster)]) 
    cluu_rad_ = pca_rad[y_pred]

    return cluu_rad_.reshape(-1, 1)





def Mapping_inverse(lst):
    return  {val: idx for idx, val in enumerate(lst)}
         


def list_vectices_unique(faces): 
    edges = np.sort(np.vstack([
        faces[:, [0, 1]],   
        faces[:, [1, 2]], 
        faces[:, [2, 0]]  
    ]), axis=1)
     
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True) 
    
    return np.unique( unique_edges[counts == 1] .flatten())  

 

class cluster_class:
    def __init__(self,faces_neighbor_index ) -> None: 
        self.faces_neighbor_index=faces_neighbor_index
  
    def Cluster_index(self, ln_elm):  
        ln_elm = np.asarray(ln_elm)      
        if  (ln_elm.size == 0): 
            self.cluster_index = np.array([])  
            self.cluster_faces_index   = np.array([[0,0,0]])
            return
        
        # ln_rm_set = set(ln_elm) 
        faces_neighbors_list = [] 
        for ii in ln_elm:
            faces_neighbors_list.append(self.faces_neighbor_index[ii]) 
        if len(faces_neighbors_list)>0:
            faces_neighbors = cluster_faces_index =np.vstack(faces_neighbors_list) 
            cluster_faces_index = faces_neighbors[np.any(np.isin(faces_neighbors, list(set(ln_elm))), axis=1)] 
            n_upd=set(np.array(cluster_faces_index).flatten())  

            self.cluster_index =np.array(list(n_upd))  
            self.cluster_faces_index=cluster_faces_index 
        else:
            self.cluster_index = np.array([])  
            self.cluster_faces_index   = np.array([[0,0,0]])

            
 

    def Cluster_deleted_index(self, ln_elm, ln_rm=np.array([])): 
        ln_elm = np.asarray(ln_elm)
        ln_rm = np.asarray(ln_rm)
        if (ln_rm.size == 0) or (ln_elm.size == 0): 
            self.cluster_index = np.array([])  
            self.cluster_faces_index   = np.array([[0,0,0]])
            return
        

        ln_rm_set = set(ln_rm)  # Convert ln_rm to a set for faster membership checks
        cluster_faces_index = []

        for ii in ln_elm:
            faces = np.array(self.faces_neighbor_index[ii])  # Get the neighbor indices for the element
            mask = np.isin(faces, ln_rm_set)  # Check if any of the elements are in ln_rm
            valid_faces = faces[~np.any(mask, axis=1)]  # Keep only faces where no element is in ln_rm
            cluster_faces_index.extend(valid_faces.tolist())  # Add the valid faces to the list

        n_upd=set(np.array(cluster_faces_index).flatten())  

        self.cluster_index =np.array(list(n_upd))  
        self.cluster_faces_index=cluster_faces_index 

    # def Cluster_deleted_faces_index(self, ln_elm, ln_rm=np.array([])):
    #     ln_rm_set = set(ln_rm)  # Convert ln_rm to a set for faster membership checks
    #     cluster_faces_index = []
    #     ln_rm_set=[]
    #     for ii in ln_rm:
    #         ln_rm_set.append(self.faces_neighbor_index[ii].flatten())
    #     facess=
    #     ln_rm_set = set(ln_rm_set)
    #     print('sett',  bool(ln_rm_set.intersection(set(ln_elm))))
    #     ln_elmm=list(set(ln_elm)-ln_rm_set )

    #     for ii in ln_elmm:
    #         faces = np.array(self.faces_neighbor_index[ii])  # Get the neighbor indices for the element
    #         mask = np.isin(faces, ln_rm_set)  # Check if any of the elements are in ln_rm
    #         valid_faces = faces[~np.any(mask, axis=1)]  # Keep only faces where no element is in ln_rm
    #         cluster_faces_index.extend(valid_faces.tolist())  # Add the valid faces to the list

    #     n_upd=set(np.array(cluster_faces_index).flatten())  
    #     print(len(ln_elmm),ln_elm.size,len(ln_rm_set),ln_rm.size,len(n_upd))
    #     self.cluster_index =np.array(list(n_upd))  
    #     self.cluster_faces_index=cluster_faces_index 



    def Cluster_faces_delete(self, node ):
        self.cluster_faces_index= node.faces[~np.any(np.isin(node.faces,set(node.vertices_index_unique )), axis=1)] 
        self.cluster_index =np.array(list(set(np.array(self.cluster_faces_index).flatten()) ))   

    # def Cluster_faces(self, node, vertices):
    #     if  vertices.size > 0:  
    #         self.cluster_faces = node.faces[  vertices]

    # def Cluster_deleted_index(self, ln_elm, ln_rm=np.array([])):
    #     ln_rm_set = set(ln_rm)  # Convert ln_rm to a set for faster membership checks
    #     ln_rmm=  list(set(np.concatenate([ self.faces_neighbor_index[ii].flatten() for ii in ln_rm]))) 
    #     cluster_faces_index =[]
    #     for ii in ln_elm:
    #         if ii not in ln_rmm:
    #             faces = np.array(self.faces_neighbor_index[ii])  # Get the neighbor indices for the element
    #             mask = np.isin(faces, ln_rm_set)  # Check if any of the elements are in ln_rm
    #             valid_faces = faces[~np.any(mask, axis=1)]  # Keep only faces where no element is in ln_rm
    #             cluster_faces_index.extend(valid_faces.tolist())  # Add the valid faces to the list

    #     n_upd=set(np.array(cluster_faces_index).flatten())  

    #     self.cluster_index =np.array(list(n_upd))  
    #     self.cluster_faces_index=cluster_faces_index 


    def Cluster_deleted_index(self, ln_elm, ln_rm=np.array([])): 
        ln_elm = np.asarray(ln_elm)
        ln_rm = np.asarray(ln_rm)
        if (ln_rm.size == 0) or (ln_elm.size == 0):
            self.cluster_index = np.array([])  
            self.cluster_faces_index   = np.array([[0,0,0]])
            return
            
        ln_rm_neighbors_set = set( np.concatenate([self.faces_neighbor_index[ii].flatten() for ii in ln_rm]))
 
        remaining_elements = np.array(ln_elm)[~np.isin(ln_elm, list(ln_rm_neighbors_set))]

        if len(remaining_elements) == 0: 
            self.cluster_index = np.array([])  
            self.cluster_faces_index   = np.array([[0,0,0]])
            return
            
        face_list = [self.faces_neighbor_index[ii] for ii in remaining_elements]
        
        if len(face_list) == 0: 
            self.cluster_index = np.array([])  
            self.cluster_faces_index   = np.array([[0,0,0]])
            return

        faces = np.concatenate(face_list)
        
        # Apply mask to filter out invalid faces
        valid_faces_mask = ~np.isin(faces, list(set(ln_rm))).any(axis=1)
        cluster_faces_index = faces[valid_faces_mask]
        
        self.cluster_index = np.unique(cluster_faces_index.flatten()) 
        self.cluster_faces_index = cluster_faces_index.tolist()




    def Cluster_faces(self):
        if self.cluster_index.size>0:
            cluster_faces_index = np.unique(self.cluster_faces_index, axis=0)
            cluster_index = self.cluster_index 
            map_inv = Mapping_inverse(cluster_index)
            
            cluster_faces_indices = np.array([
                [map_inv[face] for face in face_row]
                for face_row in cluster_faces_index
            ])
            
            self.cluster_faces= cluster_faces_indices
             
            self.cluster_faces_index  = cluster_index[cluster_faces_indices]
        else:
            self.cluster_faces_index  = np.array([[0,0,0]])


    def Cluster_faces_unique(self):
        self.cluster_faces_unique= list_vectices_unique(self.cluster_faces_index )  

 



    def get_intersection(self,i,ln_elm=None):
        if ln_elm is None:
            ln_elm=self.clu

        clu_ftmp = set(ln_elm[i])
        ii = i+1
        check_ii = len(ln_elm) - ii
        while check_ii > 1:
            ii += 1
            if clu_ftmp.intersection(ln_elm[ii]): 
                clu_ftmp.update(ln_elm[ii])
                ln_elm.pop(ii)  
                check_ii = len(ln_elm) - ii 
            else:
                check_ii -= 1 
        self.clu=ln_elm
        self.clu[i]=clu_ftmp



def Cluster_radius_proj(cluu_rad,geo_cluster,node,index_tree,radius_number=20,kmean_max_iter=300):
    geo_cluster.Cluster_proj(node.children[index_tree].vertices_index)

    pca_radius=geo_cluster.cluster_proj_radius.reshape(-1,1)
    if pca_radius.shape[0]>20:
        cluster1 = KMeans(n_clusters=radius_number, max_iter=kmean_max_iter, n_init='auto')
        cluster1.fit(pca_radius)
        y_pred = cluster1.labels_
        pca_rad=[float(np.mean(pca_radius[y_pred==i])) for i in range(radius_number)] 
        y_pred_1=np.argsort(pca_rad)
        
        cluu_rad_ = np.zeros_like(y_pred) 
        for ii,iip in enumerate(y_pred_1):
            cluu_rad_[(y_pred==iip)  ]=ii

        cluu_rad[node.children[index_tree].vertices_index]=cluu_rad_.reshape(-1,1)  
        return cluu_rad
    else:
        return cluu_rad






import copy

class dendrite:
    def __init__(self, 
                 vertices_index=None, 
                 vertices=None,
                 faces=None, 
                 intensity=None, 
                 children_number=None,
                 vertices_index_unique=None,
                 head_radius=None,
                 neck_radius=None,
                 centroid=None,
                 length=None,
                 skl_index=None,
                 ):
        self.vertices_index = vertices_index
        self.vertices=vertices
        self.faces = faces
        self.intensity = intensity
        self.children_number = children_number
        self.vertices_index_unique=vertices_index_unique
        self.neck_radius=neck_radius
        self.head_radius=head_radius
        self.centroid=centroid
        self.length=length
        self.skl_index=skl_index
        self.children = []

    def add_child(self, 
                  vertices_index=None, 
                  faces=None, 
                  intensity=None, 
                  children_number=None,
                  vertices_index_unique=None,
                 head_radius=None,
                 neck_radius=None,
                 centroid=None,
                 length=None,
                 skl_index=None,
                  ):
        child_node = dendrite(vertices_index=vertices_index, 
                                   faces=faces,
                                   intensity=intensity,
                                   children_number=children_number,
                                   vertices_index_unique=vertices_index_unique,
                                    neck_radius=neck_radius,
                                    head_radius=head_radius,
                                    centroid=centroid,
                                    length=length,
                                    skl_index=skl_index,
        )
        self.children.append(child_node) 


    def delete_child_by_index(self, index=None): 
        if index is not None and 0 <= index < len(self.children):
            self.children.pop(index)
            self.children_number=self.children_number-1
        else:
            print("Invalid index.")
 
    def delete_child(self, child_node=None): 
        if child_node in self.children:
            self.children.remove(child_node)
            self.children_number=self.children_number-1
        else:
            print("Child node not found.")


    def copy(self): 
        new_copy = dendrite(
            vertices_index=self.vertices_index.copy() if self.vertices_index is not None else None,
            vertices=self.vertices.copy() if self.vertices is not None else None,
            faces=self.faces.copy() if self.faces is not None else None,
            intensity=self.intensity.copy() if self.intensity is not None else None,
            vertices_index_unique= self.vertices_index_unique.copy() if self.vertices_index_unique is not None else None,
            neck_radius=self.neck_radius.copy() if self.neck_radius is not None else None,
            head_radius=self.head_radius.copy() if self.head_radius is not None else None,
            centroid=self.centroid.copy() if self.centroid is not None else None,
            length=self.length.copy() if self.length is not None else None,
            skl_index=self.skl_index.copy() if self.skl_index is not None else None,
            children_number=copy.deepcopy(self.children_number) if self.children_number is not None else None
        )
 
        new_copy.children = [child.copy() for child in self.children]

        return new_copy
 



 

def Branch_class_index(vertices_index,
                       cclu:cluster_class,
                       dend:curv_mesh,
                       node:dendrite=None,
                       gauss_threshold=40,
                       size_threshold=500,
                       start_index=0,
                       faces=False): 
    if node is None:
        node=dendrite() 
        ret = True
    if faces:
        cclu.Cluster_index(ln_elm=vertices_index) 
        cclu.Cluster_faces()
        cclu.Cluster_faces_unique()  
        node.vertices_index=cclu.cluster_index
        node.faces=cclu.cluster_faces
        node.intensity=np.zeros(dend.n_vert)
        node.vertices_index_unique =cclu.cluster_faces_unique
        
    dend.Cluster_class(vertices_index) 
    for ii,cl in enumerate(dend.cluster_class_sorted[start_index:]): 
        cl=np.array(cl)
         
        if (cl.size>size_threshold) and (np.max(np.abs(dend.gauss_curv[vertices_index] ) )>gauss_threshold):  
            cclu.Cluster_index(ln_elm=cl) 
            cclu.Cluster_faces()
            cclu.Cluster_faces_unique()  
            node.add_child(vertices_index=cclu.cluster_index,
                            faces=cclu.cluster_faces, 
                            vertices_index_unique =cclu.cluster_faces_unique 
                            )  
            node.intensity[vertices_index]=ii
            ii+=1
    node.children_number= len(dend.cluster_class_sorted) 
    # print(f'Number of cluster: {node.children_number}') 
    if ret:
        return node
 
 
def Cluster_contour(vertices,vertices_unique_centroid,n_clusters=30, kmean_max_iter=300):
    pca_radius=np.linalg.norm(vertices-vertices_unique_centroid,axis=1).reshape(-1,1)
    cluster1 = KMeans(n_clusters=n_clusters, max_iter=kmean_max_iter, n_init='auto')
    cluster1.fit(pca_radius)
    y_pred = cluster1.labels_
    pca_rad=[float(np.mean(pca_radius[y_pred==i])) for i in range(n_clusters)]
    y_pred_1=np.argsort(pca_rad)
    
    cluu_rad_ = np.zeros_like(y_pred)  
    for ii,iip in enumerate(y_pred_1):
        cluu_rad_[(y_pred==iip)  ]=ii
    return cluu_rad_


def Cluster_contour_class(node:dendrite,dend:curv_mesh=None,radius_level_max=20):  
    vertices_unique_centroid=np.mean(dend.vertices[node.vertices_index_unique],axis=0) 
    intensity=np.zeros_like(dend.vertices[:,0]) 
    intensity[node.vertices_index.flatten()]=Cluster_contour(dend.vertices[node.vertices_index] ,vertices_unique_centroid,radius_level_max)
    return intensity
 

def Cluster_contour_class_txt(vertices_0,dend_clu_index,center,radius_level=0,radius_level_max=20): 
    cluu_rad=np.zeros_like(vertices_0[:,0])
    cluu_rad[dend_clu_index.flatten()]=Cluster_contour(vertices_0[dend_clu_index]  ,center,radius_level_max) 
    if radius_level >0:
        cluu_rad[cluu_rad<=radius_level]=0
        cluu_rad[cluu_rad>radius_level]=1
    return cluu_rad 



def Connected_vertices_index( vertices_0, cclu: label_cluster,vertices_index): 
        
        cclu.Cluster_index(ln_elm=vertices_index)
        cclu.Cluster_faces()
        cclu.Cluster_faces_unique()

        # Create a mesh from clustered indices
        dend_cclu = curv_mesh(vertices=vertices_0[cclu.cluster_index],
                              faces=cclu.cluster_faces)
        dend_cclu.Vertices_neighbor()
  
        map_inv = Mapping_inverse(cclu.cluster_index)
        dend_cclu_unique = [map_inv[face] for face in cclu.cluster_index]
 
        hhh = [list(set(np.concatenate(dend_cclu.vertex_neighbor[ii]))) for ii in dend_cclu_unique]

        indd = []
        for iii in hhh:
            for ii in iii:
                indd.append(ii)

        dend_cclu.Cluster_class(ind_cluster=np.array(indd))
 
        return [cclu.cluster_index[ii] for ii in  dend_cclu.cluster_class_sorted[0]]

   

def Branch_division(
                cclu:cluster_class,
                dend:curv_mesh, 
                vertices_index,
                vertices_index_remove=None,
                nodee:dendrite=None, 
                size_threshold=10,
                stop_index=None,
                get_faces=True,
                vertex_neighbor=None,
                tf_cluster_faces_unique=True,):  
    nodee=dendrite() 
    if vertex_neighbor is not None:
        cclu=cluster_class(faces_neighbor_index=vertex_neighbor)
    elif dend.vertex_neighbor:
        cclu=cluster_class(faces_neighbor_index=dend.vertex_neighbor)
        
    dend.Cluster_class(ind_cluster=vertices_index)  
    if stop_index is None:
        ne_list=dend.cluster_class_sorted
    else:
        ne_list=dend.cluster_class_sorted[:stop_index]
    if tf_cluster_faces_unique:
        for ccc in  ne_list:
            if len(ccc)>size_threshold:
                # class_vertices_index.append(ccc)
                if vertices_index_remove is None:
                    cclu.Cluster_index(ln_elm=np.array(ccc))
                else:
                    cclu.Cluster_deleted_index(ln_elm=np.array(ccc), ln_rm=vertices_index_remove) 
                if len(cclu.cluster_index)>size_threshold:
                    if get_faces:
                        cclu.Cluster_faces()
                        cclu.Cluster_faces_unique() 
                        nodee.add_child(vertices_index=cclu.cluster_index,
                                        faces=cclu.cluster_faces, 
                                        vertices_index_unique =cclu.cluster_faces_unique ,
                                        )
                    else: 
                        nodee.add_child(vertices_index=cclu.cluster_index, 
                                        ) 
    else:                

        for ccc in  ne_list:
            if len(ccc)>size_threshold: 
                if vertices_index_remove is None:
                    cclu.Cluster_index(ln_elm=np.array(ccc))
                else:
                    cclu.Cluster_deleted_index(ln_elm=np.array(ccc), ln_rm=vertices_index_remove) 
                if len(cclu.cluster_index)>size_threshold:
                    if get_faces:
                        cclu.Cluster_faces() 
                        nodee.add_child(vertices_index=cclu.cluster_index,
                                        faces=cclu.cluster_faces, 
                                        # vertices_index_unique =cclu.cluster_faces_unique ,
                                        )
                    else: 
                        nodee.add_child(vertices_index=cclu.cluster_index, 
                                        ) 

 
    return nodee




def find_leaf_lineage_children(root):
    leaf_nodes = [] 
    def traverse_and_collect(node):
        if node.children_number is None:
            node.children_number = 0  
        
        if node.children_number == 0:
            leaf_nodes.append(node)  

        for child in node.children:
            traverse_and_collect(child)
 
    traverse_and_collect(root)
 
    leaf_lineage = {} 
    def get_lineage_children_count(node):
        lineage = []
        current = node
        while current:
            lineage.append(current.children_number)
            if not current.parent: 
                break
            current = current.parent
        return lineage[::-1] 
    
    def set_parent_pointers(node, parent=None):
        node.parent = parent
        for child in node.children:
            set_parent_pointers(child, node)
 
    set_parent_pointers(root) 
    
    for leaf in leaf_nodes:
        lineage = get_lineage_children_count(leaf)
        leaf_lineage[leaf] = lineage

    return leaf_lineage


def Spines_lineage(root):
    leaf_lineage = find_leaf_lineage_children(root)
 
    def find_descendants(node):
        descendants = []
        current = node
        while current:
            if current.children_number > 1:
                break
            descendants.append(current)
            current = current.parent
        return descendants
 
    leaf_descendants = {} 
    for leaf, lineage in leaf_lineage.items():
        descendants = find_descendants(leaf)
        leaf_descendants[leaf] = {
            "lineage": lineage,
            "descendants_number": [descendant.children_number for descendant in descendants],
            "descendants": [descendant  for descendant in descendants]
        } 
    return leaf_descendants
 
 
def Spines(root): 
    leaf_descendants = Spines_lineage(root)
    descendants_one=[] 
    for leaf, data in leaf_descendants.items():  
        descendants_one.append(data['descendants'][-1])
    return descendants_one

def print_children_number(node, level=0): 
    print(" " * (level * 2) + f"dend at level {level} has {node.children_number} children.")
    for child in node.children:
        print_children_number(child, level + 1)


 
 
def get_unique_class(index_unique,vertex_neighbor,threshold=3):
    index_unique_tmp=index_unique.copy()
    index_save=[] 
    chk=1
    index_i=[index_unique_tmp[0]]
    gg=1
    save=[]
    ssave=[]
    while gg>0:
        index_save=[] 
        chk=1
        index_i=[index_unique_tmp[0]]
        while (chk>0) and len(index_unique_tmp)>0: 
            index_save.extend(index_i)
            index_remain=list(set(index_unique_tmp)-set(index_i))
            vertex_nei=[]
            for uu in [ vertex_neighbor[ii]  for ii in index_i]:
                vertex_nei.extend(uu.flatten())  
            index_i=list(set(np.array(vertex_nei) ).intersection(set(index_remain))) 
            index_unique_tmp=list(set(index_remain)-set(index_i))
            chk=len(vertex_nei)
        save.append(index_save)
        ssave.extend(index_save) 
        # print(ty),
        index_unique_tmp=list(set(index_unique)-set(ssave))
        gg=len(index_unique_tmp)
    save_m,save_1=[],[]
    for uu in save:
        if len(uu)==1:
            save_1.append(uu)
        else:
            save_m.append(uu) 
    savee=[]
    for nn in save_m:
        for uuu in save_1: 
            vertex_nei=[]
            for uu in [ vertex_neighbor[ii]  for ii in uuu]:
                vertex_nei.extend(uu.flatten())  
            index_i=list(set(np.array(vertex_nei) ).intersection(set(nn))) 
            if len(index_i)>0:
                nn.append(uuu[0])
        if len(nn)>threshold:        
            savee.append(nn) 

    return savee


def Mapping_inverse(lst):
    return  {val: idx for idx, val in enumerate(lst)}
         
def Connected_vertices_index( vertices_0, cclu: label_cluster,vertices_index):  
        if len(vertices_index)<1:
             return vertices_index 
        cclu.Cluster_index(ln_elm=vertices_index)
        cclu.Cluster_faces()
        cclu.Cluster_faces_unique() 
        dend_cclu = curv_mesh(vertices=vertices_0[cclu.cluster_index],
                              faces=cclu.cluster_faces)
        dend_cclu.Vertices_neighbor()
  
        map_inv = Mapping_inverse(cclu.cluster_index)
        dend_cclu_unique = [map_inv[face] for face in cclu.cluster_index] 
        hhh = [list(set(np.concatenate(dend_cclu.vertex_neighbor[ii]))) for ii in dend_cclu_unique]

        indd = []
        for iii in hhh:
            for ii in iii:
                indd.append(ii)

        dend_cclu.Cluster_class(ind_cluster=np.array(indd)) 
        return [cclu.cluster_index[ii] for ii in  dend_cclu.cluster_class_sorted[0]]

  

 

def Mapping_inverse(lst):
    return  {val: idx for idx, val in enumerate(lst)}
         
def Connected_vertices_index( vertices_0, cclu: label_cluster,vertices_index): 
        
        if len(vertices_index)<1:
             return vertices_index 
        cclu.Cluster_index(ln_elm=vertices_index)
        cclu.Cluster_faces()
        cclu.Cluster_faces_unique() 
        dend_cclu = curv_mesh(vertices=vertices_0[cclu.cluster_index],
                              faces=cclu.cluster_faces)
        dend_cclu.Vertices_neighbor() 
        map_inv = Mapping_inverse(cclu.cluster_index)
        dend_cclu_unique = [map_inv[face] for face in cclu.cluster_index] 
        hhh = [list(set(np.concatenate(dend_cclu.vertex_neighbor[ii]))) for ii in dend_cclu_unique]

        indd = []
        for iii in hhh:
            for ii in iii:
                indd.append(ii)

        dend_cclu.Cluster_class(ind_cluster=np.array(indd))
 
        return [cclu.cluster_index[ii] for ii in  dend_cclu.cluster_class_sorted[0]]

 
def Refine_vertices_index( vertices_0, 
                          cclu: label_cluster,
                          vertices_index,
                          vertices_centroid=None,
                          refine_threshold=1.5,
                          kdtree:KDTree=None,
                          neighbor_filter=True,
                          k_neighors=50,
                          ):
    if kdtree is None:
         kdtree = KDTree(vertices_0, leaf_size=2)
    if len(vertices_index)<1:
            return vertices_index  
    
    
    if neighbor_filter: 
        neighbors = kdtree.query(vertices_0[vertices_centroid], k=k_neighors, return_distance=False).flatten()
    else:
        indices = kdtree.query_radius(((vertices_centroid[0]+vertices_centroid[1])/2).reshape(1, -1),
                                r=np.linalg.norm(vertices_centroid[0]-vertices_centroid[1])/refine_threshold)
        neighbors=indices[0]
    
    if len(neighbors)<1:
         spine_ind = vertices_index
    else:
        spine_ind=np.unique(np.concatenate((vertices_index, neighbors)))
     
    spine_ind = Connected_vertices_index( vertices_0,cclu ,spine_ind)
    spine_ind=np.array(spine_ind)  
    return spine_ind

 


def project_points_onto_line(points, vector, z):  
    vector=vector = vector / np.linalg.norm(vector)  
    return z  +  np.dot(points - z, vector)[:, np.newaxis] *vector 




import os

def get_intensity(vertices, 
                    faces, 
                    file_path_gauss_full,
                    file_path_mean_full, 
                    thr_gauss=45,
                    thr_mean=15,
                ):  
        dend_00 = curv_mesh(vertices=vertices, faces=faces) 
        dend_00.Gauss_curv()
        dend_00.Mean_curv()

        dend_00.gauss_curv[dend_00.gauss_curv < -thr_gauss] = -thr_gauss
        dend_00.gauss_curv[dend_00.gauss_curv > thr_gauss] = thr_gauss

        dend_00.mean_curv[dend_00.mean_curv < -thr_mean] = -thr_mean
        dend_00.mean_curv[dend_00.mean_curv > thr_mean] = thr_mean
        
        np.savetxt(file_path_mean_full, dend_00.mean_curv, fmt='%f')
        np.savetxt(file_path_gauss_full, dend_00.gauss_curv, fmt='%f') 
 

import os
import numpy as np

class dendrite_io:
    def __init__(self, txt_save_file, 
                 name=None, 
                 part=None,
                 itera_start=0, 
                 itera_end=None,
                 shaft_index=None,
                 vertices_index=None,
                 faces=None,
                 skl_vertices=None,
                 skl_index=None,
                 size_threshold=10,
                 ): 
        self.size_threshold=size_threshold
        self.vertices_index=vertices_index
        self.faces=faces
        self.skl_vertices=skl_vertices
        self.skl_index=skl_index
        self.name_count = name.name_count if name is not None else 'count'
        self.name_index = name.name_index if name is not None else 'index'
        self.name_faces = name.name_faces if name is not None else 'faces'
        self.name_index_unique = name.name_index_unique if name is not None else 'index_unique'
        self.name_intensity = name.name_intensity if name is not None else 'intensity'
        self.txt_spine_count = name.txt_spine_count if name is not None else 'spine_count'
        self.name_spine = name.name_spine if name is not None else 'spine'
        self.name_count = name.name_count if name is not None else 'count'
        self.name_center = name.name_center if name is not None else 'cent'
        self.txt_save_file = txt_save_file
        self.part = name.name_spine if part is not None else self.name_spine
        self.itera_start = itera_start 
        self.shaft_index=shaft_index


    def node_to_txt(self, node, intensity_len, part=None, txt_save_file=None, itera_start=None, itera_end=None,pid=None,size_threshold=None,):
        txt_save_file = txt_save_file or self.txt_save_file
        part = part or self.part
        itera_start = itera_start or self.itera_start
        size_threshold = size_threshold or self.size_threshold
        
        ctt = []
        intensity_org = -20 * np.ones(intensity_len)
        tmpp=[]
        if node.children:
            itera_end = itera_end or len(node.children)
            for ip in range(itera_start, itera_end): 
                vertices_index_i = node.children[ip].vertices_index
                faces_i = node.children[ip].faces
                vertices_index_unique_i = node.children[ip].vertices_index_unique
                intensity_org[vertices_index_i] = ip
                tmpp.extend(vertices_index_i) 
                np.savetxt(os.path.join(txt_save_file, f'{part}_{self.name_index}_{ip}.txt'), vertices_index_i, fmt='%d') 
                np.savetxt(os.path.join(txt_save_file, f'{part}_{self.name_faces}_{ip}.txt'), faces_i, fmt='%d') 
                np.savetxt(os.path.join(txt_save_file, f'{part}_{self.name_index_unique}_{ip}.txt'), vertices_index_unique_i, fmt='%d') 
                if (self.skl_index is not None) and (self.skl_vertices is not None): 
                    skl_vertices_i=self.skl_vertices[list(set(self.skl_index[vertices_index_i] ))]
                    np.savetxt(os.path.join( txt_save_file,f'{part}_{self.name_center}_curv_{ip}.txt'),skl_vertices_i, fmt='%f')

                ctt.append(ip)  
        shaft_index= self.shaft_index if self.shaft_index is not None else list(set([ii for ii in range(intensity_len)])-set(tmpp)) 

        if (len(ctt)==0) and (self.vertices_index is not None):
                ip=0
                ctt=[0]
                np.savetxt(os.path.join(txt_save_file, f'{part}_{self.name_index}_{ip}.txt'), self.vertices_index, fmt='%d') 
                np.savetxt(os.path.join(txt_save_file, f'{part}_{self.name_faces}_{ip}.txt'), self.faces, fmt='%d') 
                np.savetxt(os.path.join(txt_save_file, f'{part}_{self.name_index_unique}_{ip}.txt'), [], fmt='%d') 
                if self.skl_vertices is not None:
                    np.savetxt(os.path.join( txt_save_file,f'{part}_{self.name_center}_curv_{ip}.txt'),self.skl_vertices, fmt='%f')

        if (self.shaft_index is None) and (pid is not None): 
            nodee = Branch_division(
                cclu=pid.cclu,
                dend=pid.dend, 
                vertices_index=shaft_index, 
                stop_index=1,
                size_threshold=size_threshold,
            ) 
            np.savetxt(os.path.join(txt_save_file, self.txt_shaft_index), nodee.children[0].vertices_index, fmt='%d') 
            np.savetxt(os.path.join(txt_save_file, self.txt_shaft_faces), nodee.children[0].faces, fmt='%d') 
            np.savetxt(os.path.join(txt_save_file, self.txt_shaft_index_unique), nodee.children[0].vertices_index_unique, fmt='%d')



        intensity_org[shaft_index] = -20
        np.savetxt(os.path.join(txt_save_file, f'{part}_{self.name_intensity}.txt'), intensity_org, fmt='%d')  
        np.savetxt(os.path.join(txt_save_file, f'{part}_{self.name_count}.txt'), np.array(ctt if ctt else [0]), fmt='%d')
    


    def txt_to_node(self, txt_save_file=None, part=None,count=None,size_threshold=None):
        txt_save_file = txt_save_file or self.txt_save_file
        part = part or self.name_spine
        node_tmp = dendrite()
        if count is None:
            intensity_file = os.path.join(txt_save_file, f'{part}_{self.name_count}.txt') 
            if os.path.exists(intensity_file):
                count = np.loadtxt(intensity_file, dtype=int) 
            else:
                self.count = [0]
                return node_tmp
        for ipp in count:
            vertices_index        = np.loadtxt(os.path.join(txt_save_file, f'{part}_{self.name_index}_{ipp}.txt'), dtype=int)
            spine_faces           = np.loadtxt(os.path.join(txt_save_file, f'{part}_{self.name_faces}_{ipp}.txt'), dtype=int)
            vertices_index_unique = np.loadtxt(os.path.join(txt_save_file, f'{part}_{self.name_index_unique}_{ipp}.txt'), dtype=int) 
            
            node_tmp.add_child(vertices_index=vertices_index, 
                               faces=spine_faces, 
                               vertices_index_unique=vertices_index_unique, 
                               intensity=[ipp])
    
        self.count = count
        return  node_tmp




def Threshold_curv(curv, thre  ):  
        curv[ curv>thre]=thre
        curv[curv<=-thre]=-thre 
        return curv


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("File not found.")

def get_color( ):
    return   [
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white',
        'gray', 'cyan', 'magenta', 'lime', 'maroon', 'navy', 'olive', 'teal', 'violet', 'indigo',
        'gold', 'silver', 'turquoise', 'beige', 'coral', 'crimson', 'khaki', 'lavender', 'salmon',
        'sienna', 'tan', 'wheat', 'plum', 'orchid', 'slategray', 'dodgerblue', 'firebrick', 'chartreuse'
    ]
   


def find_point_on_vector(a, b, s):
    a, b = np.array(a), np.array(b)   
    direction = b - a
    length = np.linalg.norm(direction) 
    if length == 0:
        raise ValueError("Points A and B must be distinct.")
 
    return b + s * ( direction / length )  





def Mapping_inverse(lst):
    return  {val: idx for idx, val in enumerate(lst)}
         
def Connected_vertices_index( vertices_0, cclu: label_cluster,vertices_index): 
        
        if len(vertices_index)<1:
             return vertices_index 
        cclu.Cluster_index(ln_elm=vertices_index)
        cclu.Cluster_faces()
        cclu.Cluster_faces_unique() 
        dend_cclu = curv_mesh(vertices=vertices_0[cclu.cluster_index],
                              faces=cclu.cluster_faces)
        dend_cclu.Vertices_neighbor()
  
        map_inv = Mapping_inverse(cclu.cluster_index)
        dend_cclu_unique = [map_inv[face] for face in cclu.cluster_index] 
        hhh = [list(set(np.concatenate(dend_cclu.vertex_neighbor[ii]))) for ii in dend_cclu_unique]

        indd = []
        for iii in hhh:
            for ii in iii:
                indd.append(ii)

        dend_cclu.Cluster_class(ind_cluster=np.array(indd))
 
        return [cclu.cluster_index[ii] for ii in  dend_cclu.cluster_class_sorted[0]]

 
def Refine_vertices_index( vertices_0, 
                          cclu: label_cluster,
                          vertices_index,
                          vertices_centroid=None,
                          refine_threshold=1.5,
                          kdtree:KDTree=None,
                          neighbor_filter=True,
                          k_neighors=50):
    if kdtree is None:
         kdtree = KDTree(vertices_0, leaf_size=2)
    if len(vertices_index)<1:
            return vertices_index   
    
    if neighbor_filter: 
        neighbors = kdtree.query(vertices_0[vertices_centroid], k=k_neighors, return_distance=False).flatten()
    else:
        indices = kdtree.query_radius(((vertices_centroid[0]+vertices_centroid[1])/2).reshape(1, -1),
                                r=np.linalg.norm(vertices_centroid[0]-vertices_centroid[1])/refine_threshold)
        neighbors=indices[0]
    
    if len(neighbors)<1:
         spine_ind = vertices_index
    else:
        spine_ind=np.unique(np.concatenate((vertices_index, neighbors)))
     
    spine_ind = Connected_vertices_index( vertices_0,cclu ,spine_ind)
    spine_ind=np.array(spine_ind)  
    return spine_ind


 




class clust_pca:
    def __init__(self,vertices,vertices_index_shaft,
                 vertices_center=None,
                  vcv_length=None, ):
        pass
        self.centroid= centroid= np.mean(vertices, axis=0) 
        pca  = PCA(n_components=3)
        pca.fit(vertices)
         
        self.normal_vector = pca.components_[0] 
        proj_0 = np.dot(vertices - centroid, self.normal_vector) 

        self.pc_start = centroid +  proj_0.min()* self.normal_vector
        self.pc_end= centroid +  proj_0.max()* self.normal_vector 
        self.pc_bound=np.array([self.pc_start,self.pc_end])

        vertices_shaft_wrap=vertices[vertices_index_shaft] 
        if vertices_center is None:
            self.vertices_shaft_wrap= np.vstack((vertices_shaft_wrap, self.pc_bound))
        else:
            self.vertices_center=vertices_center
            self.vertices_shaft_wrap= np.vstack((vertices_shaft_wrap, self.pc_bound,self.vertices_center))
            self.vcv_length=vcv_length



def dot_(V1,V2,axis=1):
    return np.sum(V1*V2, axis=axis, keepdims=True) 

def volume(vertices=None,faces=None): 

    centroid =mmean(vertices, axis=0)
    return np.abs(np.sum(dot_(vertices[faces[:,0]] - centroid ,
                np.cross(vertices[faces[:,1]] - centroid,
                            vertices[faces[:,2]] - centroid ))) /6.0)



def get_change_name(data_name,split_char,ext):
    ppp=data_name.split(split_char) 
    path_init=f'{ppp[0]}'
    ppp=ppp[1:]
    for pp in ppp[:-1]:
        path_init=f'{path_init}{split_char}{pp}'
    return f'{path_init}{split_char}{ext}' 


def get_conversion_file(data_path,data_name,split_char='.',ext='csv'):
    data_path0=os.path.join(data_path,data_name)
    with open(data_path0, 'r') as file:
        lines = file.readlines() 

    processed_lines = []
    for line in lines:
        if line.startswith('#'): 
            header = line.strip('# \n').replace(' ', ',')
            processed_lines.append(header)
        else: 
            processed_line = ','.join(line.split())
            processed_lines.append(processed_line)
            
    data_name_exit=get_change_name(data_name,split_char,ext)
    data_path_pro=os.path.join(data_path,data_name_exit) 
    with open(data_path_pro, 'w') as file:
        for line in processed_lines:
            file.write(line + '\n') 
    return data_path_pro

def loadtxt(path,dtype=float): 
    if os.path.exists(path) and os.path.getsize(path)>0:
        cxc=np.loadtxt(path,dtype=dtype) 
        return cxc if cxc.ndim>0 else np.array([]) 
    else:
        return np.array([]) 
    
def format_arrayb(count): 
    if count.ndim == 1: 
        if len(count) == 2:
            if (count[0] == 0) and (count[1] == 0): 
                return count.reshape(1, -1)
    return count

    
def format_array(count): 
    if (count.ndim == 1) and (count.shape[0]==2):  
        return count.reshape(1, -1)
    elif (count.ndim == 2) and (count.shape[0]==1) and (count.shape[1]==2): 
        return count.reshape(1, -1) 
    elif (count.ndim == 2) and (count.shape[0]==1) : 
        return count[0]
    elif (count.ndim == 2) and (count.shape[1]==1) : 
        return np.array(count.flatten().tolist()) 
    else:
        return count 

def loadtxt_count(path):
    if os.path.exists(path):
        return format_array(np.loadtxt(path,dtype=int,ndmin=2))
    else:
        return np.array([]) 

def closest_distances_group( vertices, vertices_group,num_chunks=100): 
    total = len( vertices)
     
    indices = np.linspace(0, total, num_chunks + 1, dtype=int)
     
    chunk_mins = []
    chunk_args = [] 
    for i in range(num_chunks):
        start, end = indices[i], indices[i + 1]
        chunk = vertices[start:end] 
        dist = np.linalg.norm( vertices_group[:, None] - chunk, axis=2) 
        chunk_mins.extend(np.min(dist, axis=0))
        chunk_args.extend(np.argmin(dist, axis=0))
        
    return np.array(chunk_mins).reshape(-1, 1) , np.array(chunk_args).reshape(-1, 1) 


def find_min_max_no_cross(arr,k=None): 
    k=k if k is not None else len(arr)//2 
 
    n = len(arr)
    k = min(k, n)  
    min_val = arr[0]
    max_val = arr[-1]
    min_idx = 0
    max_idx = n - 1
     
    for i in range(min(k, n)):
        if arr[i] <= min_val:
            min_val = arr[i]
            min_idx = i
             
    for i in range(n-1, max(n-k-1, -1), -1):
        if arr[i] >= max_val:
            max_val = arr[i]
            max_idx = i
    
    return max_val, max_idx, min_val, min_idx



def Curve_length(points): 
    return np.sum(np.linalg.norm(np.diff(np.array(points) , axis=0), axis=1) )  



def closest_distances_group( vertices, vertices_group,num_chunks=100): 
    total = len( vertices)
     
    indices = np.linspace(0, total, num_chunks + 1, dtype=int)
     
    chunk_mins = []
    chunk_args = [] 
    for i in range(num_chunks):
        start, end = indices[i], indices[i + 1]
        chunk = vertices[start:end]
 
        dist = np.linalg.norm( vertices_group[:, None] - chunk, axis=2)  
         
        chunk_mins.extend(np.min(dist, axis=0))
        chunk_args.extend(np.argmin(dist, axis=0))
     
    return np.array(chunk_mins).reshape(-1, 1) , np.array(chunk_args).reshape(-1, 1) 












from sklearn.decomposition import PCA  
from sklearn.neighbors import KDTree  
 
def Mapping_inverse(lst):
    return  {val: idx for idx, val in enumerate(lst)}
         
def Connected_vertices_index( vertices_0, cclu: label_cluster,vertices_index): 
        
        if len(vertices_index)<1:
             return vertices_index 
        cclu.Cluster_index(ln_elm=vertices_index)
        cclu.Cluster_faces()
        cclu.Cluster_faces_unique() 
        dend_cclu = curv_mesh(vertices=vertices_0[cclu.cluster_index],
                              faces=cclu.cluster_faces)
        dend_cclu.Vertices_neighbor()
  
        map_inv = Mapping_inverse(cclu.cluster_index)
        dend_cclu_unique = [map_inv[face] for face in cclu.cluster_index] 
        hhh = [list(set(np.concatenate(dend_cclu.vertex_neighbor[ii]))) for ii in dend_cclu_unique]

        indd = []
        for iii in hhh:
            for ii in iii:
                indd.append(ii)

        dend_cclu.Cluster_class(ind_cluster=np.array(indd))
 
        return [cclu.cluster_index[ii] for ii in  dend_cclu.cluster_class_sorted[0]]

 
def Refine_vertices_index( vertices_0, 
                          cclu: label_cluster,
                          vertices_index,
                          vertices_centroid=None,
                          refine_threshold=1.5,
                          kdtree:KDTree=None,
                          neighbor_filter=True,
                          k_neighors=50):
    if kdtree is None:
         kdtree = KDTree(vertices_0, leaf_size=2)
    if len(vertices_index)<1:
            return vertices_index  
    
    
    if neighbor_filter: 
        neighbors = kdtree.query(vertices_0[vertices_centroid], k=k_neighors, return_distance=False).flatten()
    else:
        indices = kdtree.query_radius(((vertices_centroid[0]+vertices_centroid[1])/2).reshape(1, -1),
                                r=np.linalg.norm(vertices_centroid[0]-vertices_centroid[1])/refine_threshold)
        neighbors=indices[0]
    
    if len(neighbors)<1:
         spine_ind = vertices_index
    else:
        spine_ind=np.unique(np.concatenate((vertices_index, neighbors)))
     
    spine_ind = Connected_vertices_index( vertices_0,cclu ,spine_ind)
    spine_ind=np.array(spine_ind)  
    return spine_ind


 
class clust_pca:
    def __init__(self,vertices,vertices_index_shaft,
                 vertices_center=None,
                  vcv_length=None, ):
        pass
        self.centroid= centroid= np.mean(vertices, axis=0) 
        pca  = PCA(n_components=3)
        pca.fit(vertices)
         
        self.normal_vector = pca.components_[0] 
        proj_0 = np.dot(vertices - centroid, self.normal_vector) 

        self.pc_start = centroid +  proj_0.min()* self.normal_vector
        self.pc_end= centroid +  proj_0.max()* self.normal_vector 
        self.pc_bound=np.array([self.pc_start,self.pc_end])

        vertices_shaft_wrap=vertices[vertices_index_shaft] 
        if vertices_center is None:
            self.vertices_shaft_wrap= np.vstack((vertices_shaft_wrap, self.pc_bound))
        else:
            self.vertices_center=vertices_center
            self.vertices_shaft_wrap= np.vstack((vertices_shaft_wrap, self.pc_bound,self.vertices_center))
            self.vcv_length=vcv_length
 


class clust_segment:
    def __init__(self,
                 vertices_0,
                 vertices_index,
                 clu_pca:clust_pca,
                 vertices_shaft_wrap=None,
                 vertices_index_shaft=None,
                 vertices_index_unique=None,
                 size_threshold=500,
                 ):
        pass
        normal_vertices=clu_pca.normal_vector
        centroid_0=clu_pca.centroid
        self.vertices_0=vertices_0
        self.vertices_index_unique=vertices_index_unique
        self.vertices_index=vertices_index
        if vertices_index_shaft is not None:
            vertices_shaft_wrap=clu_pca.vertices_shaft_wrap 
        self.vertices_clu=vertices_clu=vertices_0[self.vertices_index] 
        centroid_clu= np.mean(vertices_clu, axis=0)    
        centroid_clu_proj=  project_points_onto_line(points=centroid_clu.reshape(1,-1),
                                                     vector=normal_vertices,
                                                     z=centroid_0)[0] 

        line_length_1      = np.dot(vertices_clu-centroid_clu_proj,normal_vertices)   
        line_start_1  = centroid_clu_proj + line_length_1.min() *normal_vertices
        line_end_1    = centroid_clu_proj+  line_length_1.max() *normal_vertices  
        iid=(np.dot(vertices_shaft_wrap-line_start_1,line_start_1-line_end_1)<0) & (np.dot(vertices_shaft_wrap-line_end_1,line_end_1-line_start_1)<0)
        self.vertices_shaft_wrap_clu=vertices_shaft_wrap_clu=vertices_shaft_wrap[iid] 
        if len(vertices_shaft_wrap_clu)<size_threshold: 
            self.proj_len=None
        else: 
            self.proj_len=clu_pca.vcv_length[vertices_index]
            mnn=np.argmin(self.proj_len) 
            self.proj_len_min=self.proj_len[mnn]
            self.proj_len_norm=np.abs(self.proj_len[np.argmax(self.proj_len)]-self.proj_len[mnn])
             
    def Cyl_threshold(self,zoom_threshold=0.05):
        if self.proj_len is None:
             return np.array([[0,0,0]])
        else: 
            return self.vertices_clu[self.proj_len>=self.proj_len_min+self.proj_len_norm*(zoom_threshold)]

    def IsSpine(self):
        if self.proj_len is None:
             return False
        else:
            vv=np.zeros_like(self.vertices_0[:,0])
            vv[self.vertices_index]=self.proj_len  
            unique_indices = self.vertices_index_unique
            other_indices = list(set(self.vertices_index) - set(unique_indices))

            max_unique = np.max(vv[unique_indices]) if unique_indices.size else -np.inf
            max_other = np.max(vv[other_indices]) if len(other_indices) else -np.inf

            return max_unique < max_other
         




 
 
from sklearn.decomposition import PCA

def order_points_along_pca(vertices): 
    centroid = vertices.mean(axis=0)
    V = vertices - centroid 
    pca = PCA(n_components=1)
    pca.fit(V)
    pc1 = pca.components_[0]  
    projections = V @ pc1 
    order = np.argsort(projections)
    ordered_points = vertices[order]

    return ordered_points, order, pc1





