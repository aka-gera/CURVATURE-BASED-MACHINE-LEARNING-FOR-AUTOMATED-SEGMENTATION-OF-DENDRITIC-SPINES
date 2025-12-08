

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from dend_fun_0.curvature import curv_mesh as curv_mesh
import dend_fun_0.help_fun as hf 
import dend_fun_0.geometry as geo 
from dend_fun_0.help_funn import dendrite,curv_mesh, cluster_class,label_cluster,dendrite_io,mmean
from sklearn.neighbors import KDTree 
import numpy as np 
from scipy.spatial import  distance_matrix
from scipy.interpolate import splprep, splev
from scipy.integrate import quad 
from scipy.spatial import cKDTree 

from scipy.spatial import  distance_matrix
from itertools import permutations 
import numpy as np
from scipy.interpolate import splprep, splev


from dend_fun_0.curvature import curv_mesh as curv_mesh
 
# from vedo import Points



 

 

def shortest_path(points):
    min_path_length = float('inf')
    best_path = None 
    for perm in permutations(range(len(points))):
        path_length = sum(np.linalg.norm(points[perm[i]] - points[perm[i+1]]) for i in range(len(points) - 1))
        if path_length < min_path_length:
            min_path_length = path_length
            best_path = perm

    # print( 'end')
    return min_path_length, best_path


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

    # @property
    def Max(self):
        self.vert_max_0,self.vert_max_1=self.vert_0[self.ind_x[-1],:],self.vert_1[self.ind_y[-1],:]
        


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

    # @property
    def Max(self):
        self.vert_max_0,self.vert_max_1=self.vert_0[self.ind_x[-1],:],self.vert_1[self.ind_y[-1],:]
  
class spline_interpolator:
    def __init__(self, vertices, line_num_points=50, spline_smooth=2):
        points = vertices.T   
        
        if vertices.shape[0] > 2: 
            self.tck, self.u = splprep([points[0], points[1], points[2]], s=spline_smooth, k=max(1, min(3, len(points[0]) - 1)))
            
            x_fine, y_fine, z_fine = splev(np.linspace(0, 1, line_num_points), self.tck)
            self.curve_vertices = np.column_stack([x_fine, y_fine, z_fine])   
            self.curve_length=np.array(Curve_length(self.curve_vertices))
        
        elif vertices.shape[0] == 2: 
            self.curve_vertices = vertices
            self.curve_length = np.linalg.norm(vertices[1] - vertices[0])
        
        else: 
            self.curve_vertices = vertices
            self.curve_length = 0

 
    def curve_speed(self, u):
        dx, dy, dz = splev(u, self.tck, der=1) 
        return np.sqrt(dx**2 + dy**2 + dz**2)



def project_points_onto_line(points, vector, z):  
    vector=vector = vector / np.linalg.norm(vector)  
    return z  +  np.dot(points - z, vector)[:, np.newaxis] *vector 

 
 


class skeleton:
    def __init__(self,vertices_0, line_num_points = 1000,pca=None) -> None: 

        self.vertices=vertices_0
        self.line_points = np.linspace(0, 1, line_num_points)
        self.pca=pca
    def Branch(self):
        vertices=self.vertices 
        self.centroid_branch = centroid= mmean(vertices, axis=0) 

        self.pca_branch=self.pca=pca_branch = PCA(n_components=1)
        pca_branch.fit(vertices)
        
        princi_dir = pca_branch.components_[0]  
        proj_0 = np.dot(vertices - centroid, princi_dir) 

        self.branch_start = centroid +  proj_0.min()* princi_dir
        self.branch_end= centroid +  proj_0.max()* princi_dir  


    def Cluster(self,spn_core =None,vertices=None,line_num_points=10) -> None:   
        self.vertices_spp=vertices  
        if (spn_core is None):
            self.centroid=  mmean(vertices, axis=0) 
            self.pca = PCA(n_components=3) 
            self.pca.fit(self.vertices_spp)
            self.pc3,self.pc2,self.pc1  =self.pca_components =self.pca.components_
             # print('im nope')
        else: 
            if vertices is None:
                self.vertices_spp=spn_core.neck_vertices 
            self.centroid= spn_core.centroid 
            self.pc3,self.pc2,self.pc1=self.pca_components=spn_core.pca_components 

        if line_num_points is None:
            self.line_points =self.line_points 
        else:
            self.line_points=np.linspace(0,1,line_num_points)
         
 

    def Cluster_PCA(self,vertices=None,centroid=None,pca=None,pca_components=None,line_num_points = None,normal_vertices=None):
        if line_num_points is None:
            self.line_points =self.line_points 
        else:
            self.line_points=np.linspace(0,1,line_num_points)
        if centroid is None:
            centroid=self.centroid

        if pca_components is None:
            pca_components =self.pca_components 
        else: 
            pc3,pc2,pc1  = pca_components

        if vertices is None:
            vertices_spp=self.vertices_spp -centroid
            # pca=self.pca
            pc3,pc2,pc1  =pca_components #self.pca.components_   
        else:
            vertices_spp=vertices-centroid 
        
        pc3,pc2,pc1  =pca_components  
 
 
        line_length_1      = np.dot(vertices_spp,pc1)   
        self.line_start_1  = centroid + line_length_1.min() *pc1
        self.line_end_1    = centroid+  line_length_1.max() *pc1 
 
 
        line_length_2      = np.dot(vertices_spp, pc2)  
        self.line_start_2  = centroid + line_length_2.min() * pc2
        self.line_end_2    = centroid + line_length_2.max() * pc2 
 
 
        line_length_3      = np.dot(vertices_spp, pc3)  
        self.line_start_3  = centroid + line_length_3.min() * pc3
        self.line_end_3    = centroid + line_length_3.max() * pc3 


    def Cluster_diameter(self):
        diam_vertices= [np.vstack(([self.line_start_2,self.line_end_2])) ,np.vstack((self.line_start_1,self.line_end_1))]

        diam = np.linalg.norm(np.vstack((self.line_start_2-self.line_end_2,self.line_start_1-self.line_end_1)),axis=1)
        diam_sort=np.argsort(diam)#[::-1]
        self.diam = diam[diam_sort]
        self.diam_vertices=[diam_vertices[diam_sort[0]],diam_vertices[diam_sort[1]]] 
        self.diam_start=self.vertices_spp[np.argmin(np.linalg.norm(self.diam_vertices[1][0]-self.vertices_spp,axis=1) )].reshape(1,-1)
        self.diam_end=self.vertices_spp[np.argmin(np.linalg.norm(self.diam_vertices[1][1]-self.vertices_spp,axis=1) )].reshape(1,-1)
        self.diam_length=float(np.linalg.norm(self.diam_start-self.diam_end)) 


 
    def Proj_pca_line_2(self,semi_line=True): 
        self.cluster_pca_points_proj=hf.Closest_point_on_line(self.diam_vertices[1][0], self.diam_vertices[1][1], point=self.vertices_spp) 
        # self.diam_vertices_start=self.cluster_pca_points_proj[np.dot(self.cluster_pca_points_proj-self.centroid,self.diam_vertices[1][0]-self.centroid)>0,:]
        proj_dist=np.linalg.norm(self.diam_vertices[1][0]-self.cluster_pca_points_proj,axis=1) 
        self.diam_start=self.vertices_spp[np.argmin(proj_dist)].reshape(1,-1)

        # self.diam_vertices_end=self.cluster_pca_points_proj[np.dot(self.cluster_pca_points_proj-self.centroid,self.diam_vertices[1][1]-self.centroid)>0,:]
        proj_dist=np.linalg.norm(self.diam_vertices[1][1]-self.cluster_pca_points_proj,axis=1) 
        self.diam_end=self.vertices_spp[np.argmin(proj_dist)].reshape(1,-1)

 
    def Proj_pca_line_3(self,semi_line=True): 
        self.cluster_pca_points_proj=hf.Closest_point_on_line(self.line_start_3, self.line_end_3, point=self.vertices_spp)
        if semi_line: 
            self.cluster_pca_points_proj=self.cluster_pca_points_proj[np.dot(self.cluster_pca_points_proj-self.centroid,self.line_start_3-self.centroid)>0,:]
        proj_dist=np.linalg.norm(self.line_start_3-self.cluster_pca_points_proj,axis=1) 
        self.line_start_3_proj=self.vertices_spp[np.argmin(proj_dist)].reshape(1,-1)


    def Proj_pca_branch(self): 

        cluster_pca_points=np.vstack((self.line_start_1,
                                      self.line_end_1,
                                      self.line_start_2,
                                      self.line_end_2,
                                      self.line_start_3,
                                      self.line_end_3,
                                      self.centroid)) 
        cluster_pca_points_proj=hf.Closest_point_on_line(self.branch_start, self.branch_end, point=cluster_pca_points)
        proj_dist=np.linalg.norm(cluster_pca_points-cluster_pca_points_proj,axis=1)[:-1] 

        self.pca_points=cluster_pca_points
        self.pca_points_proj=cluster_pca_points_proj

        self.farthest_point=cluster_pca_points[np.argmax(proj_dist),:].reshape(1,-1)
        self.closest_point =cluster_pca_points[np.argmin(proj_dist),:].reshape(1,-1)
        self.branch_start=self.branch_start.reshape(1,-1)
        self.branch_end=self.branch_end.reshape(1,-1)
        self.centroid_proj_on_pca = self.pca_points_proj[-1]  


    def Lines(self,line_num_points = None):
        if line_num_points is None:
            self.line_points =self.line_points 
        else:
            self.line_points=np.linspace(0,1,line_num_points)
        tt=self.line_points
        
        # self.branch     = np.outer(1-tt,self.branch_start) + np.outer(tt, self.branch_end)
        self.pca_line_1 = np.outer(1-tt,self.line_start_1) + np.outer(tt, self.line_end_1)
        self.pca_line_2 = np.outer(1-tt,self.line_start_2) + np.outer(tt, self.line_end_2)
        self.pca_line_3 = np.outer(1-tt,self.line_start_3) + np.outer(tt, self.line_end_3)


    def Centroid_curv(self,pca_line=None,semi_line=True,vertices=None): 
        if pca_line is None:
            pca_line=self.pca_line_3
        if semi_line: 
               pca_line=pca_line[np.dot(pca_line-self.centroid,self.line_start_3-self.centroid)>0,:]
        if vertices is None:
            vertices =self.vertices_spp

        centroid_curv=[]
        self.pca_vector_director  = pca_line[0,:]- pca_line[-1,:] 
        for i in range(pca_line.shape[0]-1 ):
            vertices_seq= vertices[
            (np.dot(pca_line[i,:]  - vertices,self.pca_vector_director )>=0)&
            (np.dot(pca_line[i+1,:]-vertices,self.pca_vector_director )<0),:
            ]  
            if vertices_seq.shape[0]>0:
                centroid_curv.append(mmean(vertices_seq,axis=0) )  
        self.centroid_curv=np.array(centroid_curv) 



    def Centroid_curv_threshold(self,threshold_detection=.05): 
        vertices_proj=  project_points_onto_line(points=self.centroid_curv,
                                                        vector=self.pca_vector_director ,
                                                        z=self.centroid) 
        proj_length=np.linalg.norm( vertices_proj-self.centroid_curv,axis=1) 
        print( mmean(proj_length) ,np.std(proj_length))  
        outliers_index=  proj_length-mmean(proj_length) < threshold_detection * np.std(proj_length) 
        self.centroid_curv_threshold=self.centroid_curv[outliers_index]


    def Centroid_curv_threshold_far(self,threshold_detection=.05): 
        vertices_proj=  project_points_onto_line(points=self.centroid_curv,
                                                        vector=self.pca_vector_director ,
                                                        z=self.centroid) 
        proj_length=np.linalg.norm( vertices_proj-self.centroid_curv,axis=1) 
        print( mmean(proj_length) ,np.std(proj_length))  
        outliers_index=  proj_length-mmean(proj_length) > threshold_detection * np.std(proj_length) 
        self.centroid_curv_threshold=self.centroid_curv[outliers_index]



    def Cluster_heigth_length(self,poly_degree=4):
        self.Centroid_curv()
        centroid_curv=self.centroid_curv
        if centroid_curv.shape[0]>4:
            inter_poly=geo.Interpolation_poly(vertices=centroid_curv,degrees=[poly_degree]) 
            self.cluster_heigth_length=inter_poly.Curve_length()
        else:
            self.cluster_heigth_length=0. 


    def Cluster_metric(self,):
        centroid_seq=[]
        vertices_min=[] 
        vertices_max=[]  
        self.vertices_min_ort_cen=np.zeros_like(self.centroid_curv[:-1,:]) 

        pca_line=self.pca_line_3
        vector_director  = pca_line[0,:]- pca_line[-1,:]#self.normal_vector#self.pca_line_3[i,:]-self.pca_line_3[i+1,:]
        for i in range(pca_line.shape[0]-1 ):
            vertices_seq=self.vertices_spp[
            (np.dot(pca_line[i,:]  -self.vertices_spp,vector_director)>=0)&
            (np.dot(pca_line[i+1,:]-self.vertices_spp,vector_director)<0),:
            ] 
            # print(vertices_seq)
            if vertices_seq.shape[0]>0: 
                centroid_s=mmean(vertices_seq,axis=0)
                centroid_seq.append(centroid_s) 

                dist_sorted=geo.Sorted_distance_matrix(vert_0=vertices_seq, vert_1=vertices_seq)
                dist_sorted.Max() 
                vert_0,vert_1=dist_sorted.vert_max_0,dist_sorted.vert_max_1
                vertices_min.append(vert_0)
                vertices_max.append(vert_1)  
  
        self.centroid_seq=centroid_seq=np.array(centroid_seq)
        self.vertices_min=vertices_min=np.array(vertices_min) 
        self.vertices_max=vertices_max=np.array(vertices_max) 
 
         

    def Spine_neck_metric(self,line_num_points=None): 
        if line_num_points is None:
            self.line_points =self.line_points 
        else:
            self.line_points=np.linspace(0,1,line_num_points) 
        
        chck_max=0
        chck_min=1e9
        chck_max_tmp=0   
        centroid_s=centroid=(self.centroid+self.vertices_unique_proj_closed) /2
        cc=1
        while cc>0.05:
            centroid=centroid_s
            # print(centroid)
            centroid_s=(centroid_s+self.vertices_unique_proj_closed) /2
            cc=np.linalg.norm(centroid-centroid_s)

        pca_line= np.outer(1-self.line_points,self.vertices_unique_proj_closed) + np.outer(self.line_points,centroid)#self.pca_line_3
        neck_proj=(np.dot(self.vertices_spp-centroid,self.vertices_unique_proj_closed-centroid)>0)
        vector_director  =(self.vertices_unique_proj_closed-centroid) 
        for i in range(pca_line.shape[0]-1  ):
            vertices_seq=self.vertices_spp[ 
            neck_proj&
            (np.dot(pca_line[i,:]  -self.vertices_spp,vector_director)>0)&
            (np.dot(pca_line[i+1,:]-self.vertices_spp,vector_director)<=0),:
            ]     
            if vertices_seq.shape[0]>0:  
                dist_sorted=geo.Sorted_distance_matrix(vert_0=vertices_seq, vert_1=vertices_seq)
                dist_sorted.Max() 
                vert_0,vert_1=dist_sorted.vert_max_0,dist_sorted.vert_max_1 
                neck_norm_unit=np.linalg.norm(vert_0-vert_1) 
                if neck_norm_unit<= chck_min:
                    chck_min=neck_norm_unit    
                    self.neck_radius_seq=vertices_seq
                    self.neck_radius=neck_norm_unit
                    self.neck_radius_points=np.vstack((vert_0,vert_1)) 
         

    def Cluster_head_metric(self,line_num_points=None): 
        if line_num_points is None:
            self.line_points =self.line_points 
        else:
            self.line_points=np.linspace(0,1,line_num_points) 
        chck_max=0
        chck_max_tmp=0   
        chck_min=1e9  
        pca_line=self.pca_line_3
        vector_director  = pca_line[0,:]- pca_line[-1,:]
        for i in range(pca_line.shape[0]-1 ): 
            vertices_seq=self.vertices_spp[
            (np.dot(pca_line[i,:]  -self.vertices_spp,vector_director)>=0)&
            (np.dot(pca_line[i+1,:]-self.vertices_spp,vector_director)<=0) 
            ]  
            if vertices_seq.shape[0]>0:  
                dist_sorted= Sorted_distance_matrix(vert_0=vertices_seq, vert_1=vertices_seq) 
                dist_sorted.Max() 
                vert_0,vert_1=dist_sorted.vert_max_0,dist_sorted.vert_max_1 
                neck_norm_unit=np.linalg.norm(vert_0-vert_1) 

                self.head_diam_seq=vertices_seq 
                self.head_diam_points=np.vstack((vert_0,vert_1)) 
                self.head_diam_centroid=(pca_line[i,:]+pca_line[i+1,:])/2
                
                if (neck_norm_unit>chck_max): 
                    chck_max_tmp+=1
                    chck_max=neck_norm_unit  
                    self.head_diam_seq=vertices_seq 
                    self.head_diam_points=np.vstack((vert_0,vert_1)) 
                    self.head_diam_centroid=(pca_line[i,:]+pca_line[i+1,:])/2
        self.Cluster_PCA(centroid=self.head_diam_centroid) 
        self.Cluster_diameter() 
        self.head_diam_length=self.diam_length


  


# from vedo import Points
# def smooth( vertices=None, 
#                 f=100,
#                 N=10,
#                     subsample_thre=.005,
#                   ):
#     if vertices.shape[0]>3: 
#         if vertices.shape[0]>18:
#             pcl_sp = Points(vertices).subsample(subsample_thre)  
#             print('pcl_sp.points',pcl_sp.points.shape)
#             # f=10
#         else:
#             pcl_sp = Points(vertices) 
#         for i in range(N):
#             pcl_sp = pcl_sp.clone().smooth_mls_1d(f=f) 
#         return pcl_sp.points
#     else:
#         return np.mean(vertices,axis=0).reshape(1,-1)
    

class center_curvature:
    def __init__(self,
                 vertices,
                 vertices_index=None,
                 line_num_points=200,
                 line_num_points_inter=300,
                 spline_smooth=1,
                 get_norm=True,
                 f=3,
                    N=2,
                    smooth_tf=False,
                    subsample_thre=.01,): 
        
        if vertices_index is None:
            vertices_index=np.arange(vertices.shape[0])

        skl = skeleton(vertices_0=vertices,)
        if smooth_tf:
            vertices_smooth=self.smooth( vertices=vertices[vertices_index],  
                    f=f,
                    N=N, 
                    subsample_thre=subsample_thre,
                        )
        else:
            vertices_smooth=vertices[vertices_index]
        skl.Cluster( vertices= vertices_smooth)
        skl.Cluster_PCA()
        skl.Lines(line_num_points=line_num_points)  
        # skl.Cluster_PCA(vertices=vertices_0[shaft_index],line_num_points = line_num_points) 
        skl.Centroid_curv(semi_line=False)
        self.centroid_curv=skl.centroid_curv
        self.inter_poly = spline_interpolator(vertices=skl.centroid_curv,spline_smooth=spline_smooth,line_num_points= line_num_points_inter  ) 
        self.vertices_center=self.inter_poly.curve_vertices 
        if self.vertices_center.shape[0]>0:
            self.vertices_center_length=self.inter_poly.curve_length
            if get_norm:
                self.vcv_length= np.min(np.linalg.norm(self.inter_poly.curve_vertices[:, None] - vertices, axis=2), axis=0)
        else:
            self.vertices_center=vertices_smooth
            if get_norm:
                self.vcv_length= self.vertices_center_length=Curve_length(vertices_smooth)
        if self.vcv_length.ndim==0:
            self.vcv_length=np.array([self.vcv_length])
            # np.min(np.linalg.norm(self.inter_poly.curve_vertices[:, None] - vertices, axis=2), axis=0)
    def smooth(self, vertices=None, 
                f=10,
                N=10,
                 subsample_thre=None ):
        return smooth( vertices=vertices, 
                f=f,
                N=N,
                 subsample_thre=subsample_thre, )
    


 
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


def get_kmean_mean(dist_norm, n_clusters=4, kmean_max_iter=500):
    pca_radius=np.array(dist_norm).reshape(-1,1)
 

    cluster1 = KMeans(n_clusters=n_clusters, max_iter=kmean_max_iter, n_init='auto')   
    labels = cluster1.fit_predict(pca_radius)
 
    means = np.array([dist_norm[labels == i].mean() for i in range(n_clusters)])
 
    return means[labels]







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

  




def align_points(points, tolerance=1e-6, max_iter=100): 
     
    tree = cKDTree(points)

    points_A = points   
    points_B = points + 0.000005  
    aligned_points_B = points_B.copy() 
    for _ in range(max_iter): 
        distances, indices = tree.query(aligned_points_B) 
        if np.all(np.abs(distances) < tolerance):
            break 
    return  points_A[indices]
 
 

def get_center_line(vertices ,
                    vertices_center,
                    subsample_thre=.02,
                    f=0.99,
                    N=10,
                    num_chunks=100, 
                    line_num_points=300,
                    line_num_points_inter=700,
                    spline_smooth=1.,
                    num_points=50,
                    ctl_run_thre=1,
                    smooth_mls=False,
                    ):
    # if smooth_mls:
    #     pcl_sp = Points( vertices ).subsample(subsample_thre)  
    #     for i in range(N):
    #         pcl_sp = pcl_sp.clone().smooth_mls_1d(f=f) 
    #     vertices_ii=pcl_sp.points
    # else:
    vertices_ii=vertices
    clo_min,gf= closest_distances_group(vertices_ii, vertices_center,num_chunks=num_chunks)
    nnn=vertices_ii[np.argmin(clo_min)]
    mmm=vertices_center[gf[np.argmin(clo_min)]].reshape(1,-1) 

    points_be  = np.linspace(mmm.flatten() , nnn.flatten() , num_points, axis=0)
 
    vec=np.vstack((points_be,vertices_ii))



    # vec=align_points(vec) 
    jj=0
    while jj<ctl_run_thre:
        ctl_tmp = center_curvature(vertices=vec, 
                line_num_points= line_num_points,
                line_num_points_inter= line_num_points_inter,
                spline_smooth= spline_smooth,
            )
        jj+=1 
    ctl_tmp.vertices_center = ctl_tmp.vertices_center if np.linalg.norm(mmm - ctl_tmp.vertices_center[-1, :]) > np.linalg.norm(mmm - ctl_tmp.vertices_center[0, :]) else ctl_tmp.vertices_center[::-1, :]

    return ctl_tmp,mmm



def get_center_lines(vertices ,
                    vertices_center,
                    subsample_thre=.02,
                    f=0.99,
                    N=10,
                    num_chunks=100, 
                    line_num_points=300,
                    line_num_points_inter=700,
                    spline_smooth=1.,
                    num_points=50,
                    ctl_run_thre=1):
    ctl_tmp,mmm=get_center_line(vertices =vertices, 
                            vertices_center=vertices_center ,
                            subsample_thre=subsample_thre,
                            f=f,
                            N=N,
                            num_chunks=num_chunks,
                            num_points=num_points)
    ctl_tmp,_=get_center_line(vertices=ctl_tmp.vertices_center,
                            vertices_center=vertices_center ,
                            subsample_thre=subsample_thre,
                            f=f,
                            N=N,
                            num_chunks=num_chunks,
                            num_points=num_points)
    jj=0
    while jj<ctl_run_thre:
        ctl_tmp = center_curvature(vertices=ctl_tmp.vertices_center, 
                line_num_points= line_num_points,
                line_num_points_inter= line_num_points_inter,
                spline_smooth= spline_smooth,
            )
        jj+=1
    ctl_tmp.vertices_center = ctl_tmp.vertices_center if np.linalg.norm(mmm - ctl_tmp.vertices_center[-1, :]) < np.linalg.norm(mmm - ctl_tmp.vertices_center[0, :]) else ctl_tmp.vertices_center[::-1, :] 
    return ctl_tmp




 
 





