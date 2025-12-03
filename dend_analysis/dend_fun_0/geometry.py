import numpy as np 

import plotly.graph_objects as go 


def Rotate_vertices(points, angle, axis='z'): 
    # Define rotation matrices for each axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Apply the rotation matrix to the points
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points



class geometry:
    def __init__(self,vertices,index=None,line_num_points = 1000) -> None: 
        self.index=index
        self.vertices=vertices

        self.tt = np.linspace(0, 1, line_num_points)
        centroid = np.mean(vertices, axis=0) 
        self.centroid_all = centroid

        pca = PCA(n_components=1)
        pca.fit(vertices)
        
        princi_dir = pca.components_[0] 
        
        proj_0 = np.dot(vertices - centroid, princi_dir) 

        dend_points_start = centroid +  proj_0.min()* princi_dir
        dend_points_end= centroid +  proj_0.max()* princi_dir 
        self.dend_points_start,self.dend_points_end=dend_points_start,dend_points_end




    def Cluster_PCA(self,cluster_index=None):
        if cluster_index is None: 
            vertices_cluster=self.vertices
        else:
            vertices_cluster=self.vertices[cluster_index] 
            self.cluster_index=cluster_index
        
        centroid_0 = np.mean(vertices_cluster, axis=0)
        self.vertices_cluster,self.centroid_cluster=vertices_cluster,centroid_0

        pca = PCA(n_components=3)
        pca.fit(vertices_cluster)
        self.pca=pca 


        princi_dir_1 = pca.components_[0]
        proj_1 = np.dot(vertices_cluster - centroid_0,princi_dir_1)   
        line_start_1 = centroid_0 + proj_1.min() *princi_dir_1
        line_end_1 = centroid_0 +proj_1.max()*princi_dir_1 

        self.line_start_1,self.line_end_1=line_start_1,line_end_1





        princi_dir_2 = pca.components_[1] 
        proj_2 = np.dot(vertices_cluster - centroid_0, princi_dir_2)  
        line_start_2 = centroid_0 + proj_2.min() * princi_dir_2
        line_end_2 = centroid_0 + proj_2.max()* princi_dir_2 

        self.line_start_2,self.line_end_2=line_start_2,line_end_2



        princi_dir_3 = pca.components_[2] 
        proj_3 = np.dot(vertices_cluster - centroid_0, princi_dir_3)  
        line_start_3 = centroid_0 + proj_3.min() * princi_dir_3
        line_end_3 = centroid_0 + proj_3.max()* princi_dir_3

        self.line_start_3,self.line_end_3=line_start_3,line_end_3



        cluster_pca_points=np.vstack((line_start_1,line_end_1,line_start_2,line_end_2,line_start_3,line_end_3,centroid_0)) 
        cluster_pca_points_proj=Closest_point_on_line(self.dend_points_start, self.dend_points_end, point=cluster_pca_points)
        proj_dist=np.linalg.norm(cluster_pca_points-cluster_pca_points_proj,axis=1)[:-1] 
        proj_min_max_index=np.where(proj_dist== np.min(proj_dist))[0][0] ,np.where(proj_dist== np.max(proj_dist) )[0][0]   

        self.proj_min_max_index=proj_min_max_index

        self.cluster_pca_points=cluster_pca_points
        self.cluster_pca_points_proj=cluster_pca_points_proj

        self.cluster_farthest_point=cluster_pca_points[proj_min_max_index[1]] 
        self.cluster_closest_point =cluster_pca_points[proj_min_max_index[0]] 

        self.cluster_centroid=cluster_pca_points[-1] 
        self.cluster_centroid_proj_on_pca = self.cluster_pca_points_proj[-1]  


        self.cluster_proj_radius=np.max(np.linalg.norm(self.cluster_centroid_proj_on_pca-cluster_pca_points_proj,axis=1))
        
        
    def Cluster_index_filter(self,cluster_index):
        self.cluster_index=cluster_index.copy() 
        vertices_cluster=self.vertices[cluster_index] 
        # ip=1
        reference_point =self.cluster_closest_point#self.cluster_centroid# ((2**ip-1)*self.cluster_closest_point+self.cluster_centroid)/(2**ip) #
        cluster_edge     =reference_point-vertices_cluster
        vector_director  =reference_point-self.cluster_centroid_proj_on_pca 
        # cluster_edge_2     =self.cluster_closest_point-vertices_cluster
        vector_director_2=reference_point-self.cluster_farthest_point
        cluster_points_proj=Closest_point_on_line(self.dend_points_start, 
                                                         self.dend_points_end, 
                                                         point=vertices_cluster)
        cluster_proj_radius = np.linalg.norm(cluster_points_proj-self.cluster_centroid_proj_on_pca,axis=1)
        self.cluster_index=np.array(
                    self.cluster_index[
                    (np.dot(cluster_edge,vector_director  )<0)& 
                    (np.dot(cluster_edge,vector_director_2)>0)&
                    (cluster_proj_radius <= self.cluster_proj_radius)
                    ])



    def Cluster_proj(self,cluster_index=None):
        if cluster_index is None:
            vertices_cluster=self.vertices 
        else:
            self.cluster_index=cluster_index.copy()
            vertices_cluster=self.vertices[cluster_index]  
            
        self.Cluster_PCA(cluster_index)
        # cluster_edge     =self.cluster_centroid-vertices_cluster
        # vector_director  =self.cluster_centroid-self.cluster_centroid_proj_on_pca 
        # # cluster_edge_2     =self.cluster_closest_point-vertices_cluster
        # vector_director_2=self.cluster_centroid-self.cluster_farthest_point
        self.cluster_points_proj=Closest_point_on_line(self.dend_points_start, 
                                                         self.dend_points_end, 
                                                         point=vertices_cluster)
        self.cluster_proj_radius = np.linalg.norm(self.cluster_points_proj-vertices_cluster,axis=1)

        self.cluster_points_proj_2=Closest_point_on_line(self.line_start_2, 
                                                         self.line_end_2, 
                                                         point=vertices_cluster)
        self.cluster_proj_radius_2 = np.linalg.norm(self.cluster_points_proj_2-vertices_cluster,axis=1)

        self.cluster_points_proj_3=Closest_point_on_line(self.line_start_3, 
                                                         self.line_end_3, 
                                                         point=vertices_cluster)
        self.cluster_proj_radius_3 = np.linalg.norm(self.cluster_points_proj_3-vertices_cluster,axis=1)
    
 

    def Lines(self):
        tt=self.tt
        self.pca_line = np.outer(1-tt, self.dend_points_start) + np.outer(tt,self.dend_points_end)
        self.pca_line_1 = np.outer(1-tt, self.line_start_1) + np.outer(tt, self.line_end_1)
        self.pca_line_2 = np.outer(1-tt,self.line_start_2) + np.outer(tt, self.line_end_2)
        self.pca_line_3 = np.outer(1-tt,self.line_start_3) + np.outer(tt, self.line_end_3)

 


def Lines_plot(v1,v2):
    dim = v1.shape[1]
    x = np.column_stack((v1[:, 0], v2[:, 0])).flatten()
    y = np.column_stack((v1[:, 1], v2[:, 1])).flatten()
    if dim ==3:
        z = np.column_stack((v1[:, 2], v2[:, 2])).flatten()
 
    x = np.insert(x, slice(2, None, 2), None)
    y = np.insert(y, slice(2, None, 2), None)
    if dim == 3:
        z = np.insert(z, slice(2, None, 2), None)
        return np.column_stack((x,y,z))
    else:
        return  np.column_stack((x,y))


import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


import plotly.graph_objects as go
from sklearn.decomposition import PCA
# from scipy.spatial import ConvexHull, Delaunay
# from scipy.ndimage import distance_transform_edt
# import cv2

def dot_(V1, V2, axis=-1):
    return np.sum(V1 * V2, axis=axis, keepdims=True)


def Closest_point_on_line(line_start, line_end, point): 
    
    line_vec = line_end - line_start 
    point_vec = point - line_start 

    line_vec_norm = line_vec / np.linalg.norm(line_vec)#np.repmat(,point.shape[0],point[1])  
    return line_start  + np.outer( np.dot(point_vec, line_vec_norm),line_vec_norm)



def points_skelete_array(vertices, faces, points):
    # Extract vertices of all faces using advanced indexing
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute edges of the triangles
    edge1 = v1 - v0  # Shape: (num_faces, 3)
    edge2 = v2 - v0  # Shape: (num_faces, 3)
    
    # Ray direction (fixed for all points)
    ray_direction = np.array([1.0, 0.5, 0.1])  # Shape: (3,)
    
    # Expand points to be broadcastable with faces (num_points, 1, 3)
    ray_origin = points[:, np.newaxis, :]  # Shape: (num_points, 1, 3)
    
    # Compute the cross product of the ray direction and the second edge
    h = np.cross(ray_direction, edge2)  # Shape: (num_faces, 3)
    
    # Compute dot product of edge1 and h, and reshape for broadcasting
    a = dot_(edge1, h, axis=-1).flatten()  # Shape: (num_faces,)
    
    # Handle near-zero values for robust calculations
    tolerance = 1e-9
    valid_intersections = np.abs(a) > tolerance  # Shape: (num_faces,)
    
    # Prepare to broadcast further calculations only on valid intersections
    f = np.zeros_like(a)
    f[valid_intersections] = 1.0 / a[valid_intersections]
    
    # Calculate vector s from the ray origin to the first vertex (broadcast over points)
    s = ray_origin - v0  # Shape: (num_points, num_faces, 3)
    
    # Calculate u parameter and check bounds
    u = f * dot_(s, h, axis=-1).squeeze()  # Shape: (num_points, num_faces)
    
    # Check if u is within valid range
    valid_u = (u >= 0.0) & (u <= 1.0)  # Shape: (num_points, num_faces)
    
    # Calculate the cross product of s and edge1
    q = np.cross(s, edge1)  # Shape: (num_points, num_faces, 3)
    
    # Calculate v parameter and check bounds
    v = f * dot_(ray_direction, q, axis=-1).squeeze()  # Shape: (num_points, num_faces)
    
    # Check if v is within valid range and u + v <= 1.0
    valid_v = (v >= 0.0) & (u + v <= 1.0)  # Shape: (num_points, num_faces)
    
    # Calculate t to check the intersection along the positive ray direction
    t = f * dot_(edge2, q, axis=-1).squeeze()  # Shape: (num_points, num_faces)
    valid_t = t > tolerance  # Ensure intersection is along the ray's positive direction
    
    # Combine all valid checks
    valid_hits = valid_intersections & valid_u & valid_v & valid_t  # Shape: (num_points, num_faces)
    
    # Count intersections per point
    intersections = np.sum(valid_hits, axis=1)  # Shape: (num_points,)
    
    # Points are skelete if an odd number of intersections occur
    skelete_array = intersections % 2 == 1  # Shape: (num_points,)
    
    return skelete_array


def project_points_onto_vector(points, vector):  
    return ( np.dot(points, vector)[:, np.newaxis] / np.dot(vector, vector)) * vector
     

def project_points_onto_line(points, vector, z): 
    # Calculate the projections of points onto the line defined by vector through point z
    vector=vector = vector / np.linalg.norm(vector)  
    return z  +  np.dot(points - z, vector)[:, np.newaxis] *vector


from scipy.spatial import  distance_matrix

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

    # def Sorted(self):
    #     self.vert_sorted_0,self.vert_sorted_1=self.vert_0[self.ind_x,:],self.vert_1[self.ind_y,:]#, mat_flatten[mat_index]
    # @property
    def Min(self):
        self.vert_min_0,self.vert_min_1=self.vert_0[self.ind_x[0],:],self.vert_1[self.ind_y[0],:]

    # @property
    def Max(self):
        self.vert_max_0,self.vert_max_1=self.vert_0[self.ind_x[-1],:],self.vert_1[self.ind_y[-1],:]
        


class Interpolation_poly:
    def __init__(self,vertices,degrees = [5,6, 7],threshold_detection=1.5):

        # Calculate cumulative distance for parameterization
        cumulative_distance = np.zeros(vertices.shape[0])
        cumulative_distance[1:] = np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))
 

        # Set up a polynomial regression model with cross-validation
        degrees = [6, 7]  # Degrees to test
        best_degree = 0
        best_score = float('-inf')
        best_model = None

        # Try multiple polynomial degrees and select the best one using cross-validation
        for degree in degrees:
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            scores = cross_val_score(model, cumulative_distance.reshape(-1, 1), vertices, scoring='neg_mean_squared_error', cv=5)
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score 
                best_model = model

        # Train the best model on the entire dataset
        best_model.fit(cumulative_distance.reshape(-1, 1), vertices)
        self.fitted_vertices=fitted_vertices = best_model.predict(cumulative_distance.reshape(-1, 1))

        # Calculate residuals for outlier detection
        residuals = np.linalg.norm(vertices - fitted_vertices, axis=1)

        # Define a threshold for outlier detection (e.g., mean + 2*std)
        threshold = np.mean(residuals) + threshold_detection * np.std(residuals)
        self.outliers_index=outliers = residuals > threshold

        # Separate inliers and outliers
        inliers = ~outliers
        self.outlier_vertices= vertices[outliers]

        # Train a new model on the inlier data
        best_model.fit(cumulative_distance[inliers].reshape(-1, 1), vertices[inliers])
        self.fitted_vertices_no_outliers = best_model.predict(cumulative_distance.reshape(-1, 1))

    def Curve_length(self):  
        return np.sum(np.linalg.norm(np.diff(self.fitted_vertices, axis=0), axis=1))  

