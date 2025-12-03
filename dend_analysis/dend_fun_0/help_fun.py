import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.io as pio
 
 
def list_vectices_unique(faces): 
    edges = np.sort(np.vstack([
        faces[:, [0, 1]],   
        faces[:, [1, 2]], 
        faces[:, [2, 0]]  
    ]), axis=1)
     
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True) 
    
    return np.unique( unique_edges[counts == 1] .flatten())  

 


def Closest_point_on_line(line_start, line_end, point): 
    
    line_vec = line_end - line_start 
    point_vec = point - line_start 

    line_vec_norm = line_vec / np.linalg.norm(line_vec)#np.repmat(,point.shape[0],point[1])  
    return line_start  + np.outer( np.dot(point_vec, line_vec_norm),line_vec_norm)

def Rotate_vertices(points, angle, axis='z'):  
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
 
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points

def plotly_scatter(points, marker=None, mode='markers', color='red', symbol=None, size=2, opacity=1.0, name=None, showlegend=True): 
    dim = points.shape[1]
    if marker is None:
        markerr = dict(
            size=size,
            color=color,
            symbol=symbol,
            opacity=opacity  # Adding transparency
        )
    else:
        markerr = marker 
        if 'opacity' not in markerr:
            markerr['opacity'] = opacity

    if dim == 3:
        return go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode=mode,
            marker=markerr,
            name=name,
            showlegend=showlegend
        ) 
    else:
        return go.Scatter(
            x=points[:, 0],
            y=points[:, 1], 
            mode=mode,
            marker=markerr,
            name=name,
            showlegend=showlegend,
        )



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


def plotly_lines(points,points_proj,dash='dash',color='grey',width=2,mode='lines',showlegend=False): 
    connecting_lines = []
    for i in range(points.shape[0]):
        connecting_lines.append(go.Scatter3d(
            x=[points[i, 0], points_proj[i, 0]],
            y=[points[i, 1], points_proj[i, 1]],
            z=[points[i, 2], points_proj[i, 2]],
            mode=mode,
            line=dict(color=color, width=width, dash=dash),
            showlegend=showlegend,
        ))
    return connecting_lines
 

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
            cluster_index = self.cluster_index
        vertices_cluster=self.vertices[cluster_index] 
        self.cluster_index=cluster_index
        
        centroid_0 = np.mean(vertices_cluster, axis=0)
        self.vertices_cluster,self.centroid_cluster=vertices_cluster,centroid_0

        pca = PCA(n_components=3)
        pca.fit(vertices_cluster)

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



        cluster_pca_points=np.row_stack((line_start_1,line_end_1,line_start_2,line_end_2,line_start_3,line_end_3,centroid_0)) 
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


    def Cluster_index_filter(self,cluster_index):
        self.cluster_index=cluster_index.copy() 

        cluster_edge     =self.cluster_centroid-self.vertices_cluster
        vector_director  =self.cluster_centroid-self.cluster_centroid_proj_on_pca 
        vector_director_2=self.cluster_centroid-self.cluster_farthest_point

        self.cluster_index=np.array(
                    self.cluster_index[
                    (np.dot(cluster_edge,vector_director  )<0) & 
                    (np.dot(cluster_edge,vector_director_2)<0)
                    ])



    def Lines(self):
        tt=self.tt
        self.pca_line = np.outer(1-tt, self.dend_points_start) + np.outer(tt,self.dend_points_end)
        self.pca_line_1 = np.outer(1-tt, self.line_start_1) + np.outer(tt, self.line_end_1)
        self.pca_line_2 = np.outer(1-tt,self.line_start_2) + np.outer(tt, self.line_end_2)
        self.pca_line_3 = np.outer(1-tt,self.line_start_3) + np.outer(tt, self.line_end_3)

 

class reconstruct_cluster:
    def __init__(self,vertices,cluu,cluuu_facess,geo_cluster) -> None:  
        self.geom_cluster=geo_cluster
        cluster_faces_index =[]
        ln_elm= cluu.copy()

        for ii in ln_elm:
            for jj in cluuu_facess[ii]:
                cluster_faces_index.append(list(jj)) 

        # cluster_index= np.array(list( set(cluu_faces.flatten())))

        n_upd=set(np.array(cluster_faces_index).flatten()) 
        ln_elm=np.array(list(n_upd.difference(set(ln_elm)))).copy() 
        cluster_index=np.array(list(n_upd)) 
        n_upd_tmp=n_upd.copy()

        self.geom_cluster.Cluster_PCA(cluster_index=cluster_index)
        self.geom_cluster.Cluster_index_filter(cluster_index)
        ln_elm=self.geom_cluster.cluster_index
        
        imax=0
        while (imax<vertices.shape[0]) and (ln_elm.size >0): 
            for ii in ln_elm:
                for jj in cluuu_facess[ii]:
                    cluster_faces_index.append(list(jj))   
            cluster_index=  set(np.array(cluster_faces_index).flatten()) 

            n_upd.update(cluster_index) 
            ln_elm=np.array(list(n_upd.difference(n_upd_tmp))).copy() 
            # print(ln_elm.size)
            if ln_elm.size<=0:
                break
            # ln_elm=np.array(list(n_upd))
            self.geom_cluster.Cluster_index_filter(ln_elm)
            ln_elm=self.geom_cluster.cluster_index
            n_upd_tmp=n_upd.copy()
            imax+=1

        # self.cluster_index=n_upd.copy()
        # self.geom_cluster=geo_cluu
        self.cluster_index =np.array(list(n_upd))  
        self.cluster_faces_index=cluster_faces_index



    def Cluster_faces(self):
        cluster_faces_index=np.unique(self.cluster_faces_index,axis=0)
        faces_cluuu=[]
        cluster_index=self.cluster_index
        if len(cluster_faces_index)>0:
            for i in range(cluster_index.size):
                faces_cluuu.append([
                    np.where(cluster_index==cluster_faces_index[i][0])[0][0],
                    np.where(cluster_index==cluster_faces_index[i][1])[0][0],
                    np.where(cluster_index==cluster_faces_index[i][2])[0][0]
                    ])
        self.cluster_faces=np.array(faces_cluuu)  
        self.cluster_faces_unique=list_vectices_unique(cluster_index[faces_cluuu])  




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
            cluster_index = self.cluster_index
        vertices_cluster=self.vertices[cluster_index] 
        self.cluster_index=cluster_index
        
        centroid_0 = np.mean(vertices_cluster, axis=0)
        self.vertices_cluster,self.centroid_cluster=vertices_cluster,centroid_0

        pca = PCA(n_components=3)
        pca.fit(vertices_cluster)

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



        cluster_pca_points=np.row_stack((line_start_1,line_end_1,line_start_2,line_end_2,line_start_3,line_end_3,centroid_0)) 
        cluster_pca_points_proj=Closest_point_on_line(self.dend_points_start, self.dend_points_end, point=cluster_pca_points)
        proj_dist=np.linalg.norm(cluster_pca_points-cluster_pca_points_proj,axis=1)[:-1] 
        proj_min_max_index=np.where(proj_dist== np.min(proj_dist))[0][0] ,np.where(proj_dist== np.max(proj_dist) )[0][0]   

        self.cluster_proj_radius=np.max(np.linalg.norm(centroid_0-cluster_pca_points_proj,axis=1))
        
        
        self.proj_min_max_index=proj_min_max_index

        self.cluster_pca_points=cluster_pca_points
        self.cluster_pca_points_proj=cluster_pca_points_proj

        self.cluster_farthest_point=cluster_pca_points[proj_min_max_index[1]] 
        self.cluster_closest_point =cluster_pca_points[proj_min_max_index[0]] 

        self.cluster_centroid=cluster_pca_points[-1] 
        self.cluster_centroid_proj_on_pca = self.cluster_pca_points_proj[-1]  


    def Cluster_index_filter(self,cluster_index):
        self.cluster_index=cluster_index.copy() 
        vertices_cluster=self.vertices[cluster_index] 

        cluster_edge     =self.cluster_centroid-vertices_cluster
        vector_director  =self.cluster_centroid-self.cluster_centroid_proj_on_pca 
        # cluster_edge_2     =self.cluster_closest_point-vertices_cluster
        vector_director_2=self.cluster_centroid-self.cluster_farthest_point
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



    def Lines(self):
        tt=self.tt
        self.pca_line = np.outer(1-tt, self.dend_points_start) + np.outer(tt,self.dend_points_end)
        self.pca_line_1 = np.outer(1-tt, self.line_start_1) + np.outer(tt, self.line_end_1)
        self.pca_line_2 = np.outer(1-tt,self.line_start_2) + np.outer(tt, self.line_end_2)
        self.pca_line_3 = np.outer(1-tt,self.line_start_3) + np.outer(tt, self.line_end_3)






import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np

def create_scatter_plot(cluu,dend, geo_cluster ): 
    geo_cluster.Cluster_PCA(cluster_index=cluu)
    geo_cluster.Lines()

    intensity = dend.mean_curv()[cluu]
    vertices_ = dend.vertices[cluu]
    point = geo_cluster.cluster_pca_points
    line_points_1 = geo_cluster.pca_line_1
    line_points_2 = geo_cluster.pca_line_2
    line_points_3 = geo_cluster.pca_line_3
    far_point = geo_cluster.cluster_farthest_point
    close_point = geo_cluster.cluster_closest_point

    scatter = [
        plotly_scatter(points=vertices_, color=intensity),
        plotly_scatter(points=line_points_1, color='blue'),
        plotly_scatter(points=line_points_2, color='black'),
        plotly_scatter(points=line_points_3, color='black'),
        plotly_scatter(points=point, color='green'),
        plotly_scatter(points=far_point.reshape(1, -1), color='yellow', size=7),
        plotly_scatter(points=close_point.reshape(1, -1), color='orange', size=7),
        plotly_scatter(points=point[-1].reshape(1, -1), color='grey', size=7),
    ]
    
    return scatter

def create_subplots(clu, clu_new ,dend,geo_cluster , width=500, height=400):
    # Create subplots
    fig = make_subplots(rows=1, 
                        cols=2, 
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        horizontal_spacing=0.0)
 
    scatter_1 = create_scatter_plot(clu, dend,geo_cluster )
    for scatter in scatter_1:
        fig.add_trace(scatter, row=1, col=1)
 
    scatter_2 = create_scatter_plot(clu_new, dend,geo_cluster )
    for scatter in scatter_2:
        fig.add_trace(scatter, row=1, col=2)

    # Update layout
    fig.update_layout(
        width=2 * width,  
        height=height
    )

    return fig





class reconstruct_cluster2:
    def __init__(self,vertices,cluu,cluuu_facess,geo_cluster) -> None:  
        self.geom_cluster=geo_cluster
        cluster_faces_index =[]
        ln_elm= cluu.copy()

        for ii in ln_elm:
            for jj in cluuu_facess[ii]:
                cluster_faces_index.append(list(jj)) 

        # cluster_index= np.array(list( set(cluu_faces.flatten())))

        n_upd=set(np.array(cluster_faces_index).flatten()) 
        ln_elm=np.array(list(n_upd.difference(set(ln_elm)))).copy() 
        cluster_index=np.array(list(n_upd)) 
        n_upd_tmp=n_upd.copy()

        self.geom_cluster.Cluster_PCA(cluster_index=cluster_index)
        self.geom_cluster.Cluster_index_filter(cluster_index)
        ln_elm=self.geom_cluster.cluster_index
        
        imax=0
        while (imax<vertices.shape[0]) and (ln_elm.size >0): 
            for ii in ln_elm:
                for jj in cluuu_facess[ii]:
                    cluster_faces_index.append(list(jj))   
            cluster_index=  set(np.array(cluster_faces_index).flatten()) 

            n_upd.update(cluster_index) 
            ln_elm=np.array(list(n_upd.difference(n_upd_tmp))).copy() 
            # print(ln_elm.size)
            if ln_elm.size<=0:
                break
            # ln_elm=np.array(list(n_upd))
            self.geom_cluster.Cluster_index_filter(ln_elm)
            ln_elm=self.geom_cluster.cluster_index
            n_upd_tmp=n_upd.copy()
            imax+=1

        # self.cluster_index=n_upd.copy()
        # self.geom_cluster=geo_cluu
        self.cluster_index =np.array(list(n_upd))  
        self.cluster_faces_index=cluster_faces_index



    def Cluster_faces(self):
        cluster_faces_index=np.unique(self.cluster_faces_index,axis=0)
        faces_cluuu=[]
        cluster_index=self.cluster_index
        if len(cluster_faces_index)>0:
            for i in range(cluster_index.size):
                faces_cluuu.append([
                    np.where(cluster_index==cluster_faces_index[i][0])[0][0],
                    np.where(cluster_index==cluster_faces_index[i][1])[0][0],
                    np.where(cluster_index==cluster_faces_index[i][2])[0][0]
                    ])
        self.cluster_faces=np.array(faces_cluuu)  
        self.cluster_faces_unique=hf.list_vectices_unique(cluster_index[faces_cluuu])  









class reconstruct_cluster:
    def __init__(self,vertices,clu,index,cluuu_facess,geo_cluster) -> None:  
        self.geom_cluster=geo_cluster 
        self.clu= clu.copy() 

        self.get_intersection(index)
        ln_elm= np.array(list(self.clu[index])) 
        cluster_faces_index=[]
        for ii in ln_elm:
            for jj in cluuu_facess[ii]:
                cluster_faces_index.append(list(jj)) 
  
        
        n_upd=set(np.array(cluster_faces_index).flatten()) 
        ln_elm=np.array(list(n_upd.difference(set(ln_elm)))).copy() 
        cluster_index=np.array(list(n_upd)) 
        if ln_elm.size>0: 
            n_upd_tmp=n_upd.copy()

            self.geom_cluster.Cluster_PCA(cluster_index=cluster_index)
            self.geom_cluster.Cluster_index_filter(cluster_index)
            ln_elm=self.geom_cluster.cluster_index
            self.clu[index]=ln_elm

        imax=0
        while (imax<vertices.shape[0]) and (ln_elm.size >0) : 
            self.get_intersection(index)
            ln_elm= np.array(list(self.clu[index]))

            for ii in ln_elm:
                for jj in cluuu_facess[ii]:
                    cluster_faces_index.append(list(jj)) 
  

            cluster_index=  set(np.array(cluster_faces_index).flatten()) 

            n_upd.update(cluster_index) 
            ln_elm=np.array(list(n_upd.difference(n_upd_tmp))).copy() 
            # print(ln_elm.size)
            if ln_elm.size<=0:
                break
            # ln_elm=np.array(list(n_upd))
            self.geom_cluster.Cluster_index_filter(ln_elm)
            ln_elm=self.geom_cluster.cluster_index
            n_upd_tmp=n_upd.copy()
            imax+=1
            self.clu[index]=ln_elm

        # self.cluster_index=n_upd.copy()
        # self.geom_cluster=geo_cluu
        self.cluster_index =np.array(list(n_upd))  
        self.cluster_faces_index=cluster_faces_index 


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


    def Cluster_faces(self):
        cluster_faces_index=np.unique(self.cluster_faces_index,axis=0)
        faces_cluuu=[]
        cluster_index=self.cluster_index
        if len(cluster_faces_index)>0:
            for i in range(cluster_index.size):
                faces_cluuu.append([
                    np.where(cluster_index==cluster_faces_index[i][0])[0][0],
                    np.where(cluster_index==cluster_faces_index[i][1])[0][0],
                    np.where(cluster_index==cluster_faces_index[i][2])[0][0]
                    ])
        self.cluster_faces=np.array(faces_cluuu)  
        self.cluster_faces_unique=hf.list_vectices_unique(cluster_index[faces_cluuu])  





class reconstruct_cluster_best:
    def __init__(self,vertices,clu,index,cluuu_facess,geo_cluster) -> None:  
        self.geom_cluster=geo_cluster 
        self.clu= clu.copy() 

        self.get_intersection(index)
        ln_elm= np.array(list(self.clu[index])) 
        cluster_faces_index=[]
        for ii in ln_elm:
            for jj in cluuu_facess[ii]:
                cluster_faces_index.append(list(jj)) 
  
        
        n_upd=set(np.array(cluster_faces_index).flatten()) 
        ln_elm=np.array(list(n_upd.difference(set(ln_elm)))).copy() 
        cluster_index=np.array(list(n_upd)) 
        if ln_elm.size>0: 
            n_upd_tmp=n_upd.copy()

            self.geom_cluster.Cluster_PCA(cluster_index=cluster_index)
            self.geom_cluster.Cluster_index_filter(cluster_index)
            ln_elm=self.geom_cluster.cluster_index
            self.clu[index]=ln_elm

        imax=0
        while (imax<vertices.shape[0]) and (ln_elm.size >0) : 
            self.get_intersection(index)
            ln_elm= np.array(list(self.clu[index]))

            for ii in ln_elm:
                for jj in cluuu_facess[ii]:
                    cluster_faces_index.append(list(jj)) 
  

            cluster_index=  np.array(list(set(np.array(cluster_faces_index).flatten())))
            self.geom_cluster.Cluster_index_filter(cluster_index)
            ln_elm=self.geom_cluster.cluster_index

            # ln_elm=set(np.array(cluster_faces_index).flatten())

            n_upd.update(ln_elm) 
            ln_elm=np.array(list(n_upd.difference(n_upd_tmp))).copy() 
            # print(ln_elm.size)
            if ln_elm.size<=0:
                break
            # ln_elm=np.array(list(n_upd))
            n_upd_tmp=n_upd.copy()
            imax+=1
            self.clu[index]=ln_elm

        # self.cluster_index=n_upd.copy()
        # self.geom_cluster=geo_cluu
        self.cluster_index =np.array(list(n_upd))  
        self.cluster_faces_index=cluster_faces_index 


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


    def Cluster_faces(self):
        cluster_faces_index=np.unique(self.cluster_faces_index,axis=0)
        faces_cluuu=[]
        cluster_index=self.cluster_index
        if len(cluster_faces_index)>0:
            for i in range(cluster_index.size):
                faces_cluuu.append([
                    np.where(cluster_index==cluster_faces_index[i][0])[0][0],
                    np.where(cluster_index==cluster_faces_index[i][1])[0][0],
                    np.where(cluster_index==cluster_faces_index[i][2])[0][0]
                    ])
        self.cluster_faces=np.array(faces_cluuu)  
        self.cluster_faces_unique=hf.list_vectices_unique(cluster_index[faces_cluuu])  






# class neck:
#     def __init__(self,vertices_0,node:hff.dendrite) -> None: 
#         vertices_index_unique=node.vertices_index_unique#node.vertices_index[node.unique_vertices]
#         vertices_index_align= np.concatenate((vertices_index_unique,
#         np.array([int(ii) for ii in list(set([int(ii) for ii in node.vertices_index])-set(vertices_index_unique))])))

#         # vertices_index_unique=node.vertices_index[node.unique_vertices]
#         vertices_clu_unique=vertices_0[vertices_index_unique,:]
#         vertices_unique_centroid= np.mean(vertices_clu_unique,axis=0)

#         vertices_clu_unique_size=vertices_index_unique.size
#         points=vertices_clu_new =vertices_0[vertices_index_align,:] 
#         # vertices_clu_unique=vertices_0[vertices_index_unique,:]
#         vertices_unique_centroid= np.mean(vertices_clu_unique,axis=0)

#         pca = PCA(n_components=3) 
#         pca.fit(points)
#         components = pca.components_ 
#         normal_vector  = components[0]
#         centroid = np.mean(points, axis=0) 
#         direction=int(np.sign(np.dot(vertices_unique_centroid - centroid, normal_vector)))
#         direction

#         self.vertices_unique_proj= geo.project_points_onto_line(points=vertices_clu_unique,vector=normal_vector,z=centroid) 
#         index_neck=np.argmin(np.linalg.norm(self.vertices_unique_proj-centroid,axis=1))
#         self.vertices_unique_proj_closed=self.vertices_unique_proj[index_neck,:]
#         self.vertices_unique_neck = vertices_clu_unique[index_neck,:]

#         direction = np.dot(vertices_unique_centroid - centroid, normal_vector)

#         self.pca_save=[]
#         self.centroid_save =[]
#         self.normal_vector=[]
#         cts=0
#         stp=1 
#         while stp and (points.shape[0]>vertices_clu_unique_size):
#             group_0 = []
#             group_1 = []
#             group_2 = []
#             group_3 = []
            
#             for point in points: 
#                 projection = np.dot(point - centroid,self.vertices_unique_proj_closed- centroid)
#                 if  ( (np.dot(point - self.vertices_unique_proj_closed,self.vertices_unique_proj_closed- centroid)<0)): #((direction>0) and (projection>0)) or ( (direction<=0) and (projection<=0)): # or ( (direction<=0) and (projection<=0)):(direction>0) and 
#                     group_1.append(point)
#                 else:
#                     group_2.append(point)
#                     # if (np.dot(point - centroid,self.vertices_unique_proj_closed- centroid)<0):
#                     #     group_2.append(point)
                    
#                     # else:
#                     #     group_3.append(point) 

#             # group_0 = np.array(group_0)
#             group_1 = np.array(group_1)
#             group_2 = np.array(group_2) 
            
#             self.centroid_save.append(centroid)
#             self.pca_save.append(pca) 
#             self.normal_vector.append(normal_vector)

#             # print( points.shape[0],vertices_clu_unique_size)
#             llnorm=np.linalg.norm(points[:vertices_clu_unique_size,:]-vertices_clu_unique)
#             # llnorm=1
#             print(llnorm,points.shape[0],vertices_clu_unique_size)
#             if llnorm>0:   
#                 stp=0
                
#                 # for point in group_2:
#                 #     # ttt=np.linalg.norm(vertices_clu_new-point,axis=1)
#                 #     # indd = np.where(ttt<=1e-6)  
#                 #     self.vertices_clu_new=vertices_clu_new=np.delete(vertices_clu_new,np.where(vertices_clu_new==point)[0][0], axis=0)
#                 break   
#             self.core_bellow=group_1  
#             self.core_above=group_2  
#             self.core=points#np.vstack((group_1,group_2)) 

#             points=group_1
#             pca = PCA(n_components=3) 
#             pca.fit(points)
#             components = pca.components_ 
#             normal_vector  = components[0]
#             centroid= np.mean(points, axis=0) 
#             projection = np.dot(point-centroid, normal_vector)
#             cts+=1
 
#         group_0 = []
#         group_1 = [] 
#         group_2 = []
#         # ctss=cts-3
#         centroid_save=self.centroid_save
#         ctss=len(centroid_save)-2
#         self.centroid_neck=centroid_save[ctss] 
#         self.centroid=centroid=centroid_save[ctss]
#         self.components=components = self.pca_save[ctss].components_
#         self.normal_vector=normal_vector  =self.normal_vector[ctss]
#         self.pca=self.pca_save[ctss]


#         for point in vertices_clu_new: 
#             projection = np.dot(point- self.vertices_unique_proj_closed, normal_vector)
#             projection_2 = np.dot(point- self.centroid, self.vertices_unique_proj_closed-self.centroid)
#             if( (np.dot(point - self.vertices_unique_proj_closed,self.vertices_unique_proj_closed- centroid)>0)):#     ((direction>0) and (projection>0)) or ( (direction<=0) and (projection<=0)): #  ((direction>0) and (projection>0)) or ( (direction<=0) and (projection<=0)): #((direction>0) and (projection>0)) :#or ( (direction<=0) and (projection<=0)):# 
#                 group_0.append(point)
#             else:
#                 group_1.append(point)
#                 if projection_2>0:
#                     group_2.append(point) 
#                 else:
#                     group_3.append(point)
#         self.core_off = np.array(group_0) 
#         self.core = np.array(group_1)
#         self.core_neck = np.array(group_2) 
#         self.core_head = np.array(group_3)  

#         # self.vertices_up=points[np.dot(points - self.centroid, self.vertices_unique_proj_closed- self.centroid)<0,:]
#         # self.vertices_up_proj= geo.project_points_onto_line(points=self.vertices_up,vector=normal_vector,z=centroid) 
#         # index_up=np.argmax(np.linalg.norm(self.vertices_up_proj-centroid,axis=1))
#         self.vertices_up_fartest=None#self.vertices_up_proj[index_up ,:]

