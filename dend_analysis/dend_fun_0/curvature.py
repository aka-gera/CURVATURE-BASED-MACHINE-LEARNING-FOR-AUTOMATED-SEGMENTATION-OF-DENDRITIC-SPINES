# 
import numpy as np   
import os
from dend_fun_0.help_plotly import plotly_mesh,plotly_analysis,plotly_convergence_rate
import trimesh
from tqdm import tqdm 

from sklearn.neighbors import KDTree 
eps = 1e-14 
  
 

def convergence_rate(error):
    sav = np.zeros((error.shape[0],error.shape[1]-1))
    for i in range(error.shape[1]-1):
        sav[:,i] = -(np.log(error[:,i])-np.log(error[:,i+1]))/np.log(2) 

    return sav


 

def vectices_adjacent(faces,index_Judge): 

    LocalMat = []
    for face in faces:
        if face[0] == index_Judge:
            LocalMat.append(face)
        elif face[1] == index_Judge:
            LocalMat.append([face[1], face[2], face[0]])
        else:
            LocalMat.append([face[2], face[0], face[1]])
    return np.array(LocalMat)


 


def norm_weighted(val,weight,ord=2): return (np.sum((np.abs(val)**ord)*weight))**(1/ord)



def pad_list(lst, size, pad_value=-1): return lst + [pad_value] * (size - len(lst))





def list_vectices_unique(faces): 
    edges = np.sort(np.vstack([
        faces[:, [0, 1]],   
        faces[:, [1, 2]], 
        faces[:, [2, 0]]  
    ]), axis=1)
     
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True) 
    
    return np.unique( unique_edges[counts == 1] .flatten())  




class curv_mesh:
    def __init__(self,
                 vertices,
                 faces,
                 faces_class=None,
                 faces_class_faces=None,
                 vertex_neighbor=None ,
                 mean_curv=None,
                 gauss_curv=None,
                 Ap = None,
                 Hp=None,):
 
        self.vertices = vertices
        self.faces = faces
        self.gauss_curv = gauss_curv
        self.mean_curv = mean_curv
        self.Ap = Ap
        self.Hp = Hp
 
        self.n_vert =  vertices.shape[0]  
        self.faces_class_faces = faces_class_faces
        self.faces_class=faces_class
        self.vertex_neighbor=vertex_neighbor  
        
    def Curvature(self):
        X1,X2,X3 = self.vertices[self.faces[:,0]],self.vertices[self.faces[:,1]],self.vertices[self.faces[:,2]]

        E1,E2,E3 = -(X2-X3), -(X3-X1), -(X1-X2) 
        self.E1,self.E2,self.E3 = E1,E2,E3 


        normal = np.cross(X2-X1, X3-X1)

        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        self.faces_area = normal_norm/2 

        unit_t = normal/(normal_norm +eps)
        self.unit_t = unit_t
 

        angle_1 = np.arccos( cos_vect(-E3,E2 ))
        angle_2 = np.arccos( cos_vect(-E3,E1 ))
        angle_3 = np.arccos( cos_vect(-E1,E2 ))

        face_cross_1,face_cross_2,face_cross_3 = np.cross(unit_t, E1),np.cross(unit_t, E2),np.cross(unit_t, E3)

        self.Dp = self.add_vertex_array(angle_1,angle_2,angle_3)
        self.Hp = (-1/2)*self.add_vertex_array(face_cross_1,face_cross_2,face_cross_3)
        self.Ap = (1/6)*self.add_vertex_array(normal_norm,normal_norm,normal_norm)
        self.Up = self.add_vertex_array(unit_t,unit_t,unit_t)
        self.areas = np.sum(self.Ap)



    def Faces_class_faces(self):
        self.faces_class_faces= self.faces_class_faces or self.faces_classification_faces() 
            

 
    def Cluster_class(self,ind_cluster):
        self.Faces_class_faces()
        self.cluster_class=self.classification_cluster(ind_cluster=ind_cluster)
        self.cluster_class_sorted=[self.cluster_class[vv] for vv in np.argsort([len(uu) for uu in self.cluster_class ])[::-1]]
        # self.vertex_neighbor=[ self.faces[list(list(self.Cluster_faces_index(ii)))] for ii in range(self.n_vert)]


    def Vertices_neighbor(self): 
        self.vertex_neighbor=self.vertex_neighbor or [ self.faces[list(list(self.Cluster_faces_index(ii)))] for ii in range(self.n_vert)]


    def Mean_curv(self,eps = 1e-16):
        if self.mean_curv is None:
            if (self.Hp is None) or (self.Ap is None):
                self.Curvature()
            AverageCurv = np.linalg.norm(self.Hp,axis=1, keepdims=True)

            
            # Calculate cosine values
            CosNorm = cos_vect(-self.Hp,self.Up )  
            
            # Get the sign of the curvature 
            AverageCurv = AverageCurv *CosNorm/(np.abs(CosNorm) +eps)
            AverageCurv[np.abs(CosNorm) < eps] = eps
            
            AverageCurv = (1/2)*np.array(AverageCurv)
            self.mean_curv= AverageCurv/(self.Ap+eps)


    def Gauss_curv(self):
        if self.gauss_curv is None:
            if (self.Hp is None) or (self.Ap is None):
                self.Curvature()

            self.gauss_curv= (2*np.pi-self.Dp)/(self.Ap+ eps)
            # return (2*np.pi-self.Dp)/self.Ap

    def Threshold_gauss_mean(self,gauss_thre=None,mean_thre=None):
        if self.gauss_curv is None:
            self.Gauss_curv() 
        gauss_curv=self.gauss_curv
        if gauss_thre is not None:
            gauss_curv[  (gauss_curv)>gauss_thre]=gauss_thre
            gauss_curv[gauss_curv<=-gauss_thre]=-gauss_thre
        self.gauss_curv = gauss_curv

        if self.mean_curv is None:
            self.Mean_curv()
        mean_curv=  self.mean_curv  
        if mean_thre is not None:
            mean_curv[  (mean_curv)>mean_thre]=mean_thre
            mean_curv[mean_curv<=-mean_thre]=-mean_thre 
        self.mean_curv = mean_curv
 


    def Gauss_mean_curv_average(self,
                                gauss_curv=None,
                                mean_curv=None,
                                Ap=None, 
                                numNeighbours=0): 
        if gauss_curv is None:
            self.Gauss_curv() 
            gauss_curv=self.gauss_curv
        if  mean_curv is None:
            self.Mean_curv()
            mean_curv=self.mean_curv
        if Ap is None:
            self.Curvature()
            Ap=self.Ap

        defects= gauss_curv*Ap
        defectsAvg=4*mean_curv*Ap
        area= Ap
        vertices=self.vertices 
        if numNeighbours<0:
            return []

        elif numNeighbours == 0:
            return defects/(area+ eps), defectsAvg/(4*area+ eps)
        else:
            kdtree = KDTree(vertices, leaf_size=2)
            neighbors = kdtree.query(vertices, k=numNeighbours, return_distance=False)
 
            sum_defectAvg = np.sum(np.take(defectsAvg, neighbors), axis=1)
            sum_defect = np.sum(np.take(defects, neighbors), axis=1)
            sum_area = np.sum(np.take(area, neighbors), axis=1)
 
            GaussCurv = sum_defect / (sum_area+eps)
            MeanCurv = sum_defectAvg / (4 * sum_area+eps) 

            return GaussCurv, MeanCurv


    def Gauss_curv_average(self,
                                gauss_curv=None, 
                                Ap=None, 
                                numNeighbours=0): 
        if gauss_curv is None:
            self.Gauss_curv() 
            gauss_curv=self.gauss_curv 
        if Ap is None:
            self.Curvature()
            Ap=self.Ap

        defects= gauss_curv*Ap 
        area= Ap
        vertices=self.vertices 
        if numNeighbours<0:
            return []

        elif numNeighbours == 0:
            return defects/(area +eps)
        else:
            kdtree = KDTree(vertices, leaf_size=2)
            neighbors = kdtree.query(vertices, k=numNeighbours, return_distance=False) 
            sum_defect = np.sum(np.take(defects, neighbors), axis=1)
            sum_area = np.sum(np.take(area, neighbors), axis=1) 
            return   sum_defect / (sum_area+eps) 


    def Mean_curv_average(self, 
                                mean_curv=None,
                                Ap=None, 
                                numNeighbours=0):   
        if  mean_curv is None:
            self.Mean_curv()
            mean_curv=self.mean_curv
        if Ap is None:
            self.Curvature()
            Ap=self.Ap
 
        defectsAvg=4*mean_curv*Ap
        area= Ap
        vertices=self.vertices 
        if numNeighbours<0:
            return []

        elif numNeighbours == 0:
            return defectsAvg/(4*area+eps)
        else:
            kdtree = KDTree(vertices, leaf_size=2)
            neighbors = kdtree.query(vertices, k=numNeighbours, return_distance=False) 
            sum_defectAvg = np.sum(np.take(defectsAvg, neighbors), axis=1) 
            sum_area = np.sum(np.take(area, neighbors), axis=1)
  
            return sum_defectAvg / (4 * sum_area+eps)  


    def force_bend(self,K_bend=1.):
        if (self.Hp is None) and (self.Ap is None):
            self.Curvature()

        E1,E2,E3,U=self.E1,self.E2,self.E3,self.unit_t
        faces = self.faces

        HA1=self.Hp[faces[:,0]]/self.Ap[faces[:,0]]
        HA2=self.Hp[faces[:,1]]/self.Ap[faces[:,1]]
        HA3=self.Hp[faces[:,2]]/self.Ap[faces[:,2]]

        H_avg = (dot_(HA1,HA1)+dot_(HA2,HA2)+dot_(HA3,HA3))/3. 

        C= (np.cross(E1,HA1)+np.cross(E2,HA2)+np.cross(E3,HA3))/(self.faces_area +eps)

        H_avg_UC = H_avg-dot_(U,C)

        F1 = 0.5*H_avg_UC*np.cross(U,E1) + 0.5*np.cross(C,E1) + np.cross(U,-(HA2 - HA3))
        F2 = 0.5*H_avg_UC*np.cross(U,E2) + 0.5*np.cross(C,E2) + np.cross(U,-(HA3 - HA1))
        F3 = 0.5*H_avg_UC*np.cross(U,E3) + 0.5*np.cross(C,E3) + np.cross(U,-(HA1 - HA2))


        return 0.5*K_bend* self.add_vertex_array(F1,F2,F3)


    def energy_bend(self,K_bend=1.):
        if (self.Hp is None) and (self.Ap is None):
            self.Curvature()
        return K_bend*.5*np.sum(dot_(self.Hp,self.Hp)/(self.Ap+eps))
    

    def volume(self,vertices=None,faces=None):
        if vertices is None:
            vertices=self.vertices
        if faces is None:
            faces = self.faces

        centroid =np.mean(vertices, axis=0)
        return np.sum(dot_(vertices[faces[:,0]] - centroid ,
                    np.cross(vertices[faces[:,1]] - centroid,
                                vertices[faces[:,2]] - centroid ))) /6.0
    

    def Unique_vertices(self):
        self.unique_vertices=self.vertices_index_unique =list_vectices_unique(faces=self.faces)
 

    def Plotly_mesh(self,
                    vertices=None,
                    faces=None,
                    intensity=None,
                   colorscale='purd', 
                   showscale=True,
                   width=900,
                   height=700,
                   colorbar=None,):
        if vertices is None:
            vertices=self.vertices
        if faces is None:
            faces=self.faces
        return plotly_mesh(vertices=vertices,
                   faces=faces,
                   intensity=intensity,
                   colorscale=colorscale, 
                   showscale=showscale,
                   width=width,
                   height=height,
                   colorbar=colorbar)


 

    def faces_classification(self,faces=None, n_vert=None): 
        
        if faces is None:
            faces = self.faces
        if n_vert is None:
            n_vert = self.n_vert

        ring_tmp = [[] for _ in range(n_vert)]

        for idx, face in enumerate(faces):
            for vertex in face:
                ring_tmp[vertex].append(idx)


        ring_tmp1 = [[i for i in ring_tmp[k] if faces[i][0] == k] for k in range(n_vert)]
        ring_tmp2 = [[i for i in ring_tmp[k] if faces[i][1] == k] for k in range(n_vert)]
        ring_tmp3 = [[i for i in ring_tmp[k] if faces[i][2] == k] for k in range(n_vert)]

        # Find max length for each subset (0th, 1st, 2nd position) separately
        max_length_0 = max(len(lst) for lst in ring_tmp1)
        max_length_1 = max(len(lst) for lst in ring_tmp2)
        max_length_2 = max(len(lst) for lst in ring_tmp3)

        ring_0 = np.array([pad_list(ring_tmp1[k], max_length_0) for k in range(n_vert)])
        ring_1 = np.array([pad_list(ring_tmp2[k], max_length_1) for k in range(n_vert)])
        ring_2 = np.array([pad_list(ring_tmp3[k], max_length_2) for k in range(n_vert)])
        
        return [ring_0,ring_1,ring_2,ring_0==-1,ring_1==-1,ring_2==-1]
 


    def add_vertex_array(self,V1,V2,V3,faces_class=None,n_vert=None): 
        if n_vert is None:
            n_vert = self.n_vert
        if faces_class is None:
            if self.faces_class is None:
                self.faces_class=self.faces_classification(self.faces,self.n_vert)
            faces_class = self.faces_class


        rgg0,rgg1,rgg2=V1[faces_class[0]],V2[faces_class[1]],V3[faces_class[2]]
        
        rgg0[faces_class[3]]=0  
        rgg1[faces_class[4]]=0  
        rgg2[faces_class[5]]=0   

        return rgg0.sum(axis=1)+rgg1.sum(axis=1)+rgg2.sum(axis=1) 


 
    def faces_classification_faces(self,faces=None,faces_class=None, n_vert=None):  
        if faces is None:
            faces = self.faces
        if n_vert is None:
            n_vert = self.n_vert
        if faces_class is None:
            if self.faces_class is None:
                self.faces_class=self.faces_classification(self.faces,self.n_vert)
            faces_class = self.faces_class
        
        rgg0,rgg1,rgg2=faces[faces_class[0]],faces[faces_class[1]],faces[faces_class[2]]
        rgg0[faces_class[3]]=0
        rgg1[faces_class[4]]=0
        rgg2[faces_class[5]]=0

        Judge=[]
        for i in range(n_vert):
            JudgeTriV=np.concatenate((rgg0[i:i+1], rgg1[i:i+1], rgg2[i:i+1]), axis=1)[0]
            mask = ~np.all(JudgeTriV == -1, axis=1).flatten()
            Judge.append(set(JudgeTriV[mask, :].flatten()))
        return Judge 



    def Cluster_faces_index(self, cll,faces_class=None):   
        if faces_class is None:
            if self.faces_class is None:
                self.faces_class=self.faces_classification(self.faces,self.n_vert)
            faces_class = self.faces_class
        return list(set(np.concatenate([faces_class[i][cll][faces_class[i][cll] != -1] for i in range(3)])))

  

    def classification_cluster(self, ind_cluster=None, faces_class_faces=None):
        if faces_class_faces is None:
            faces_class_faces = self.faces_class_faces

        ind_cluster_copy = ind_cluster.copy()
        clu = []  
        i = 0
        iui=len(ind_cluster_copy)

        while (len(ind_cluster_copy)> 0) and (iui>0):
            clu.append([])  
            k = ind_cluster_copy[0]
            Ju = set(faces_class_faces[k])
          
            while Ju:
                n_elm = set()  
                matched_indices = list(set(ind_cluster_copy).intersection(Ju))
 
                if len(matched_indices) == 0:
                    break
 
                clu[i].extend(matched_indices )  
 
                ind_cluster_copy =  list(set(ind_cluster_copy)-set(matched_indices)) 

                for jj in matched_indices :
                    n_elm.update(faces_class_faces[jj])
 
                if n_elm == Ju:
                    break

                Ju = n_elm.difference(clu[i])

            i += 1
            iui-=1
            # print(iui)
        return clu

 
class part:
    def __init__(self,faces,face_class) -> None:
        self.vertices = None 
        self.forces =  None
        self.energy = None
        self.p = None 
        self.faces = faces
        self.faces_class = face_class
        self.volume = None
        self.areas = None

    
    def zeros(self, shape):
        self.vertices = np.zeros((shape,3))
        self.p = np.zeros((shape,3))
        self.forces = np.zeros((shape,3))
        self.energy = 0.
        self.volume=0.
        self.areas=0.


    def initiation(self,mg,K_bend=1.):
        self.vertices =mg.vertices
        self.forces = mg.force_bend(K_bend=K_bend) 
        self.energy = np.zeros((mg.n_vert,1))
        self.p = self.vertices#+ self.forces * dt
        self.Ap=mg.Ap
        self.areas=mg.areas


    def update_features(self,vertices=None,K_bend=1.):
        if vertices is None:
            vertices = self.vertices
        mesh = curv_mesh(vertices=vertices,
                         faces=self.faces, 
                         faces_class=self.faces_class ) 
        self.forces = mesh.force_bend(K_bend=K_bend)# -.01*vertices# 
        self.energy = mesh.energy_bend(K_bend=K_bend)
        self.volume = mesh.volume(vertices=vertices)
        self.areas= mesh.areas

    def update_force(self,vertices=None,K_bend=1.):
        if vertices is None:
            vertices = self.vertices
        mesh = curv_mesh(vertices=vertices,
                         faces=self.faces, 
                         faces_class=self.faces_class ) 
        self.forces = mesh.force_bend(K_bend=K_bend)# -.01*vertices# 


    def update_energy(self,vertices=None,K_bend=1.):
        if vertices is None:
            vertices = self.vertices
        mesh = curv_mesh(vertices=vertices,
                         faces=self.faces, 
                         faces_class=self.faces_class ) 
        self.energy = mesh.energy_bend(K_bend=K_bend)



class saves:
    def __init__(self,n_step,n_error,dt=1) -> None:
        self.dt = dt
        self.n_step = n_step
        self.n_error = n_error
        self.error = np.zeros((n_step,n_error-1))     
        self.energy = np.zeros((n_step,n_error))    
        self.volume = np.zeros((n_step,n_error))    
        self.areas = np.zeros((n_step,n_error))

    def graph_analysis_energy_volume(self, 
                                        width=1000, 
                                        height=400, 
                                        color_template='plotly', 
                                        paper_bgcolor=None, 
                                        color_font=None, 
                                        size_font=20, 
                                        title=' '):  
        
        return plotly_analysis(energy=self.energy,
                               volume=self.volume,
                               dt=self.dt,
                                width=width, 
                                height=height, 
                                color_template=color_template, 
                                paper_bgcolor=paper_bgcolor, 
                                color_font=color_font, 
                                size_font=size_font, 
                                title=title)
    
    def graph_convergence_rate(self, 
                                        width=600, 
                                        height=400, 
                                        color_template='plotly', 
                                        paper_bgcolor=None, 
                                        color_font=None, 
                                        size_font=20, 
                                        title=' '):  
        
        return plotly_convergence_rate(
                               error=self.error,
                               dt=self.dt,
                                width=width, 
                                height=height, 
                                color_template=color_template, 
                                paper_bgcolor=paper_bgcolor, 
                                color_font=color_font, 
                                size_font=size_font, 
                                title=title)






class euler:
    def run(self,X,dt):  
        vertices = X.vertices + dt * X.forces 

        X.vertices= vertices 
        X.update_features()   

 


class simulation:
    def __init__(self,
                 vertices=None,
                 faces=None,
                 mmg=None,
                 Xa=None,
                 dt=1e-3,
                 K_bend=1.,
                 n_step=10,
                 n_error=None,
                 save_file=None,
                 disp_time=None,
                 integrator=None,
                smooth_path=None, 
                txt_vertices=None,):
        n_error= max(n_error,1)
        self.dt = dt
        self.K_bend = K_bend
        self.n_step = n_step
        self.n_error = n_error
        Xsave = saves(n_step=n_step,n_error=n_error,dt=dt)
        if mmg is None:
            mmg = curv_mesh(vertices=vertices,faces=faces)  
        if Xa is None: 
            Xa = [part(faces=mmg.faces,face_class=mmg.faces_class) for _ in range(n_error)]
            for i in range(n_error):
                Xa[i].initiation(mmg) 

        if disp_time is None:
            disp_time = int(n_step//10)
        else:
            disp_time = min(max(1,int(n_step//disp_time)),n_step)
        
        if integrator is None:
            integrator = euler()

        if save_file is None:
            save_file = f'file_{n_step}\\'
            os.makedirs(save_file, exist_ok=True)


        voll=0
        if (n_error>=3)  :
            for i in  tqdm(range(n_step), desc=f"volume: {voll:.6f}") : 
                energy_tmp = []
                volume_tmp = []
                error_tmp = []
                area_tmp = []
                for j in range(n_error):
                    integrator.run(Xa[j],dt=2**j*dt)
                    energy_tmp.append(Xa[j].energy)   
                    volume_tmp.append(Xa[j].volume)    
                    area_tmp.append(Xa[j].areas)
                voll=volume_tmp[0]
                for j in range(n_error-1):
                    error_tmp.append(norm_weighted(Xa[j+1].vertices-Xa[j].vertices,Xa[j+1].Ap,ord=2))
                Xsave.energy[i,:]=energy_tmp
                Xsave.volume[i,:]=volume_tmp
                Xsave.error[i,:]= error_tmp
                Xsave.areas[i,:]= area_tmp

                if (i%disp_time)==0:
                    for k in range(n_error):
                        np.savetxt(f'{save_file}vertices_{k}.txt', Xa[k].vertices, fmt='%f')  
                    np.savetxt(f'{save_file}volume.txt', Xsave.volume[:i,:], fmt='%f')  
                    np.savetxt(f'{save_file}energy.txt', Xsave.energy[:i,:], fmt='%f')  
                    np.savetxt(f'{save_file}error.txt', Xsave.error[:i,:], fmt='%f')   
                    np.savetxt(f'{save_file}areas.txt', Xsave.areas[:i,:], fmt='%f')
                    # print(f'iteration = {i}')

        else:

            pbar=tqdm(range(n_step), desc=f"volume: {voll:.6f}")
            for i in  pbar : 
                energy_tmp = []
                volume_tmp = [] 
                area_tmp = []
                for j in range(n_error):
                    integrator.run(Xa[j],dt= dt)
                    energy_tmp.append(Xa[j].energy)   
                    volume_tmp.append(Xa[j].volume)    
                    area_tmp.append(Xa[j].areas) 
                Xsave.energy[i,:]=Xa[j].energy
                Xsave.volume[i,:]=Xa[j].volume 
                Xsave.areas[i,:]= Xa[j].areas
                if (volume_tmp[0]<0) or (area_tmp[0]<0):
                    return
                pbar.set_description(f"Volume: {volume_tmp[0]:.3f} | Area: {area_tmp[0]:.3f} Energy: {energy_tmp[0]:.3f}")
                voll=volume_tmp[0]
                if (i%disp_time)==0:
                    for k in range(n_error):
                        np.savetxt(os.path.join(save_file,f'vertices_{k}.txt'), Xa[k].vertices, fmt='%f')  
                    mesh=trimesh.Trimesh(vertices=Xa[k].vertices ,faces=faces) 
                    mesh.export(smooth_path ) 
                    np.savetxt(os.path.join(save_file,f'volume.txt'), Xsave.volume[:i,:], fmt='%f')  
                    np.savetxt(os.path.join(save_file,f'energy.txt'), Xsave.energy[:i,:], fmt='%f')   
                    np.savetxt(os.path.join(save_file,f'areas.txt'), Xsave.areas[:i,:], fmt='%f')



        self.Xa = Xa
        self.Xsave = Xsave
        self.mmg = curv_mesh(vertices=vertices,faces=faces) 

   








def cos_vect(V1,V2,eps = 1e-9):  return   np.sum(V1*V2, axis=1, keepdims=True)/(np.linalg.norm(V1, axis=1, keepdims=True)*np.linalg.norm(V2, axis=1, keepdims=True)+eps)





def dot_(V1,V2,axis=1): return np.sum(V1*V2, axis=axis, keepdims=True)


 