from dend_fun_0.curvature import curv_mesh as curv_mesh 
from dend_fun_0.help_funn import dendrite,  cluster_class,label_cluster,Branch_division,get_intensity,Threshold_curv,Impute_intensity,remove_file,loadtxt,loadtxt_count
from dend_fun_0.help_funn import mappings_vertices,clust_pca,dendrite_io,get_color,Refine_vertices_index,volume,closest_distances_group,find_min_max_no_cross,clust_segment
import os
import numpy as np 
from dend_fun_2.metric import center_curvature
clor=get_color()
from sklearn.decomposition import PCA
import dend_fun_0.help_fun as hf

 


from scipy.spatial import KDTree

from sklearn.cluster import DBSCAN
def fit_quadratic_curve(points): 
    t = np.linspace(0, 1, len(points))
    cof= 1 if len(points) < 3 else 2 
    cx = np.polyfit(t, points[:,0], cof)
    cy = np.polyfit(t, points[:,1], cof)
    cz = np.polyfit(t, points[:,2], cof)
    
    return cx, cy, cz

def eval_curve(t, coeffs):
    cx, cy, cz = coeffs
    x = np.polyval(cx, t)
    y = np.polyval(cy, t)
    z = np.polyval(cz, t)
    return np.column_stack([x,y,z])

def residuals(points, coeffs):
    t = np.linspace(0, 1, len(points))
    curve_points = eval_curve(t, coeffs)
    return np.linalg.norm(points - curve_points, axis=1)

import numpy as np

def fit_line(points): 
    centroid = np.mean(points, axis=0)
    uu, ss, vv = np.linalg.svd(points - centroid)
    direction = vv[0]  # first principal component
    return centroid, direction / np.linalg.norm(direction)

def residuals_line(points, centroid, direction): 
    diffs = points - centroid
    cross = np.cross(diffs, direction)
    dists = np.linalg.norm(cross, axis=1)
    return dists

def fit_best(points,degree=1): 
    if degree==1:
        centroid, direction = fit_line(points) 
        return residuals_line(points, centroid, direction)

 

import numpy as np 
from scipy.interpolate import splprep, splev

class CurveFitter:
    def __init__(self, cluster1, cluster2, 
                 mode="poly", degree=2, 
                 use_arclength=True, 
                 line_num_points=50, spline_smooth=0): 
        self.cluster1 = np.asarray(cluster1)
        self.cluster2 = np.asarray(cluster2)
        self.mode = mode
        self.degree = degree
        self.use_arclength = use_arclength
        self.line_num_points = line_num_points
        self.spline_smooth = spline_smooth
 
        self.model1 = self._fit_curve(self.cluster1)
        self.model2 = self._fit_curve(self.cluster2)
 
    def _parameterize_by_arclength(self, points):
        diffs = np.diff(points, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        return s / s[-1] if s[-1] > 0 else np.linspace(0,1,len(points))

    def _fit_curve(self, points):
        if self.mode == "poly":
            if self.use_arclength and len(points) >= 2:
                t = self._parameterize_by_arclength(points)
            else:
                t = np.linspace(0, 1, len(points))
            deg = min(self.degree, len(points)-1)
            cx = np.polyfit(t, points[:,0], deg)
            cy = np.polyfit(t, points[:,1], deg)
            cz = np.polyfit(t, points[:,2], deg)
            return (cx, cy, cz)
        elif self.mode == "spline":
            pts = points.T
            k = min(3, len(points)-1)  # spline degree
            tck, u = splprep([pts[0], pts[1], pts[2]], 
                             s=self.spline_smooth, k=k)
            return (tck, u)
        else:
            raise ValueError("mode must be 'poly' or 'spline'")

    def _eval_curve(self, points, model):
        if self.mode == "poly":
            if self.use_arclength and len(points) >= 2:
                t = self._parameterize_by_arclength(points)
            else:
                t = np.linspace(0, 1, len(points))
            cx, cy, cz = model
            x = np.polyval(cx, t)
            y = np.polyval(cy, t)
            z = np.polyval(cz, t)
            return np.column_stack([x, y, z])
        elif self.mode == "spline":
            tck, u = model
            u_fine = np.linspace(0, 1, len(points))
            x, y, z = splev(u_fine, tck)
            return np.column_stack([x, y, z])

    def _residuals(self, points, model):
        curve_pts = self._eval_curve(points, model)
        return np.linalg.norm(points - curve_pts, axis=1)
 
    def self_residuals(self):
        r1 = self._residuals(self.cluster1, self.model1)
        r2 = self._residuals(self.cluster2, self.model2)
        return r1, r2

    def cross_residuals(self):
        r1_to_2 = self._residuals(self.cluster1, self.model2)
        r2_to_1 = self._residuals(self.cluster2, self.model1)
        return r1_to_2, r2_to_1
 


class mapping_skl():
    def __init__(self,vertices,skeleton_points ):
        self.vertices=vertices 
        vec=set([tuple(np.round(fb, 7))  for fb in skeleton_points])
        vecs=skeleton_points
        ghf=[]
        self.mappk={} 
        lenb,lena= 1,set({})
        while (lenb>0):
            tree = KDTree(vecs) 
            _,  skl_index = tree.query( vertices)  
            for fb,vf in zip(skl_index,vertices):
                fbv = tuple(np.round(skeleton_points[fb], 7)) 
                # fbv=tuple(skeleton_points[fb])

                if fbv not in self.mappk:
                    self.mappk[fbv]=[]
                self.mappk[fbv].append(vf)
                ghf.append(fbv)
            lenc=set( self.mappk.keys()) 
            gghf=vec-lenc
            lenb=len( lenc- lena)
            lena=lenc
            vecs=np.array([list(gf) for gf in gghf]) 
        self.remain=set([tuple(nn) for nn in self.vertices])-set(self.mappk.keys()) 

        maprem,mapkey=list(self.remain),list(self.mappk.keys())
        ktre=KDTree(np.array(mapkey))
        indx=ktre.query(np.array(maprem))[1]
        uuiuu=[mapkey[ui] for ui in indx]
        for fg,fbv in zip(uuiuu,maprem):
            if fbv not in self.mappk:
                self.mappk[fbv]=[]
            self.mappk[fbv].extend(self.mappk[fg])



    def mapping_inv(self, skeleton_points):
        vertices_mapped=[]
        for fb in skeleton_points:
            fbv = tuple(np.round(fb, 7))  
            vertices_mapped.extend(self.mappk[fbv]) 
       
        return  np.array(vertices_mapped) 



class mapping_skl():
    def __init__(self,vertices,skeleton_points,skl_index ):
        self.vertices=vertices 
        # tree = KDTree( skeleton_points) 
        # _,  skl_index = tree.query(vertices)  
        self.mappk={} 
        for fb,vf in zip(skl_index,vertices): 
            fbv=tuple(skeleton_points[fb])
            if fbv not in self.mappk:
                self.mappk[fbv]=[]
            self.mappk[fbv].append(vf)
  

    def mapping_inv(self, skeleton_points):
        vertices_mapped = []
        for fb in skeleton_points:
            fbv = tuple(fb)
            values = self.mappk.get(fbv)
            if values is not None:
                vertices_mapped.extend(values)
            else:
                print(f"Missing mapping for {fbv}")
        return np.array(vertices_mapped)



def get_pca_analysis(points): 
    centered = points - np.mean(points, axis=0) 
 
    pca = PCA(n_components=3)
    pca.fit(centered)
 
    pc1 = pca.components_[0]
 
    projections = centered @ pc1
 
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    return points[[min_idx, max_idx]]






class region_branch:
    def __init__(self,region_index,dend,skl_vectices,zoom_thre=100,zoom_thre_min=5,zoom_thre_max=25,):
        pass
        self.region_index,self.dend,self.skl_vectices=region_index,dend,skl_vectices
        self.intensity_tickness=np.zeros(dend.vertices.shape[0]) 
        self.zoom_thre,self.zoom_thre_min,self.zoom_thre_max=zoom_thre,zoom_thre_min,zoom_thre_max
    def get_region_branch(self,size_threshold= 3,):
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        tree = KDTree(self.skl_vectices) 
        cclu=cluster_class(faces_neighbor_index=self.dend.vertex_neighbor) 
        self.mapp={}
        for ii in set(self.region_index): 
            indexx=np.where(self.region_index==ii)[0]
            nodee=Branch_division(
                            cclu=cclu,
                            dend= self.dend, 
                            vertices_index=indexx,
                            size_threshold= size_threshold,
                            tf_cluster_faces_unique=False,)
            for nod in nodee.children:
                indxx=nod.vertices_index
                facess=nod.faces 
                dist,indx=tree.query(vertices[indxx] )  
                self.mapp[tuple(indxx)]=dict(faces=facess, ) 
    
    

    def get_skl_smooth(self,
                        subsample_thre=.02,
                        f=0.99,
                        N=10,
                        num_chunks=100, 
                        num_points=50,
                        ctl_run_thre=1,
                            line_num_points= None,
                            line_num_points_inter= None,
                            spline_smooth= None,):
        if not hasattr(self, "mapp") or not self.mapp:
            self.get_region_branch() 
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        mmap_keys=list(self.mapp.keys()) 
        tree = KDTree(skl_vectices)    
        skl_vertices_index =[]  
        self.intensity_thickness=np.zeros(vertices.shape[0])
        for liss in mmap_keys:
            lis=list(liss)
            if len(lis)<3:
                continue
            dist,indx=tree.query(vertices[lis] )   
            ctl_tmp = center_curvature(vertices=skl_vectices[indx], 
                    line_num_points= line_num_points,
                    line_num_points_inter= line_num_points_inter,
                    spline_smooth= spline_smooth,
                )
            skl_vertices_index.append(ctl_tmp.vertices_center)  
        self.skl_vertices_index=np.vstack(skl_vertices_index)
        tree = KDTree(self.skl_vertices_index)   
        self.intensity_thickness=tree.query(vertices)[0]



    def get_inverse_mapping(self, index): 
        if not hasattr(self, "mapp") or not self.mapp:
            self.get_region_branch() 
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        set_ind=set(index)
        set_ind_copy=set_ind.copy() 
        indexx=[] 
        set_ind_copy_tmp=set()
        for lis in self.mapp:
            inter= set_ind.intersection(lis)
            if len(inter)>3:
                set_ind_copy_tmp.update(inter) 
                indexx.extend(lis) 
            if len(set_ind_copy-set_ind_copy_tmp )<4:
                break
        cclu=cluster_class(faces_neighbor_index=self.dend.vertex_neighbor) 
        node =Branch_division(
                        cclu= cclu,
                        dend=self.dend, 
                        vertices_index=indexx,
                        size_threshold= 2, 
                        stop_index=1,
                        ) 
        return node





    def get_skl_region(self,skl_shaft_vectices,shaft_index,spine_index,zoom_thre=None,zoom_thre_min=None,zoom_thre_max=None,):
        if not hasattr(self, "mapp") or not self.mapp:
            self.get_region_branch()
        zoom_thre     = zoom_thre     or self.zoom_thre
        zoom_thre_min = zoom_thre_min or self.zoom_thre_min
        zoom_thre_max = zoom_thre_max or self.zoom_thre_max

        get_neck_index=self.get_neck_region(skl_shaft_vectices,shaft_index,spine_index,
                                            zoom_thre=zoom_thre,
                                            zoom_thre_min=zoom_thre_min,
                                            zoom_thre_max=zoom_thre_max,) 
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        mmap_keys=list(self.mapp.keys())
        set_ind=set(spine_index)
        set_ind_copy=set_ind.copy()
        set_ind_copy_tmp=set()
        skl_index_loc=set() 
        skl_distance_loc=set() 
        tree = KDTree(skl_vectices) 
        indall= tree.query(vertices[list(set_ind)] )[1]
        indall_neck=tree.query(vertices[get_neck_index] )[1]
        skl_spine_index =[]
        skl_neck_index =[]
        scatter=[]
        gcol=get_color()
        ii=0
        jj=0
        for lis in mmap_keys:
            inter= set_ind.intersection(lis)
            set_ind_copy_tmp.update(inter)
            dist,indx=tree.query(vertices[list(inter)] )  
            if len(inter)>3:
                self.intensity_tickness[list(lis)]=np.mean(dist)
                
                skl_index_loc.update(set(indx).intersection(indall)) 
                nehh=list(set(lis).intersection(set(get_neck_index)))
                if len(nehh)>0:
                    indall_neck=list(set(tree.query(vertices[nehh] )[1]))
                    skl_neck_index.append(indall_neck)
                    skl_spine_index.append(list(set(indx).intersection(indall)-set(indx).intersection(set(indall_neck))))
                    iic =min(jj,len(gcol)-1)  
                    jj+=1
                    scatter.append(hf.plotly_scatter(points=skl_vectices[indall_neck] , color=gcol[iic], size=6.95 ,name='neck'))
                    scatter.append(hf.plotly_scatter(points=skl_vectices[list(set(indx).intersection(indall)-set(indx).intersection(set(indall_neck)))] , color=gcol[iic], size=3.95 ))
                    continue  
                skl_spine_index.append(list(set(indx).intersection(indall)))
                iic =min(ii,len(gcol)-1)  
                ii+=1
                scatter.append(hf.plotly_scatter(points=skl_vectices[list(set(indx).intersection(indall))] , color=gcol[iic], size=3.95 )) 
            if len(set_ind_copy-set_ind_copy_tmp )<4:
                break
        return skl_spine_index,skl_neck_index,scatter

 

    def get_skl_neck_region(self,skl_shaft_vectices,shaft_index,spine_index,zoom_thre=None,zoom_thre_min=None,zoom_thre_max=None,):
        if not hasattr(self, "mapp") or not self.mapp:
            self.get_region_branch()
        zoom_thre     = zoom_thre     or self.zoom_thre
        zoom_thre_min = zoom_thre_min or self.zoom_thre_min
        zoom_thre_max = zoom_thre_max or self.zoom_thre_max
        get_neck_index=self.get_neck_region(skl_shaft_vectices,shaft_index,spine_index,
                                            zoom_thre=zoom_thre,
                                            zoom_thre_min=zoom_thre_min,
                                            zoom_thre_max=zoom_thre_max,)
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        mmap_keys=list(self.mapp.keys()) 
        set_ind=set(spine_index)
        set_ind_copy=set(get_neck_index)
        set_ind_copy_tmp=set() 

        tree = KDTree(skl_vectices) 
        indall= tree.query(vertices[list(set_ind)] )[1]
        indall_neck=tree.query(vertices[get_neck_index] )[1]  
        skl_neck_index =[]  
        for lis in mmap_keys:
            inter= set_ind.intersection(lis)
            dist,indx=tree.query(vertices[list(inter)] )  
            if len(inter)>3:   
                nehh=set(lis).intersection(set(get_neck_index))
                set_ind_copy_tmp.update(nehh)
                if len(nehh)>0:
                    indall_neck=tree.query(vertices[list(nehh)] )[1]
                    skl_neck_index.extend(indall_neck)
            if len(set_ind_copy-set_ind_copy_tmp )<4:
                break
        return list(set(skl_neck_index) )


 

    def get_neck_region(self,skl_shaft_vectices,shaft_index,spine_index,zoom_thre=None,zoom_thre_min=None,zoom_thre_max=None,size_threshold=3):
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        mapp=mappings_vertices(vertices_0=vertices)  
        vcv_length= KDTree(skl_shaft_vectices).query(vertices)[0] 
        clu_pca=clust_pca(vertices,shaft_index,
                        vertices_center=  skl_shaft_vectices,
                        vcv_length=vcv_length)   

        cls=clust_segment(vertices_0=self.dend.vertices ,
                        vertices_index=spine_index,
                        vertices_index_shaft=shaft_index,
                        clu_pca= clu_pca, 
                        size_threshold=size_threshold, 
                        )   
        llin=np.linspace(0,1,zoom_thre) 
        spine_index_cls_10=mapp.Mapping_inverse(cls.Cyl_threshold(zoom_threshold= llin[min(zoom_thre_min,zoom_thre)]) )
        spine_index_cls_30=mapp.Mapping_inverse(cls.Cyl_threshold(zoom_threshold= llin[min(zoom_thre_max,zoom_thre)]) )
        return list(set(spine_index_cls_10)-set(spine_index_cls_30))
    



class skl_spine_segment:
    def __init__(self,vertices,faces, skl_vertices,skl_shaft_vectices,vertices_center,pid,skl_index):
        self.pid=pid   
        self.dend = pid.dend
        self.vertices=vertices
        self.skl_vertices=skl_vertices
        self.skl_shaft_vectices=skl_shaft_vectices
        self.mapp=mappings_vertices(vertices_0=self.vertices) 
        self.vertices_center=vertices_center
        self.mapp_skl=mapping_skl( vertices=self.vertices,skeleton_points=self.skl_vertices,skl_index=skl_index, )

    def get_segments(self,shaft_path,
                     thre_distance_min=1.75,
                     thre_distance_max=1.75,
                     thre_explained_variance_ratio=0.97,
                     size_threshold=100,
                     tf_get_scatter=False,
                    intensity=None,
                    spine_count=None,
                    tf_merge=False):
 
        if spine_count is None:
            spine_count=np.loadtxt(os.path.join(shaft_path, f'spine_count.txt'),dtype=int,ndmin=1)
        if len(spine_count)>0:
        # 
            data={ii:{'data':[],'scatter':[],'index':[],'faces':[],'segment_cclu':{}} for ii in spine_count}
            if intensity is None:
                intensity=-20*np.ones_like(self.vertices[:,0])
            io=0
            node=dendrite() 
            for ip in spine_count: 
                data[ip]['index']= np.loadtxt(os.path.join(shaft_path, f'{self.pid.name_spine_index}_{ip}.txt'),dtype=int)
                data[ip]['faces']= np.loadtxt(os.path.join(shaft_path, f'{self.pid.name_spine_faces}_{ip}.txt'),dtype=int)
                data[ip]['spine_center_curv']=np.loadtxt(os.path.join(shaft_path, f'spine_center_curv_{ip}.txt'),dtype=float,ndmin=2)  
                data_sp=data[ip]
                if len(data[ip]['spine_center_curv'])<2:
                    continue
                data_sp,intensity,io=self.get_segment(data_sp=data_sp,  
                            thre_distance_min=thre_distance_min, 
                            thre_distance_max=thre_distance_max,
                            thre_explained_variance_ratio=thre_explained_variance_ratio,
                            size_threshold=size_threshold,
                            intensity=intensity,
                            tf_get_scatter=tf_get_scatter,
                            io=io,
                            tf_merge=tf_merge,) 
                data[ip]=data_sp 
                if data_sp  is not None:
                    for ccluo in data_sp['segment_cclu'].values():
                        if len(ccluo.cluster_index)>2:
                            node.add_child(vertices_index=ccluo.cluster_index,
                                            faces=ccluo.cluster_faces, 
                                            vertices_index_unique =ccluo.cluster_faces_unique ,
                                            )
            return data,node,intensity
    


    def get_segment(self,
                    data_sp, 
                    io=0,  
                    thre_distance_min=1.75,
                    thre_distance_max=1,
                    thre_explained_variance_ratio=0.97,
                    size_threshold=100,
                    intensity=None,
                    num_chunks=100,
                    tf_get_scatter=False,
                    tf_merge=False,):
        vertices_center=self.vertices_center
        X=data_sp['spine_center_curv']
        cclu=cluster_class(faces_neighbor_index= self.dend.vertex_neighbor)
        # print('[[[]]]',X)
        tree = KDTree(X) 
        sav=[]
        for i in range(len(X)):
            dist, idx = tree.query(X[i], k=3)  
            sav.append(dist[1])

        ff=sav[np.argsort(sav)[-2]]+1e-6
        db = DBSCAN(eps=ff*1., min_samples=2)  
        labels = db.fit_predict(X)
        unique_labels = set(labels)
        dat={}
        for ii,label in enumerate(unique_labels):
            if label<0:
                continue 
            dat[label]={nm:[] for nm in ['shaft_close','spine_close','spine_far','spine_extremities','distance_close','cluster','color','scatter']} 
            dat[label]['cluster']=vertices_ii=X[labels == label]
            clo_min,gf= closest_distances_group(vertices_ii, vertices_center,num_chunks=num_chunks)
            dat[label]['spine_close']=vertices_ii[np.argmin(clo_min)]
            dat[label]['shaft_close']=vertices_center[gf[np.argmin(clo_min)]].reshape(1,-1) 
            dat[label]['spine_extremities'] = points=dat[label]['cluster'] if len(dat[label]['cluster'])<=2 else get_pca_analysis(dat[label]['cluster'])
            dst=[np.linalg.norm(dat[label]['spine_close']-va) for va in points]
            dat[label]['spine_far'] = points[np.argmax(dst)] 
            dat[label]['distance_close']= np.linalg.norm(dat[label]['spine_close']-dat[label]['shaft_close'])
            dat[label]['distance_far']  = np.linalg.norm(dat[label]['spine_far']  -dat[label]['shaft_close'])
            dat[label]['color'] = label  if label<len(clor) else len(clor)-1
        labs=list(dat.keys())
        xval=[dat[label]['distance_close'] for label in labs] 
        yval=[dat[label]['distance_far'] for label in labs]
        distance_min=min(xval)*thre_distance_min
        distance_max=min(xval)*thre_distance_max 
        if len(xval)==0:
            return 
 
        label_min=[labs[nn] for nn in np.where((xval<=distance_min)&(yval>distance_max))[0]]
        if len(label_min)==0: 
            label_min=[labs[np.argmin(xval)]]  



        mapo={}
        label_min_tmp=label_min.copy()
        for il,lav in enumerate(label_min_tmp):
            if lav not in mapo:
                mapo[lav]=[]
            remo=[]
            ktt=KDTree(dat[lav]['cluster'])
            for ilb,lavb in enumerate(label_min_tmp[il+1:]):
                vec=np.vstack((dat[lav]['cluster'],dat[lavb]['cluster'])) 
   



                pca = PCA(n_components=3)
                pca.fit(vec)     
                points=dat[lavb]['spine_extremities'] #  dat[lavb]['cluster'] if len(dat[lavb]['cluster'])<=2 else get_pca_analysis(dat[lavb]['cluster'])  
                dss=min(ktt.query(points)[0]) 
                if dss<ff*2.:
                    mapo[lav].append(lavb)
                    remo.append(lavb) 

                elif pca.explained_variance_ratio_[0] > thre_explained_variance_ratio: 
                    mapo[lav].append(lavb)
                    remo.append(lavb) 
        mapoo={}
        bggb=[]
        for ky,mmv in mapo.items():
            for mn in mmv:
                mapo[ky].extend(mapo[mn])
            mapo[ky]=list(set(mapo[ky]))
            if ky not in bggb:
                mapoo[ky]= list(set(mapo[ky]))
            bggb.extend(mapo[ky]) 

        mapo={}
        bggb=[]
        maj=list(mapoo.keys()) 
        if len(maj)>1:
            for iu,ky in enumerate(maj):
                for kyy in maj[iu+1:]:
                    if ky not in bggb:
                        mapo[ky]=[ky]
                    if len(set(mapoo[ky]).intersection(set(mapoo[kyy])))>0:
                        if ky not in mapo[ky]:
                            mapo[ky].extend(list(set(mapoo[ky])|set(mapoo[kyy])))
                            mapo[ky].extend([kyy]) 
                            bggb.extend(mapo[ky])
                    else:
                        if kyy not in bggb:
                            mapo[kyy]=[kyy] 
                            bggb.extend(mapo[kyy])
        else:
            mapo=mapoo  
        label_min=[]
        for iu,va in mapo.items():
            for iuu in va:
                dat[iu]['cluster']=np.vstack((dat[iu]['cluster'],dat[iuu]['cluster']))
                dat[iuu]['color']=dat[iu]['color']
            label_min.append(iu)
                     

        ktrees=[KDTree(dat[label]['cluster']) for label in label_min]

        cll={
            ii:{nm:[] for nm in [ 'cluster','color','scatter','label','index']} 
            for ii in label_min
            }
        for key in label_min:
            cll[key]['cluster']=dat[key]['cluster'].tolist()
            cll[key]['color']  =dat[key]['color']

        label_max=list(set(dat.keys())-set(label_min))
        for ii in label_max:
            dtmp = 1e10
            # dtmp=0.0
            for jj, ktree in zip(label_min, ktrees): 
                mind, idx = ktree.query(dat[ii]['spine_close'])

                vec=np.vstack((dat[ii]['cluster'],dat[jj]['cluster'])) 
                fitter = CurveFitter(dat[ii]['cluster'],dat[jj]['cluster'],  mode="spline", spline_smooth=1)
                r1_self, r2_self = fitter.self_residuals()
                r1_to_2, r2_to_1 = fitter.cross_residuals() 
                mind=np.mean(r1_to_2)+ np.mean(r2_to_1)

                if mind < dtmp:
                    dtmp = mind
                    idx_out=idx
                    jj_out=jj 
 
            cll[jj_out]['cluster'].extend(dat[ii]['cluster'].tolist())
            dat[ii]['color'] = dat[jj_out]['color']
         

        for key in label_min:
            cll[key]['cluster']=np.array(cll[key]['cluster'])

        data_sp['data']=dat
        if tf_get_scatter:
            for label in dat:
                data_sp['scatter'].append(hf.plotly_scatter(points=dat[label]['cluster'] , color=clor[dat[label]['color']], size=3.9, name=f'all {label}'))

            for label in label_min:
                data_sp['scatter'].append(hf.plotly_scatter(points=cll[label]['cluster'] , color=clor[cll[label]['color']], size=3.9, name=f'center {label}'))

 
        index_all=[] 
        for hh in label_min: 
            vert=self.mapp_skl.mapping_inv(cll[hh]['cluster'])
            index=self.mapp.Mapping_inverse(vert).tolist()
            if not tf_merge:
                index= list(set(data_sp['index']).intersection(set(index)))
            cll[hh]['index']=index
            index_all.extend(index)
 
        vertices_00=self.vertices
        index_all_rem= list(set(data_sp['index'])-set(index_all))
        ktrees=[KDTree(vertices_00[cll[label]['index']]) for label in label_min]
        iki=[ktree.query(vertices_00[index_all_rem]) for ktree in ktrees ]
        ikki=np.argmin(np.array([iki[idx][0] for idx in range(len(label_min))]),axis=0)
        ikkki=np.array([iki[idx][1] for idx in range(len(label_min))]).T

 
        result = ikkki[np.arange(len(ikki)), ikki]
        for hh,vh in enumerate(label_min): 
            cll[vh]['index'].extend([index_all_rem[mm] for mm in np.where(ikki==hh)[0].tolist()])
        for hh in label_min:
            cclu=cluster_class(faces_neighbor_index= self.dend.vertex_neighbor)
            index=cll[hh]['index']
            if len(index)>size_threshold:
                cclu.Cluster_index(ln_elm= index )
                cclu.Cluster_faces()
                cclu.Cluster_faces_unique()
                if len(cclu.cluster_index)>0: 
                    intensity[index]=io
                    io+=1 
                    data_sp['segment_cclu'][io]=cclu

        return data_sp,intensity,io


class region_branch:
    def __init__(self,region_index,dend,skl_vectices,zoom_thre=100,zoom_thre_min=5,zoom_thre_max=25,):
        pass
        self.region_index,self.dend,self.skl_vectices=region_index,dend,skl_vectices
        self.intensity_tickness=np.zeros(dend.vertices.shape[0]) 
        self.zoom_thre,self.zoom_thre_min,self.zoom_thre_max=zoom_thre,zoom_thre_min,zoom_thre_max
    def get_region_branch(self,size_threshold= 3,):
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        tree = KDTree(self.skl_vectices) 
        cclu=cluster_class(faces_neighbor_index=self.dend.vertex_neighbor) 
        self.mapp={}
        for ii in set(self.region_index): 
            indexx=np.where(self.region_index==ii)[0]
            nodee=Branch_division(
                            cclu=cclu,
                            dend= self.dend, 
                            vertices_index=indexx,
                            size_threshold= size_threshold,
                            tf_cluster_faces_unique=False,)
            for nod in nodee.children:
                indxx=nod.vertices_index
                facess=nod.faces 
                dist,indx=tree.query(vertices[indxx] )  
                self.mapp[tuple(indxx)]=dict(faces=facess, ) 
    
    

    def get_skl_smooth(self,
                        subsample_thre=.02,
                        f=0.99,
                        N=10,
                        num_chunks=100, 
                        num_points=50,
                        ctl_run_thre=1,
                            line_num_points= None,
                            line_num_points_inter= None,
                            spline_smooth= None,):
        if not hasattr(self, "mapp") or not self.mapp:
            self.get_region_branch() 
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        mmap_keys=list(self.mapp.keys()) 
        tree = KDTree(skl_vectices)    
        skl_vertices_index =[]  
        self.intensity_thickness=np.zeros(vertices.shape[0])
        for liss in mmap_keys:
            lis=list(liss)
            if len(lis)<3:
                continue
            dist,indx=tree.query(vertices[lis] )   
            ctl_tmp = center_curvature(vertices=skl_vectices[indx], 
                    line_num_points= line_num_points,
                    line_num_points_inter= line_num_points_inter,
                    spline_smooth= spline_smooth,
                )
            skl_vertices_index.append(ctl_tmp.vertices_center)  
        self.skl_vertices_index=np.vstack(skl_vertices_index)
        tree = KDTree(self.skl_vertices_index)   
        self.intensity_thickness=tree.query(vertices)[0]



    def get_inverse_mapping(self, index): 
        if not hasattr(self, "mapp") or not self.mapp:
            self.get_region_branch() 
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        set_ind=set(index)
        set_ind_copy=set_ind.copy() 
        indexx=[] 
        set_ind_copy_tmp=set()
        for lis in self.mapp:
            inter= set_ind.intersection(lis)
            if len(inter)>3:
                set_ind_copy_tmp.update(inter) 
                indexx.extend(lis) 
            if len(set_ind_copy-set_ind_copy_tmp )<4:
                break
        cclu=cluster_class(faces_neighbor_index=self.dend.vertex_neighbor) 
        node =Branch_division(
                        cclu= cclu,
                        dend=self.dend, 
                        vertices_index=indexx,
                        size_threshold= 2, 
                        stop_index=1,
                        ) 
        return node





    def get_skl_region(self,skl_shaft_vectices,shaft_index,spine_index,zoom_thre=None,zoom_thre_min=None,zoom_thre_max=None,):
        if not hasattr(self, "mapp") or not self.mapp:
            self.get_region_branch()
        zoom_thre     = zoom_thre     or self.zoom_thre
        zoom_thre_min = zoom_thre_min or self.zoom_thre_min
        zoom_thre_max = zoom_thre_max or self.zoom_thre_max

        get_neck_index=self.get_neck_region(skl_shaft_vectices,shaft_index,spine_index,
                                            zoom_thre=zoom_thre,
                                            zoom_thre_min=zoom_thre_min,
                                            zoom_thre_max=zoom_thre_max,) 
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        mmap_keys=list(self.mapp.keys())
        set_ind=set(spine_index)
        set_ind_copy=set_ind.copy()
        set_ind_copy_tmp=set()
        skl_index_loc=set() 
        skl_distance_loc=set() 
        tree = KDTree(skl_vectices) 
        indall= tree.query(vertices[list(set_ind)] )[1]
        indall_neck=tree.query(vertices[get_neck_index] )[1]
        skl_spine_index =[]
        skl_neck_index =[]
        scatter=[]
        gcol=get_color()
        ii=0
        jj=0
        for lis in mmap_keys:
            inter= set_ind.intersection(lis)
            set_ind_copy_tmp.update(inter)
            dist,indx=tree.query(vertices[list(inter)] )  
            if len(inter)>3:
                self.intensity_tickness[list(lis)]=np.mean(dist)
                
                skl_index_loc.update(set(indx).intersection(indall)) 
                nehh=list(set(lis).intersection(set(get_neck_index)))
                if len(nehh)>0:
                    indall_neck=list(set(tree.query(vertices[nehh] )[1]))
                    skl_neck_index.append(indall_neck)
                    skl_spine_index.append(list(set(indx).intersection(indall)-set(indx).intersection(set(indall_neck))))
                    iic =min(jj,len(gcol)-1)  
                    jj+=1
                    scatter.append(hf.plotly_scatter(points=skl_vectices[indall_neck] , color=gcol[iic], size=6.95 ,name='neck'))
                    scatter.append(hf.plotly_scatter(points=skl_vectices[list(set(indx).intersection(indall)-set(indx).intersection(set(indall_neck)))] , color=gcol[iic], size=3.95 ))
                    continue  
                skl_spine_index.append(list(set(indx).intersection(indall)))
                iic =min(ii,len(gcol)-1)  
                ii+=1
                scatter.append(hf.plotly_scatter(points=skl_vectices[list(set(indx).intersection(indall))] , color=gcol[iic], size=3.95 )) 
            if len(set_ind_copy-set_ind_copy_tmp )<4:
                break
        return skl_spine_index,skl_neck_index,scatter

 

    def get_skl_neck_region(self,skl_shaft_vectices,shaft_index,spine_index,zoom_thre=None,zoom_thre_min=None,zoom_thre_max=None,):
        if not hasattr(self, "mapp") or not self.mapp:
            self.get_region_branch()
        zoom_thre     = zoom_thre     or self.zoom_thre
        zoom_thre_min = zoom_thre_min or self.zoom_thre_min
        zoom_thre_max = zoom_thre_max or self.zoom_thre_max
        get_neck_index=self.get_neck_region(skl_shaft_vectices,shaft_index,spine_index,
                                            zoom_thre=zoom_thre,
                                            zoom_thre_min=zoom_thre_min,
                                            zoom_thre_max=zoom_thre_max,)
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        mmap_keys=list(self.mapp.keys()) 
        set_ind=set(spine_index)
        set_ind_copy=set(get_neck_index)
        set_ind_copy_tmp=set() 

        tree = KDTree(skl_vectices) 
        indall= tree.query(vertices[list(set_ind)] )[1]
        indall_neck=tree.query(vertices[get_neck_index] )[1]  
        skl_neck_index =[]  
        for lis in mmap_keys:
            inter= set_ind.intersection(lis)
            dist,indx=tree.query(vertices[list(inter)] )  
            if len(inter)>3:   
                nehh=set(lis).intersection(set(get_neck_index))
                set_ind_copy_tmp.update(nehh)
                if len(nehh)>0:
                    indall_neck=tree.query(vertices[list(nehh)] )[1]
                    skl_neck_index.extend(indall_neck)
            if len(set_ind_copy-set_ind_copy_tmp )<4:
                break
        return list(set(skl_neck_index) )


 

    def get_neck_region(self,skl_shaft_vectices,shaft_index,spine_index,zoom_thre=None,zoom_thre_min=None,zoom_thre_max=None,size_threshold=3):
        vertices,skl_vectices=self.dend.vertices,self.skl_vectices
        mapp=mappings_vertices(vertices_0=vertices)  
        vcv_length= KDTree(skl_shaft_vectices).query(vertices)[0] 
        clu_pca=clust_pca(vertices,shaft_index,
                        vertices_center=  skl_shaft_vectices,
                        vcv_length=vcv_length)   

        cls=clust_segment(vertices_0=self.dend.vertices ,
                        vertices_index=spine_index,
                        vertices_index_shaft=shaft_index,
                        clu_pca= clu_pca, 
                        size_threshold=size_threshold, 
                    #  vcv_length=vcv_length,
                        )   
        llin=np.linspace(0,1,zoom_thre) 
        spine_index_cls_10=mapp.Mapping_inverse(cls.Cyl_threshold(zoom_threshold= llin[min(zoom_thre_min,zoom_thre)]) )
        spine_index_cls_30=mapp.Mapping_inverse(cls.Cyl_threshold(zoom_threshold= llin[min(zoom_thre_max,zoom_thre)]) )
        return list(set(spine_index_cls_10)-set(spine_index_cls_30))
    


         