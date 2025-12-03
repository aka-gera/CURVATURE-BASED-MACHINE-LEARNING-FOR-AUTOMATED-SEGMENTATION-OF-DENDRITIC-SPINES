

import os

import pickle
import numpy as np 
from dend_fun_0.help_funn import get_color, loadtxt,loadtxt_count,closest_distances_group  
import dend_fun_0.help_plotly as hp  
import plotly.graph_objects as go 
from dend_fun_0.help_save_iou import iou_train
import dend_fun_0.help_fun as hf
clor=get_color()
from dend_fun_0.help_pinn_data_fun import get_cluster_length_using_center_curve

import dend_fun_0.density as den

def get_scatter_center(self,spine_path,shaft_vertices_center_path,save_data=True):
    paath=os.path.join(spine_path,self.txt_spine_count)
    if not os.path.exists(paath):
        return 
    count= loadtxt_count(os.path.join(spine_path,self.txt_spine_count))
    mmm=count.ndim
    count=count if mmm==2 else count.reshape(-1,1)
    vertices_center_shaft=np.loadtxt(os.path.join(shaft_vertices_center_path, self.txt_shaft_vertices_center),ndmin=2)
    vertices_center_shaft=np.atleast_2d(vertices_center_shaft) 

    scatter_spine = {}
    scatter_spine[0]=[]
    scatter_spine[0].append(hf.plotly_scatter(points=self.vertices_00, color='purple', size=3.3, name='dendrite',opacity=0.5))

    vertices_spine=np.loadtxt( os.path.join(shaft_vertices_center_path, f'shaft_index.txt'),dtype=int )
    scatter_spine[0].append(hf.plotly_scatter(points=vertices_center_shaft, color='red', size=3.3, name='shaft skl'))

    scatter_spine[1]=[]
    scatter_spine[1].append(hf.plotly_scatter(points=vertices_center_shaft, color='red', size=3.3, name='shaft skl'))
    scatter_spine[1].append(hf.plotly_scatter(points=self.vertices_00[vertices_spine], color='purple', size=3.3, opacity=0.2, name='shaft',  ))
    xco=0
    for i in range(count.shape[0]): 
        ii=count[i,0]
        if ii<0:
            continue
        name=f'{ii}_{count[i,1]}' if mmm==2 else f'{ii}'
        ixi=i if i<len(clor) else len(clor)-1
        key=i+2    
        scatter_spine[key]=[]
        cent_path=os.path.join(spine_path, f'spine_center_curv_{name}.txt')
        print('center path',cent_path,os.path.exists(cent_path)) 
        if os.path.exists(cent_path):
            vertices_center=np.loadtxt( cent_path,ndmin=2 ) 
            if vertices_center.shape[0]>0:  
                vercent=hf.plotly_scatter(points=vertices_center, color=clor[ixi], size=5, opacity=0.8, name=f'skl_{name}') 
                scatter_spine[key].append(vercent)
                scatter_spine[0].append(vercent)

        vertices_spine=np.loadtxt( os.path.join(spine_path, f'spine_index_{name}.txt'),dtype=int )  
        scatter_spine[key].append(hf.plotly_scatter(points=self.vertices_00[vertices_spine], color=clor[ixi], size=2, opacity=0.5, name=f'spn_{name}',  )) 

    return  scatter_spine

  


def get_iou_graph(self,spine_path,mp=None ,save_data=True):
    sze_check_un = loadtxt(os.path.join(spine_path, self.txt_spine_iou), dtype=float)  
    if sze_check_un.ndim>1:
        scatter_graph = [
                go.Scatter(
                    x=sze_check_un[:, 0],
                    y=sze_check_un[:,-1],
                    mode='markers',
                    marker=dict(color=sze_check_un[:, 1], size=10,symbol='square' ),  
                    name='IOU Union'
                ), 
                go.Scatter(
                    x=sze_check_un[:, 0],
                    y=sze_check_un[:,-2],
                    mode='markers',
                    marker=dict(color=sze_check_un[:, 1], size=10,symbol='circle' ),  
                    name='IOU'
                ), 
                hf.plotly_scatter(points=hf.Lines_plot(sze_check_un[:, [0,-2]],
                                                        sze_check_un[:, [0,-1]]),
                                                        color=sze_check_un[:,1],
                                                        size=4,
                                                        mode='lines+markers',
                                                        name='Union'),
            ]
    else:
        scatter_graph =[] 
    if save_data:
        with open(os.path.join(spine_path,'plot_iou_graph.pkl'), 'wb') as f:
            pickle.dump(scatter_graph, f) 
    else:
        return  scatter_graph


class graph_iou:
    def __init__(self,pid,path_file,path_train,center_path):
        pid.get_dend_data()   
        self.path_file=path_file
        self.pid=pid
        self.path_train=path_train
        self.center_path=center_path   


    def scatter_center(self): 
        spine_path=self.path_file[self.path_train[self.center_path]]    
        shaft_vertices_center_path=self.path_file[self.path_train['data_shaft_path']]   
        scatter_spine=get_scatter_center(self.pid, spine_path,shaft_vertices_center_path) 
        with open(os.path.join(spine_path,'plot_data_center_curv.pkl'), 'wb') as f:
            pickle.dump(scatter_spine, f)  


    def iou_graph(self,iou_thre=0.2):
        spine_path=self.path_file[self.path_train['dest_appr_iou']]
        mp_path=os.path.join( spine_path,self.pid.pkl_mp)
        if not os.path.exists(mp_path): 
            iou_tr=iou_train(
                            path_true=self.path_file[self.path_train['data_true_iou']],
                            path_appr=self.path_file[self.path_train['dest_appr_iou']],
                            iou_thre=iou_thre)
            iou_tr.get_mapping(save=True)
            iou_tr.get_iou_save(save=True)

        with open(mp_path, "rb") as file:
            mp = pickle.load(file) 
        get_iou_graph(self.pid,spine_path,mp ,save_data=True)







def get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict={}, iou_per=70,labels = ['False', 'True'],nbinsx=100,):
        
        sze_check =np.array(sze_check )
        sze_check_un =np.array(sze_check_un )
        sze_checks_0 =np.array(sze_checks_0 ) 
        y_pred=(sze_check >= int(iou_per)/100).astype(int)   
        y_tru=(sze_checks_0  >= 0).astype(int)   
        iou_dict['single']={} 
        iou_dict['single']['cm']=cm=hp.Confusion_matrix(y_true=y_tru,y_predicted=y_pred)
        accuracy, precision, recall, f1_score=hp.Compute_metrics(cm)  
        iou_dict['single']['metrics']=dict(accuracy=accuracy, 
                                           precision=precision, 
                                           recall=recall, 
                                           f1_score=f1_score)   
        iou_dict['single']['heatmap_cm']=go.Heatmap(
            z=cm,
            x=labels,   
            y=labels,   
            colorscale='Blues',
            text=cm,   
            texttemplate="%{text}",  
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ) 

        iou_dict['single']['histogram']=go.Histogram(
            x=sze_check ,  
            nbinsx=nbinsx,  # Number of bins, adjust based on data range
            name="IoU",  # Legend name
            marker=dict(color="blue"),  # Bar color
            opacity=0.6  # Set opacity
        )
 
        y_pred=(sze_check_un  >= int(iou_per)/100).astype(int)     
        iou_dict['union']={} 
        iou_dict['union']['cm']=cm=hp.Confusion_matrix(y_true=y_tru,y_predicted=y_pred)
        accuracy, precision, recall, f1_score=hp.Compute_metrics(cm)  
        iou_dict['union']['metrics']=dict(accuracy=accuracy, 
                                           precision=precision, 
                                           recall=recall, 
                                           f1_score=f1_score)   
 
        iou_dict['union']['heatmap_cm']=go.Heatmap(
                                    z=cm,
                                    x=labels,  
                                    y=labels,  
                                    colorscale='Blues',
                                    text=cm,  # Show numbers in cells
                                    texttemplate="%{text}",  # Format as numbers
                                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                                ) 
        
        iou_dict['union']['histogram']=go.Histogram(
            x=sze_check_un ,  
            nbinsx=nbinsx,   
            name="IoU union",   
            marker=dict(color="red"),   
            opacity=0.6   
        )
        # iou_hist.append([hist ,hist_un ])
        return iou_dict

 

class graph_cylinder_heatmap:
    def __init__(self,pid,path_file,path_train,center_path,file_path_feat=None,):
        pid.get_dend_data()   
        self.path_file=path_file
        self.pid=pid
        self.path_train=path_train
        self.center_path=center_path 
        self.file_path_feat=file_path_feat  

        self.spine_path=self.path_file[self.path_train[self.center_path]] 

    def coordinates(self): 
        spine_path=self.spine_path   
        count_path=os.path.join(self.file_path_feat, 'skl_shaft_vectices.txt')#os.path.join(self.file_path_feat, 'shaft_vertices_center.txt')
        skl_shaft_distance=np.loadtxt(os.path.join(self.file_path_feat, 'skl_shaft_distance.txt'))
        if os.path.exists(count_path): 
            shaft_vertices_center=np.loadtxt(count_path, dtype=float)
        paath=os.path.join(spine_path,self.pid.txt_spine_count)
        if not os.path.exists(paath):
            return
        count=np.loadtxt(paath,dtype=int)
        mmm=count.ndim
        count=count if mmm==2 else count.reshape(-1,1)    
        save={}  
        save['close']=[]  
        save['farthest']=[]
        save['center']=[] 
        save['point_a']=shaft_vertices_center[0, :]
        save['point_b']=shaft_vertices_center[-1, :]
        iii=0
        dst=-1
        sac=[]
        for i in range(count.shape[0]): 
            ii=count[i,0]
            name=f'{ii}_{count[i,1]}' if mmm==2 else f'{count[i,0]}'  
            iii+=1  
            count_path=os.path.join(spine_path, f'{self.pid.name_spine_index}_{name}.txt')
            cen_path=os.path.join(spine_path, f'spine_center_curv_{name}.txt')
            if os.path.exists(count_path) and os.path.exists(cen_path):  
                vertices_center=np.loadtxt(cen_path  ) 
                if len(vertices_center)>3:
                    spine_index=np.loadtxt(count_path,dtype=int) 
                    verttt=self.pid.dend.vertices[spine_index]  
                    clo_min,gf= closest_distances_group(verttt,shaft_vertices_center, num_chunks=20)
                    save['close'].append( verttt[np.argmin(clo_min)])  
                    save['center'].append(shaft_vertices_center[gf[np.argmin(clo_min)][0]])
                    sac.append(np.linalg.norm(shaft_vertices_center[gf[np.argmin(clo_min)][0]]-verttt[np.argmin(clo_min)])) 
        llo=np.linalg.norm(save['point_a']-save['point_b'])
        save['radius']=1#np.mean(skl_shaft_distance)#max(np.mean(sac),1) 
        save['height']=int(llo*500) 
        save['width']=int(save['radius']*500)
        return save



    def get_cylinder_heatmap(self,save_data=True):
        save=self.coordinates() 
        if save is None:
            return None
        if (len(save['center'])==0)or(len(save['close'])==0):
            return None
        cyl=den.get_cylinder(shaft_points=np.array(save['center']),
                            neck_points=np.array(save['close']),
                            point_a=save['point_a'], 
                            point_b=save['point_b'], 
                            radius=save['radius'],
                            width=save['width'],
                            height=save['height'],
                            ) 
        if save_data:
            with open(os.path.join(self.spine_path,'cylinder_heatmap.pkl'), 'wb') as f:
                pickle.dump(cyl, f) 
        else:
            return  cyl    





import numpy as np
import plotly.graph_objects as go

def compute_histogram(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    return hist, bin_edges

def kl_divergence(P, Q): 
    P = P / np.sum(P)
    Q = Q / np.sum(Q) 
    mask = (P > 0) & (Q > 0)
    return np.sum(P[mask] * np.log(P[mask] / Q[mask]))



def compute_kl(df_save,names,mode,width,height,title=None,bins = 500,showscale=False ,): 
    his=[]
    for nam in names:
        data_P = df_save[nam]['df'][mode]
        hist_P, _ = compute_histogram(data_P, bins)
        his.append(hist_P)

    kl_save=[]
    for hi1 in his:
        tmp=[]
        for hi2 in his:  
            tmp.append(np.abs(kl_divergence(hi1, hi2)))
        kl_save.append(tmp)
    cm=np.array(kl_save)
    cm= np.round(cm*1e4)/1e4
    labels=names


    scatter=go.Heatmap(
                        z=cm,
                        x=labels,  
                        y=labels,  
                        colorscale='Blues',
                        text=cm,  # Show numbers in cells
                        texttemplate="%{text}",  # Format as numbers
                        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                        showscale=showscale ,
                    ) 

    layout = go.Layout(
        title=title, 
        width=width, 
        height=height, 
    )

    return scatter,layout
