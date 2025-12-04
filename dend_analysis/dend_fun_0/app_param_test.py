 
import sys
import os
 
import pickle
# import dash
from dash import dcc, html, dash_table, Input, Output, State, callback 
import dash_bootstrap_components as dbc
import numpy as np
import dend_fun_0.curvature as cu  
from dend_fun_0.help_funn import get_color
import dend_fun_0.help_fun as hf
# import geometry as geo
# import help_plotly as hpp
import dend_fun_0.help_plotly as hp
from dend_fun_0.help_plotly import aka_plot 
# import density as den
import plotly.graph_objects as go 
import dend_fun_0.help_funn as hff
from dend_fun_0.get_path import get_files,get_app_param ,safe_id 
from dend_fun_0.help_graph import get_iou_graph,get_cm_iou,compute_kl 
from dend_fun_0.help_save_iou import iou_train
import pandas as pd
 
 
def add_unit(label, unit="µm",):
    return f"{label} ({unit})"

def add_unit(label: str, unit="µm",) -> str:
    label_lower = label.lower()
    if any(word in label_lower for word in ["length", "width", "diameter"]):
        return f"{label} ({unit})"
    elif "area" in label_lower:
        return f"{label} ({unit}<sup>2</sup>)"
    elif "volume" in label_lower:
        return f"{label} ({unit}<sup>3</sup>)"
    else:
        return label   

def get_metric(
    akp,mode, metric, metric_name, width=500, height=500, 
    nbinsx=30, xtitle='Length', ytitle='Count'
):
    metric_map = {
        metric_name[0]: 0,
        metric_name[1]: 1,
        metric_name[2]: 2,
    }
    
    if mode not in metric_map:
        raise ValueError(f"Invalid mode: {mode}")

    index = metric_map[mode]
    data = metric[:, index]
    
    title = f'{mode} Histogram'

    scatter_comp, layout_comp = hp.Plotly_histogram_return(
        data, nbinsx=nbinsx, title=title, xtitle=xtitle, ytitle=ytitle, width=width, height=height
    )
    
    return akp.Plotly_Figure(data=scatter_comp, layout=layout_comp)





dtype = float
class app_param(get_files,get_app_param): 
    def __init__(self, file_path_org, 
                model_sufix,
                path_train,
                index=0, 
                dend_names=None,
                dend_namess=None, 
                data_studied=None,
                dend_path_inits=None,
                path_file=None,
                prevent_initial_call=False,  
                pinn_dir_data=None,
                dend_data=None,
                path_switch=['dest_spine_path_pre','dest_spine_path'],
                model_type=None,
                obj_org_path_dict=None,
                model_sufix_dic=None, 

                 ):
        pass 
        self.file_path_org=file_path_org
        self.path_switch=path_switch
        path_dir=os.path.join(file_path_org, 'data')
        model_sufix_all=np.loadtxt(os.path.join(path_dir, 'model_sufix_all.txt'),dtype=str,ndmin=1) 
        pinn_dir_data_all=np.loadtxt(os.path.join(path_dir, 'pinn_dir_data_all.txt'),dtype=str,ndmin=1)
        path_heads=np.loadtxt(os.path.join(path_dir, 'path_heads.txt'),dtype=str,ndmin=1)
        true_keys=np.loadtxt(os.path.join(path_dir, 'true_keys.txt'),dtype=str,ndmin=1)
        if dend_data is not None: 
            dend_names=dend_names if not None else dend_data['dend_names']
            dend_namess=dend_namess if not None else dend_data['dend_namess']
            dend_path_inits=dend_path_inits if not None else dend_data['dend_path_inits'] 
        get_files.__init__(self,
                            dend_data=dend_data,
                            file_path_org=file_path_org,
                            dend_names=dend_names,
                            dend_namess=dend_namess, 
                            dend_path_inits=dend_path_inits,
                            data_studied=data_studied,   
                            model_sufix=model_sufix,
                            # path_file=path_file,
                            pinn_dir_data=pinn_dir_data,
                            pinn_dir_data_all=pinn_dir_data_all,
                            model_sufix_all=model_sufix_all,
                            true_keys=true_keys,
                            model_type=model_type,
                            path_heads=path_heads, 
                            obj_org_path_dict=obj_org_path_dict,
                            model_sufix_dic=model_sufix_dic,
                 )
        self.path_train=path_train
        self.index=index
        get_app_param.__init__(self,
                               dropdown_path_head_option=self.dropdown_path_head_option,
                               dropdown_model_suf_option=self.dropdown_model_suf_option,
                               dropdown_path_option=self.dropdown_path_option,
                               dropdown_true_keys_option=self.dropdown_true_keys_option,
                               ) 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type )
        self.get_dend_name(data_studied=data_studied,index=0 ) 
        file_path=self.file_path 
        dend_name=self.dend_name or dend_names[index]
        dend_namess=self.dend_namess   
 
        self.prevent_initial_call=prevent_initial_call
        id_name_end=f'{model_type}_{dend_name}_{index}_{file_path}_{model_sufix}'
        id_name_end=safe_id(id_name_end)   
        spine_path = self.path_file[path_train['dest_shaft_path']]   
        # spine_path = self.path_file[path_train['dest_spine_path']]  
        iou_count = np.loadtxt(os.path.join(spine_path , self.txt_spine_iou), dtype=float)
        print('iou_count',len(iou_count),id_name_end,dend_name) 
        iou_count=iou_count if iou_count.ndim==2 else np.array([[-1,-1,0,0]])
        
        self.model_test()
        self.more_param(id_name_end=id_name_end,
                        model_sufix=model_sufix,
                        dend_name=dend_name,)
        self.get_dropdown_cluster(id_name_end,iou_count[:,:2].astype(int))
        self.get_dropdown_index(id_name_end=id_name_end,
                                dend_names=dend_data['dend_names'],
                                index=index,)
        print(dend_names)
        self.get_data(
                    dend_data=self.dend_data,
                    model_sufix=model_sufix,
                    data_studied=self.data_studied,
                    index=index,
                    ) 
 
        self.app_layout = html.Div(
            style={
                # 'color': 'black',
                # 'backgroundColor': 'grey',
                # 'height': '100vh'  # Full viewport height
                'font-size': 20
            },
            children=self.Get_children()
        )
        

        self.Output=[
            Output(self.output_graph_1['id'], 'children'), 
            Output(self.output_text_1['id'], 'children'), 
        ]
        self.Input=[
            # Input(self.dropdown_true_keys['id'],      'value'),
            Input(self.dropdown_path_head['id'],      'value'), 
            Input(self.dropdown_model_suf['id'],      'value'),
            Input(self.dropdown_path['id'],      'value'),
            Input(self.dropdown_mode['id'],      'value'),
            Input(self.dropdown_dend['id'],      'value'),
            Input(self.dropdown_cluster['id'],   'value'), 
            Input(self.dropdown_intensity['id'], 'value'),
            Input(self.width_slider['id'],           'value'),
            Input(self.height_slider['id'],          'value'), 
            Input(self.dropdown_template['id'], 'value'),
            Input(self.hist_slider['id'],           'value'),
            Input(self.dropdown_index['id'],   'value'), 
        ], 
 

    def get_data(self,model_sufix,data_studied,index,dend_data=None): 
        path_train=self.path_train
        if dend_data is not None:
            self.dend_path_inits= dend_data['dend_path_inits']
            self.dend_names= dend_data['dend_names']
            self.dend_namess= dend_data['dend_namess']
        self.get_model_opt_name(model_sufix=model_sufix,model_type=self.model_type )
        self.get_dend_name(data_studied=data_studied,
                           index=index,
                            dend_names= self.dend_names,
                            dend_namess=self.dend_namess,
                            dend_path_inits=self.dend_path_inits,) 
        spine_path= self.path_file[path_train['dest_shaft_path']] 
        # spine_path= self.path_file[path_train['dest_spine_path']] 
        file_path=self.file_path 
        dend_name=self.dend_name 
        dend_namess=self.dend_namess 
        self.plot_data_iou=None 
        self.model_shap_dic={}
        self.plot_data_iou_dic={}
        self.plot_data_center_curv={} 
        self.plot_data_cylinder_heatmap={} 
        self.iou_count={}
        self.scatter_loss_dic={}
        self.scatter_iou_dic={}
        self.scatter_loss_spine_dic={}
        self.scatter_iou_spine_dic={}
        self.metric_total_dic={}
       #   self.metric_total_dic['union']['single']={}
        self.metric_total_dic={}
        self.metric_total_dic['single']={}
        self.metric_total_dic['union']={}
        self.iou_tr={}
        # self.metric_total_dic = {
        #     key: {}
        #     for iii in self.dend_path_original_mm['keys'].values() 
        #     for key in (f'single_{iii}', f'union_{iii}')
        # }

        for path_head in self.path_heads:
            for model_suf in self.model_sufix_all:
                for path in self.pinn_dir_data_all: 
                    id_path=f'{path_head}_{model_suf}_{path}' 

                    fgff=os.path.join(self.path_file[id_path], self.txt_spine_iou)
                    if os.path.exists(fgff):
                        self.iou_count[id_path]=  np.loadtxt(fgff, dtype=float)    

                    path_grap_center_curv=os.path.join(self.path_file[id_path] ,'plot_data_center_curv.pkl')
                    if os.path.exists(path_grap_center_curv):
                        with open(os.path.join(path_grap_center_curv), "rb") as file:
                            self.plot_data_center_curv[id_path] = pickle.load(file) 

                    path_grap_center_curv=os.path.join(self.path_file[id_path] ,'cylinder_heatmap.pkl')
                    if os.path.exists(path_grap_center_curv):
                        with open(os.path.join(path_grap_center_curv), "rb") as file:
                            self.plot_data_cylinder_heatmap[id_path] = pickle.load(file) 


                    ''' DONT DELETE
                    self.scatter_loss_data=[] 
                    tyy =self.inten_file_model_head_neck_loss[0]
                    loss_path=self.path_file_sub[tyy][id_path]
                    # print('ooop------===================================',loss_path)

                    if os.path.exists(loss_path):
                        self.loss_data = np.loadtxt(loss_path, dtype=float)  
                        self.loss_data= np.vstack((np.arange(len(self.loss_data)),self.loss_data)).T
                        # print('pp----------',self.loss_data.shape)
                        self.scatter_loss_data.append(
                                hf.plotly_scatter(points=self.loss_data ,
                                                color='red',
                                                size=4.07,
                                                opacity=.8,
                                                name='Approx')
                                                ) 
                    self.scatter_loss_dic[id_path]=  self.scatter_loss_data
 

                    tyy =self.inten_file_model_head_neck_iou[0] 
                    self.scatter_iou_data=[]   
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_iou[:-1],['head','neck','shaft'],['red','green','blue'])):
                        # iou_path=self.model_dir_path['head_neck']['iou'][ii]
                        iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        if os.path.exists(iou_path):
                            _data=np.loadtxt(iou_path,dtype=float)
                            print('pp----------',_data.shape) 
                            for ii in range(_data.shape[1]):
                                _dataa= np.hstack((np.arange(_data.shape[0]).reshape(-1,1),_data[:,ii:ii+1]))
                                self.scatter_iou_data.append(hf.plotly_scatter(points=_dataa ,
                                                        color=couleur,
                                                        size=4.07,
                                                        opacity=.8,
                                                        name=f'{nam}_{ii}'))

                    self.scatter_iou_dic[id_path]=self.scatter_iou_data

'''



                    scatter_loss_data=[] 
                    tyy =self.inten_file_model_spine_loss[0]
                    loss_path=self.path_file_sub[tyy][id_path]
                    # print('ooop------',loss_path)

                    if os.path.exists(loss_path):
                        self.loss_data = np.loadtxt(loss_path, dtype=float)  
                        self.loss_data= np.vstack((np.arange(len(self.loss_data)),self.loss_data)).T
                        # print('pp----------',self.loss_data.shape)
                        scatter_loss_data.append(
                                hf.plotly_scatter(points=self.loss_data ,
                                                color='red',
                                                size=4.07,
                                                opacity=.8,
                                                name='Approx')
                                                ) 

                    self.scatter_loss_spine_dic[id_path]= scatter_loss_data

                    tyy =self.inten_file_model_spine_iou[0] 
                    scatter_iou_data=[]   

                                    
                    cnt=True
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_spine_iou,['shaft','spine'],['red','blue'])):
                        # iou_path=self.model_dir_path['head_neck']['iou'][ii]
                        iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        if os.path.exists(iou_path): 
                            _data=np.loadtxt(iou_path,dtype=float)  
                            m1,m2=.2,.8
                            for ii,(symb,mdd,siz) in enumerate(zip(['star','star','star','star','square','square'],
                                                            ['train','train','train','train','test','test'],
                                                            [m1,m1,m1,m1,m2,m2])):
                                if ii<_data.shape[1]:
                                    _dataa= np.hstack((np.arange(len(_data)).reshape(-1,1),_data[:,ii:ii+1])) 
                                    scatter_iou_data.append(hf.plotly_scatter(points=_dataa ,
                                                            color=couleur,
                                                            size=10.,
                                                            opacity=siz,
                                                            name=f'{nam}_{mdd}_{ii}',
                                                            symbol=symb))

                        elif os.path.exists(self.df_metric_algorithms_dir) and cnt: 
                            df_union=pd.read_csv(self.df_metric_algorithms_dir)#.sort_values(by=f'Accuracy', ascending=True) 
                            ylabels=df_union.columns[1:]
                            xlabels=df_union.iloc[:,0]
                            df_union = df_union.set_index("Unnamed: 0")
                            df_union=df_union.T.sort_values(by=f'Accuracy', ascending=True)

                            # df_union=df_union[df_union.columns[1:]]
                            cm=np.array(df_union.select_dtypes(include=[np.number]))
                            cm= np.round(cm*1e3)/1e3
 
                            scatter_iou_data.append(
                                                    go.Heatmap(
                                                                z=cm,
                                                                x=xlabels,  
                                                                y=ylabels,  
                                                                colorscale='Blues',
                                                                text=cm,  # Show numbers in cells
                                                                texttemplate="%{text}",  # Format as numbers
                                                                # hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                                                                showscale=False ,
                                                                textfont=dict(size=18),
                                                            ) 
                                            ) 
                            cnt=False
 

                    self.scatter_iou_spine_dic[id_path]=scatter_iou_data



 



                    clor=['red','blue' ]
                    vds=[1,-1]
                    valss=['Shaft','Spine']
                    figg=go.Figure()
                    # tyy =self.inten_file_model_shap[0]  
                    # iou_path=self.path_file_sub[tyy][id_path] 
                    self.model_shap=[]   
                    head_neck_path = 'dest_shaft_path'
                    pathh=path_train[head_neck_path]
                    spine_path_save=     self.path_file[f'result_{pathh}']
                    iou_path=os.path.join(spine_path_save,'shap.csv') 
                     
                    if os.path.exists(iou_path):
                        # print('---------------->>>>>>>>><<<<<<<<<<<<<<<,,',iou_path)

                        df = pd.read_csv(iou_path)   
                        for nam,cl,vd,va in zip(df.columns[1:],clor,vds,valss):
                            figg.add_trace(
                                go.Bar(
                                    x=vd*df[nam][::-1],
                                    y=df['Feature'][::-1],
                                    orientation='h',
                                    marker_color=cl,
                                    name=f'{va}'
                                )
                            )  
                        figg.update_layout(
                            title='Diverging SHAP Summary',
                            barmode='relative',
                            xaxis_title='SHAP value',
                            yaxis_title='Feature',
                            xaxis=dict(zeroline=True),
                            bargap=.2,
                        )

                    self.model_shap_dic[id_path]=figg







        for iii,(true_path,true_key) in enumerate( self.dend_path_original_mm['keys'].items() ):
            for path_head in self.path_heads:
                for model_suf in self.model_sufix_all:
                    for path in self.pinn_dir_data_all: 
                        id_path=f'{path_head}_{model_suf}_{path}'
                        id_pathss=f'{path_head}_{model_suf}_{path}_{true_key}'
                        path_grap_iou=os.path.join(self.path_file[id_path] ,self.pkl_mp)
                        if os.path.exists(path_grap_iou):
                            with open(path_grap_iou, "rb") as file:
                                mp = pickle.load(file) 
                            self.iou_tr[id_pathss]=iou_train(
                                                path_true=self.path_file[true_path],
                                                path_appr=self.path_file[id_path],
                                                mp=mp) 
                            
                        path_grap_iou=os.path.join(self.path_file[id_path] ,f'plot_iou_graph_{true_key}.pkl')
                        if os.path.exists(path_grap_iou): 
                                with open(os.path.join(path_grap_iou), "rb") as file: 
                                    self.plot_data_iou_dic[id_pathss]=pickle.load(file) 
 






                        if model_suf !='save':
                            spine_path_save=     self.path_file[f'result_{id_path}']
                            metric_path=os.path.join( spine_path_save,f'iou_{true_key}.csv') 
                            if os.path.exists(metric_path):
                                df = pd.read_csv(metric_path)
                                # print('----=====----','im her')
                                sze_checks_0 ,sze_check ,sze_check_un=df['id_true'],df['iou_single'],df['iou_union']
                                # nhf=df['id_true']>0
                                # sze_checks_0 ,sze_check ,sze_check_un=sze_checks_0[nhf] ,sze_check[nhf] ,sze_check_un[nhf]
                                metric_name_path=df.columns
                                iou_dict={}
                                get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict=iou_dict, iou_per=70,labels = ['False', 'True'],nbinsx=300,) 
                                accuracy , precision, recall, f1_score=iou_dict['single']['metrics'].values()
                                # self.metric_total_dic[id_path]={}
                                self.metric_total_dic['single'][id_path]=dict(accuracy=accuracy,
                                                                    precision=precision,
                                                                    recall=recall,
                                                                    f1_score=f1_score)
                                accuracy, precision, recall, f1_score=iou_dict['union']['metrics'].values()
                                self.metric_total_dic['union'][id_path]=dict(accuracy=accuracy,
                                                                    precision=precision,
                                                                    recall=recall,
                                                                    f1_score=f1_score) 
            if len(self.metric_total_dic['single'])>0:
                df_union=pd.DataFrame(self.metric_total_dic['union']).T.sort_values(by='accuracy', ascending=False)
                df_union.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_union_{true_key}.csv'))
                df_single=pd.DataFrame(self.metric_total_dic['single']).T.sort_values(by='accuracy', ascending=False)
                df_single.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_single_{true_key}.csv'))
 
        self.file_path =file_path
        self.dend_name=dend_name
        self.itera=index  

        self.vertices_0=vertices_0       = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_0), dtype=float) 
        self.vertices_00=vertices_00      = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_1), dtype=float)
        self.faces=faces = np.loadtxt(os.path.join(self.file_path, self.txt_faces), dtype=int)


        print(f'Starting analysis of {dend_name}')
        print(f"Number of vertices: {len(vertices_0)}")
        print(f"Number of faces: {len(faces)}")
        
        self.inten={} 
        for port in self.pre_portions:
            self.inten[f'{port}_body']=np.zeros_like(vertices_00[:,0]) 
            mmm=os.path.join(spine_path,f'intensity_{port}_segm.txt') 
            if os.path.exists(mmm):  
                self.inten[f'{port}_body']=mmm



    def get_figure(self,true_keys, path_head,model_suf,path,vertices_pl,intensity, spid, width, height,ffcouleur ):   

        id_path=f'{true_keys}_save_save' if path_head=='true' else f'{path_head}_{model_suf}_{path}'
        id_pathss=f'{path_head}_{model_suf}_{path}_{true_keys}'
        spine_path=self.path_file[id_path] 
        # shaft_path=self.path_mapping[self.name_spine][path]  
        if (spid == 0) : 
            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl,
                                                    faces=self.faces ,
                                                    intensity=intensity,
                                                    width=width, 
                                                    height=height,
                                                    colorscale='purd')   
            self.scatter=[ 
                    hf.plotly_scatter(points=vertices_pl ,
                                      color='red',
                                      size=1.07,
                                      opacity=.8,
                                      name='Approx'),   
                    ]
            self.layout = go.Layout(width=width, 
                            height=height,
                            title=f'Dendrite', 
                            ) 
 
        elif spid==1: 
            self.spine_index =spine_index = np.loadtxt(os.path.join(spine_path,self.txt_shaft_index), dtype=int) 
            self.faces_index =faces_index = np.loadtxt(os.path.join(spine_path,self.txt_shaft_faces),dtype=int)
            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl[spine_index],
                                                    faces=faces_index,
                                                    intensity=intensity[spine_index],
                                                    width=width, 
                                                    height=height,
                                                    colorscale='purd') 

            self.scatter=[ 
                    hf.plotly_scatter(points=vertices_pl[spine_index],
                                      color='red',
                                      size=.7,
                                      opacity=.6,
                                      name='Approx'),   
                    ]  
            self.layout = go.Layout(width=width, 
                            height=height,  
                            )

        else:
            color='red'  
            clustss = spid - 2
            clustsss = spid - 2
            if id_pathss in self.iou_tr: 
                self.clusts=clustss=self.iou_tr[id_pathss].count_appr_tmp[clustss]
                
            # scatter=[] 
            self.spine_index = spine_index = np.loadtxt(os.path.join(spine_path, f'{self.name_spine}_{self.name_index}_{clustss}.txt'),dtype=int)
            self.spine_faces =spine_faces  = np.loadtxt(os.path.join(spine_path, f'{self.name_spine}_{self.name_faces}_{clustss}.txt'),dtype=int)
            print(os.path.join(spine_path, f'{self.name_spine}_{self.name_index}_{clustss}.txt') )

            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl[spine_index],
                                                    faces=spine_faces,
                                                    intensity=intensity[spine_index],
                                                    width=width, height=height,
                                                    colorscale='purd') 
            if id_pathss in self.iou_tr:
                self.scatter= self.iou_tr[id_pathss].get_graph(vertices_0=vertices_pl,index=clustsss)
                rate,rate_un=self.iou_count[id_path][clustsss,-2:]
                # clustss=int(self.iou_count[path][clustss,0])
                self.clusts=clustss=self.iou_tr[id_pathss].count_appr_tmp[clustsss]
                
                
                self.layout = go.Layout(width=width, 
                                height=height,
                                title=f'Dendrite {self.dend_name} Spine <br>UOI         : {rate:.2f}<br>UOI union: {rate_un:.2f}',  
                                )   
            else:
                self.scatter=[ 
                        hf.plotly_scatter(points=vertices_pl[spine_index],
                                        color='red',
                                        size=.7,
                                        opacity=.6,
                                        name='Approx'),   
                        ]  
                self.layout = go.Layout(width=width, 
                                height=height,  
                                )

        self.scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            bgcolor=ffcouleur 
        )



    def get_metric(self,mode,metric,width=800,height=800,nbinsx=10000,opacity=1.,color='blue',name=None): 
        self.scatter_metric,self.layout_metric=hp.Plotly_histogram_return(data=metric,
                                                                        nbinsx=nbinsx,
                                                                        title=mode,
                                                                        xtitle=self.metric_mapping['xtitle'],
                                                                        ytitle='Count',
                                                                        width=width, 
                                                                        height=height,
                                                                        opacity=opacity,
                                                                        color=color,
                                                                        name=name,)








    # def update_output(mode,dendd, clusts,   metric, intensity_type, width, height, radius_level, radius_level_max,uoi_per,templ):
    def Get_output(self, path_head,model_suf,path,mode,dendd, clusts,    intensity_type, width, height ,templ,nbin,index=None,get_return=True,hide_button_tf=True): 
        path_train=self.path_train
        true_keys='true_0'
        # model_suf=self.model_sufix if path_head == 'pinn' else 'save'  
        id_path=f'{true_keys}_save_save' if path_head=='true' else f'{path_head}_{model_suf}_{path}'
        id_pathss=f'{path_head}_{model_suf}_{path}_{true_keys}'
        self.get_data(
                    dend_data=self.dend_data,
                    model_sufix=model_suf,
                    data_studied=self.data_studied,
                    index=index,
                    ) 
        print('---------',true_keys,path,mode,dendd,path_head,'---inte',intensity_type,'===',model_suf,self.model_sufix)
        print('---------',path_train['dest_spine_path'],path_train['dest_spine_path_pre'],self.path_file_sub[intensity_type][id_path] )
        self.spine_path=spine_path=self.path_file_sub[self.inten_file_sub[0]][id_path] 
        dend_name=self.dend_name 
        bcouleur=self.bcouleur
        fsize=self.fsize 
        ppp= path.split('_')
        
        path_init=f'{ppp[0]}'
        ppp=ppp[1:]
        for pp in ppp[:-1]:
            path_init=f'{path_init}_{pp}'
        print('0000000',path_init,ppp)
        if templ=="plotly_dark":
            fcouleur='white'
            ffcouleur='black'#'black'
        else:
            fcouleur='black'
            ffcouleur='white'
        akp=aka_plot(tcouleur=templ,
                    bcouleur=bcouleur,
                    fcouleur=fcouleur,
                    fsize=fsize)
 
        vertices_pl= self.vertices_0 if dendd=='smooth' else self.vertices_00 

        intensity=None
        if intensity_type in self.inten_file_sub:  
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
            # print('spine--------',intensity_path)
        elif intensity_type in self.inten_file:
            intensity_path =self.path_file_sub[intensity_type][id_path] 
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
            # print('spine--------',intensity_path) 
        elif intensity_type in self.inten_pca:
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
            # print('spine--------',intensity_path) 
        elif intensity_type in self.inten_file_train:
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # print('spine--------',intensity_path) 
            # intensity=np.loadtxt(intensity_path, dtype=float)

        elif intensity_type in self.inten_file_model_head_neck:
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # print('spine--------',intensity_path) 
            # intensity=np.loadtxt(intensity_path, dtype=float)


        elif intensity_type in self.base_features_dict.keys():
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
 
        intensity= None 
        if os.path.exists(intensity_path):
            intensity=np.loadtxt(intensity_path, dtype=float)
            print('spine--------',intensity_path)
        else:
             print('Intensity DOESNT EXIST --------',intensity_path)
            # intensity_path =self.path_file_sub[intensity_type]['true_save_save'] 
        # elif intensity_type in ['spine_body','head_neck_body']: 
        #     intensity_path=self.inten[intensity_type] 
        # else:
        #     intensity_path = self.dend_mapping[dendd]["intensity"][intensity_type] 

        if intensity_type in ["gauss_curv_init","mean_curv_init"]:
            vertices_pl=self.vertices_00
        elif intensity_type in ["gauss_curv_smooth","mean_curv_smooth"]:
            vertices_pl=self.vertices_0

 
        figure=go.Figure()
        self.get_figure(true_keys,path_head,model_suf, path,vertices_pl,intensity, clusts, width, height,ffcouleur=ffcouleur  )
 
        

        if mode=='algorithm':
            figure=  self.figure_3d 
        elif mode=='comparison':
            figure=akp.Plotly_Figure(data=self.scatter, layout=self.layout)
            figure.update_layout(scene=self.scene)  
        elif mode=='skeleton':
            if self.plot_data_center_curv is not None: 
                *pathc, last = path.split('_')
                ppath=self.path_switch[0] if last=='pre' else 'dest_spine_path'
                scatterr=self.plot_data_center_curv[id_path][clusts][0:1]
                skl_path=os.path.join(self.file_path_feat, self.txt_skl_vectices)
                if os.path.exists(skl_path): 
                    scatterr.append(hf.plotly_scatter(points=np.loadtxt(skl_path,dtype=float), color='yellow', size=5.3, name='skeleton init.',opacity=0.5))
                for val in self.plot_data_center_curv[id_path][clusts][1:]:
                    scatterr.append(val)
                figure=akp.Plotly_Figure(data= scatterr, layout=self.layout)
                figure.update_layout(scene=self.scene)   
        elif mode in ['heatmap_cylinder','heatmap_cylinder_surface']:
            if self.plot_data_cylinder_heatmap is not None: 
                *pathc, last = path.split('_')
                ppath=self.path_switch[0] if last=='pre' else 'dest_spine_path'
                if mode =='heatmap_cylinder':
                    data=[self.plot_data_cylinder_heatmap[id_path].density_heatmap,self.plot_data_cylinder_heatmap[id_path].density_heatmap_points]
                    figure=akp.Plotly_Figure(data= data, layout=self.layout) 
                    figure.update_layout(
                        xaxis=dict(showgrid=False),  
                        yaxis=dict(showgrid=False)  
                    ) 
                elif mode =='heatmap_cylinder_surface':
                    figure=akp.Plotly_Figure(data= self.plot_data_cylinder_heatmap[id_path].density_heatmap_surface, layout=self.layout)
                figure.update_layout(scene=self.scene)
        elif mode=='IOU':
            # if self.plot_data_iou  is not None: 
            #     figure=akp.Plotly_Figure(data= self.plot_data_iou, layout=self.layout)
            #     figure.update_layout(scene=self.scene)  
            if  len(self.plot_data_iou_dic)>0:  

                figure=akp.Plotly_Figure(data= self.plot_data_iou_dic[id_pathss] , layout=self.layout)
                figure.update_layout(scene=self.scene) 




        elif mode=='accuracy':
            title='Accuracy'
            width=width
            height=height
            subplot_titles=('IOU Single','IOU Union')
            figure=akp.Plotly_Figure_Sub( subplot_titles,rows=2, cols=1, 
                                            shared_xaxes=False,
                                            shared_yaxes=False)
            if len(self.metric_total_dic )>0: 
                for ii,typ in enumerate(['single','union']):
                    # df_union=pd.read_csv(os.path.join(self.path_file[f'result_appr'],f'metric_{typ}.csv')).sort_values(by=f'single_{true_keys}', ascending=True)
                    df_union=pd.read_csv(os.path.join(self.path_file[f'result_appr'],f'metric_{typ}_{true_keys}.csv')).sort_values(by=f'accuracy', ascending=True)
                    xlabels=df_union.columns[1:]
                    ylabels=df_union.iloc[:,0]

                    df_union=df_union[df_union.columns[1:]]
                    cm=np.array(df_union.select_dtypes(include=[np.number]))
                    cm= np.round(cm*1e3)/1e3

                    scatter=go.Heatmap(
                                        z=cm,
                                        x=xlabels,  
                                        y=ylabels,  
                                        colorscale='Blues',
                                        text=cm,  # Show numbers in cells
                                        texttemplate="%{text}",  # Format as numbers
                                        # hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                                        showscale=False ,
                                        textfont=dict(size=18),
                                    ) 

                    # layout = go.Layout(
                    #     # title=title, 
                    #     width=width, 
                    #     height=height, 
                    #     xaxis=dict(title="Metric"),
                    #     yaxis=dict(title="Models"),
                    #         font=dict(size=14)
                    # )
                    figure.add_trace( scatter, row=ii+1, col=1)
                    # figure.update_layout(layout )
                # figure=akp.Plotly_Figure(data= scatter , layout=layout)
                # figure.update_layout(scene=self.scene)
            # xtitle="Metric" 
            # ytitle="Model"
            # figure.update_layout(
            #     xaxis=dict(title=xtitle),
            #     xaxis2=dict(title=xtitle),
            #     yaxis=dict(title=ytitle),
            #     yaxis2=dict(title=ytitle)
            # )
            figure.update_layout(height=2.5*height, width=width)

            figure.update_layout(
                xaxis=dict(
                    side="top",          
                    ticks="outside",     # Optional: ticks outside the plot
                    showticklabels=True, # Ensure tick labels are visible 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score'],
                )
            )

            figure.update_layout(
                xaxis2=dict(
                    side="top",          
                    ticks="outside",     # Optional: ticks outside the plot
                    showticklabels=True, # Ensure tick labels are visible 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score'],
                )
            )
            for annotation in figure['layout']['annotations']:
                annotation['y'] += 0.03   

        elif mode in self.inten_file_model_head_neck:  
            self.layout = go.Layout(width=width, 
                            height=height,  )
            # ['model_hn_loss','model_hn_iou']
            if (mode == 'model_hn_loss' ) and len( self.scatter_loss_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data=self.scatter_loss_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene) 
            elif (mode == 'model_hn_iou' ) and len(self.scatter_iou_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data= self.scatter_iou_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene) 

            if (mode == 'model_sp_loss' ) and len( self.scatter_loss_spine_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data=self.scatter_loss_spine_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene) 
            elif (mode == 'model_sp_iou' ) and len(self.scatter_iou_spine_dic[id_path])>0: 
                print('----=====----','9088==============---========================================',len(self.scatter_iou_spine_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_iou_spine_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
                          
                if path_head.startswith(('cML','CML','ML')): 
                    figure.update_layout(
                        xaxis=dict(
                            side="bottom",  # keep the bottom axis if you want
                            showticklabels=False  # hide bottom labels if you only want top
                        ),
                        xaxis2=dict(
                            side="top",
                            overlaying="x",        # align with the bottom axis
                            ticks="outside",
                            showticklabels=True,
                            tickmode="array",
                            tickvals=[0, 1, 2, 3],  # positions of your metrics
                            ticktext=['Accuracy','Precision','Recall','F1 Score'],
                        )
                    )

                    # Make sure your trace uses the top axis
                    for trace in figure.data:
                        trace.update(xaxis="x2")

                                    

            elif mode == 'model_shap':
                # figure=self.model_shap_dic[id_path]
                # print('figure',figure)
                figure=akp.Plotly_Figure(data= self.model_shap_dic[id_path]['data'], layout=self.model_shap_dic[id_path]['layout'])
                figure.update_layout(scene=self.scene) 

   

        elif mode in ['heatmap_iou','heatmap_iou_union','histogram_iou']: 
            spine_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( spine_path_save,f'iou_{true_keys}.csv') 
            if os.path.exists(metric_path):
                df = pd.read_csv(metric_path) 
                sze_checks_0 ,sze_check ,sze_check_un=df['id'],df['iou_single'],df['iou_union']
                # nhf=df['id_true']>0
                # sze_checks_0 ,sze_check ,sze_check_un=sze_checks_0[nhf] ,sze_check[nhf] ,sze_check_un[nhf]
                metric_name_path=df.columns
                iou_dict={}
                get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict=iou_dict, iou_per=70,labels = ['False', 'True'],nbinsx=nbin,)
                if mode =='heatmap_iou':
                    accuracy, precision, recall, f1_score=iou_dict['single']['metrics'].values()
                    metrics_text = (f'Accuracy: {accuracy:.3f}   '
                                    f'Precision: {precision:.3f}<br>'
                                    f'Recall    : {recall:.3f}   '
                                    f'F1 Score: {f1_score:.3f}')
                    figure=akp.Plotly_Figure(data=iou_dict['single']['heatmap_cm'], layout=self.layout)
                    figure.update_layout(scene=self.scene)

                    figure.update_layout(
                        title={
                            'text': metrics_text,
                            'x': 0.5,
                            'y': 0.92,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        }
                    )
                    figure.update_layout(
                        xaxis=dict(
                            # title=dict(text="Iterations", font=dict(size=16)),
                            tickmode="array",
                            tickvals=['accuracy','precision','recall','f1_score'],
                            ticktext=['Accuracy','Precision','Recall','F1 Score'],
                            # tickfont=dict(size=14, family="Arial", color="black")
                        ),
                        # yaxis=dict(
                        #     title=dict(text="IoU", font=dict(size=16)),
                        #     tickmode="linear",
                        #     dtick=0.1,  # step size
                        #     tickfont=dict(size=14, family="Arial", color="black")
                        # )
                    )
                elif mode =='heatmap_iou_union':
                    accuracy, precision, recall, f1_score=iou_dict['union']['metrics'].values()
                    metrics_text = (f'Accuracy: {accuracy:.3f}   '
                                    f'Precision: {precision:.3f}<br>'
                                    f'Recall    : {recall:.3f}   '
                                    f'F1 Score: {f1_score:.3f}')
                    figure=akp.Plotly_Figure(data=iou_dict['union']['heatmap_cm'], layout=self.layout)
                    figure.update_layout(scene=self.scene)

                    figure.update_layout(
                        title={
                            'text': metrics_text,
                            'x': 0.5,
                            'y': 0.92,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        } 
                    )
                    fig.update_layout(
                        xaxis=dict(
                            # title=dict(text="Iterations", font=dict(size=16)),
                            tickmode="array",
                            tickvals=['accuracy','precision','recall','f1_score'],
                            ticktext=['Accuracy','Precision','Recall','F1 Score'],
                            # tickfont=dict(size=14, family="Arial", color="black")
                        ), 
                    )
                elif mode =='histogram_iou':

                    figure=akp.Plotly_Figure(data= iou_dict['single']['histogram'] , layout=self.layout)
                    figure.add_annotation(
                        text=f'Total Count: {len(sze_checks_0)}',
                        xref='paper',
                        yref='paper',
                        x=0.98,
                        y=0.95,
                        showarrow=False,
                        font=dict(size=22)
                    )
                    figure.update_layout(scene=self.scene)  

                    figure.add_trace(iou_dict['union']['histogram'] )
                    figure.add_annotation(
                        text=f'Total Count Union.: {len(sze_checks_0)}',
                        xref='paper',
                        yref='paper',    
                        x=0.98,
                        y=0.88,
                        showarrow=False,
                        font=dict(size=22)
                    )

                    figure.update_layout(
                        barmode='overlay',  
                        title=mode,
                        xaxis_title='Values',
                        yaxis_title='Iou'
                    )
 


        elif mode in self.metric_mapping_combine['name']:  
            dend_names=self.dend_names 
            spine_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( spine_path_save,'metrics.csv') 
            if os.path.exists(metric_path):  
                df = pd.read_csv(metric_path)   
                
                headd,neckk,lengthh=df[self.metrics_combine[mode]['key'][0]],df[self.metrics_combine[mode]['key'][1]],df[self.metrics_combine[mode]['key'][2]]
                fig=hp.plotly_metric(headd,neckk,lengthh,height=height,width=width,title_size=22,colorscale='Blues',
                                         xtitle=add_unit(self.metrics_combine[mode]['label'][0]),
                                         ytitle=add_unit(self.metrics_combine[mode]['label'][1]),
                                         ztitle=add_unit(self.metrics_combine[mode]['label'][2]),
                                         marginal='box')
                figure=akp.Plotly_Figure(data= fig.data, layout=fig.layout) 
                # if mode=='vol_area_length_spine':
                m, b = np.polyfit(headd, neckk, 1) 
                y_pred = m * headd + b 
                y_mean = np.mean(neckk)
                ss_res = np.sum((neckk - y_pred)**2)
                ss_tot = np.sum((neckk - y_mean)**2)
                r2 = 1 - (ss_res / ss_tot)
                n=len(neckk) 
                reg_x = np.linspace(headd.min(), headd.max()*1.2, 100)
                reg_y = m * reg_x + b 
                # reg_y = np.clip(reg_y, neckk.min(), neckk.max()*1.2) 
                figure.add_trace(go.Scatter(x=reg_x, y=reg_y, mode="lines", name="Linear fit"))
                equation_text = f"y = {m:.3f}x + {b:.3f}<br>R<sup>2</sup> = {r2:.3f}<br>n = {n}" 
                # equation_text = ( 
                #     f"y   = {m:.3f}x + {b:.3f}\n"
                #     f"R<sup>2</sup> = {r2:.3f}\n"
                #     f"n   = {n}" 
                # ) 
                figure.add_annotation(
                    x=headd.min() + 0.01*(headd.max() - headd.min()),  
                    y=neckk.min() + 0.9*(neckk.max() - neckk.min()),   
                    text=equation_text,
                    showarrow=False,
                    font=dict(  size=20, color="white"),
                    bgcolor="black",
                    bordercolor="black", 
                    xanchor="left",    
                    align="left"   ,
                )
  
            #     figure.update_layout(
            #     # title=title,
            #     xaxis2_title='Counts',#self.metrics_combine[mode]['label'][0],
            #     yaxis2_title='Counts',self.metrics_combine[mode]['label'][1], 
            # )                
                figure.update_layout(
                                xaxis1=dict(
                                    title=dict(
                                        text=add_unit(self.metrics_combine[mode]['label'][0]),
                                        font=dict(size=20)
                                    ),
                                    # range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis1=dict(
                                    title=dict(
                                        text=add_unit(self.metrics_combine[mode]['label'][1]),
                                        font=dict(size=20),
                                    ),
                                    # range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  

                                
                                xaxis2=dict(
                                    title=dict(
                                        text="Counts",
                                        font=dict(size=20)
                                    ),
                                    # range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis2=dict(
                                    title=dict(
                                        text="Counts",
                                        font=dict(size=20),
                                    ),
                                    # range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  



                                
                                xaxis3=dict(
                                    title=dict(
                                        text="x3",
                                        font=dict(size=20)
                                    ),
                                    range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis3=dict(
                                    title=dict(
                                        text="y3",
                                        font=dict(size=20),
                                    ),
                                    range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  
            )

 
        elif mode in self.metric_mapping['name']:  
            dend_names=self.dend_names 
            spine_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( spine_path_save,'metrics.csv') 
            # figure=akp.Plotly_Figure(data=[],layout=None)
            # figure.update_layout(scene=self.scene)  

            scolor=['blue','red','yellow','purple','green']
            data_path=self.path_file['result_true']

            df_save={}
            name=name_approx='Approx'
            df_save[name]={}
            xmax=0
            xmin=-np.inf
            df_save[name]['ann']=['blue',.6,0.98,0.95]
            df_save[name]['path']=metric_path
            if os.path.exists(df_save[name]['path']):  
                df_save[name]['df']=df = pd.read_csv(df_save[name]['path'])   
                # df=df[df[df.columns[0]].astype(str).str.startswith(tuple(dend_names))] 
                df_save[name]['df']=  df
                if mode in df.columns:  
                    xmax=max(xmax,max(df[mode]))
                    xmin=min(xmin,min(df[mode]))
            namesan=[]
            names=[]
            for ii in range(1,5):
                data_name=f'spine_head_analysis.trial_{ii}.dat' 
                df_save[name]['ann']=[scolor[ii],.5,0.98,0.95-(ii*0.07)/1.3] 
                if os.path.exists( os.path.join(data_path,data_name)):
                    name=f'Annot_{ii}'
                    names.append(name)
                    namesan.append(name)
                    df_save[name]={}
                    df_save[name]['path']=hff.get_conversion_file(data_path=data_path,data_name=data_name)
                    df = pd.read_csv(df_save[name]['path'])  
                    # df=df[df[df.columns[0]].astype(str).str.startswith(tuple(dend_names))] 
                    df_save[name]['df']=  df
                    if mode in df.columns:  
                        xmax=max(xmax,max(df[mode]))
                        xmin=min(xmin,min(df[mode]))
            names.append(name_approx)
            title='KL Divergence D_KL(P||Q)'
            xaxis_title,yaxis_title='P','Q'


            if len(namesan)>0 :
                subplot_titles=('Histogram','KL Divergence D_KL(P||Q)')
                figure=akp.Plotly_Figure_Sub( subplot_titles,rows=2, cols=1, )
            else:
                figure= akp.Plotly_Figure(data=[],layout=None)

            for name in names:
                df = df_save[name]['df']
                color,opacity,x_ann,y_ann=df_save[name]['ann']  
                if os.path.exists(df_save[name]['path']):  
                    if mode in df.columns:  
                        scatter_metric,layout_metric=hp.Plotly_histogram_return(data=np.abs(df[mode]),
                                                                                nbinsx=nbin,
                                                                                title=mode,
                                                                                xtitle='Length',
                                                                                ytitle='Count',
                                                                                xrange=[xmin,xmax],
                                                                                yrange=None,
                                                                                width=width, 
                                                                                height=height,
                                                                                opacity=opacity,
                                                                                color=color,
                                                                                name=name,)
                        if len(namesan)>0 :
                            figure.add_trace( scatter_metric, row=1, col=1)
                        else:
                            figure.add_trace( scatter_metric )
                        cname=f'Count {name}  ' if name=='Approx' else f'Count {name}' 
                        figure.add_annotation(
                            text=f'{cname}: {len(df[mode])}',
                            xref='paper',
                            yref='paper',    
                            x=x_ann,
                            y=y_ann,
                            showarrow=False,
                            font=dict(size=22)
                        )  

                    figure.update_layout(layout_metric)
                        
                    if len(namesan)>0 :
                        scatter,layout=compute_kl(df_save,names,mode,width,height,bins =nbin)
                        figure.add_trace( scatter, row=2, col=1)
                        figure.update_layout(layout )
                        figure.update_layout(
                        # title=title,
                        xaxis2_title=xaxis_title,
                        yaxis2_title=yaxis_title,
                        # xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
                        # yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels), 
                        )
                        figure.update_layout(height=1.3*height, width=width)
                    else:
                        figure.update_layout(height=height, width=width)

 

        else:
            figure=go.Figure() 
        self.figure=figure
        hp.hide_button(figure,hide_button_tf) 
        if get_return:
            return dcc.Graph(figure=figure) ,f'Dendrite Name: {dend_name}'
        else:
            self.figure=figure
            self.vertices_pl,self.intensity, self.clusts=vertices_pl,intensity, clusts
            self.akp=akp

    def Get_children(self):
        dropdown_true_keys=self.dropdown_true_keys
        dropdown_path_head=self.dropdown_path_head
        dropdown_model_suf=self.dropdown_model_suf
        dropdown_path=self.dropdown_path
        dropdown_mode=self.dropdown_mode
        dropdown_dend=self.dropdown_dend
        dropdown_intensity=self.dropdown_intensity
        dropdown_cluster=self.dropdown_cluster

        # dropdown_plot=self.dropdown_plot
        dropdown_template=self.dropdown_template
        dropdown_options_style=self.dropdown_options_style
        box_style=self.box_style
        # dend_name=self.dend_name
        # itera=self.itera

        return [ 
            # Additional Dropdown for Graph 2 
            html.Br(),
            dbc.Row([
                # Column 1: Dropdowns
                dbc.Col(
                    html.Div([
                        dcc.Dropdown(
                            id=         self.dropdown_index['id'],
                            options=    self.dropdown_index['option'],
                            value=      self.dropdown_index['value'],
                            placeholder=self.dropdown_index['placeholder'],
                            style=      box_style
                        ),
                        # html.Br(),
                        # dcc.Dropdown(
                        #     id=         dropdown_true_keys['id'],
                        #     options=    dropdown_true_keys['option'],
                        #     value=      dropdown_true_keys['value'],
                        #     placeholder=dropdown_true_keys['placeholder'],
                        #     style=      box_style
                        # ),
                        html.Br(),
                        dcc.Dropdown(
                            id=         dropdown_path_head['id'],
                            options=    dropdown_path_head['option'],
                            value=      dropdown_path_head['value'],
                            placeholder=dropdown_path_head['placeholder'],
                            style=      box_style
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id=         dropdown_model_suf['id'],
                            options=    dropdown_model_suf['option'],
                            value=      dropdown_model_suf['value'],
                            placeholder=dropdown_model_suf['placeholder'],
                            style=      box_style
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id=         dropdown_path['id'],
                            options=    dropdown_path['option'],
                            value=      dropdown_path['value'],
                            placeholder=dropdown_path['placeholder'],
                            style=      box_style
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id=         dropdown_mode['id'],
                            options=    dropdown_mode['option'],
                            value=      dropdown_mode['value'],
                            placeholder=dropdown_mode['placeholder'],
                            style=      box_style
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id=         dropdown_intensity['id'],
                            options=    dropdown_intensity['option'],
                            value=      dropdown_intensity['value'],
                            placeholder=dropdown_intensity['placeholder'],
                            style=      box_style
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id=         dropdown_cluster['id'],
                            options=    dropdown_cluster['option'],
                            value=      dropdown_cluster['value'],
                            placeholder=dropdown_cluster['placeholder'],
                            style=      box_style
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id=         dropdown_dend['id'],
                            options=    dropdown_dend['option'],
                            value=      dropdown_dend['value'],
                            placeholder=dropdown_dend['placeholder'],
                            style=      box_style
                        ), 
                        html.Br(),
                        dcc.Dropdown(
                                    id=         dropdown_template['id'],
                                    options=    dropdown_template['option'],
                                    value=      dropdown_template['value'],
                                    placeholder=dropdown_template['placeholder'],
                                    style=      box_style
                        ),
                        html.Br(),
                        html.Label('Histogram Bin Count:', style=dropdown_options_style),
                        dcc.Slider(
                            id=self.hist_slider['id'],
                            min=self.hist_slider['min'],
                            max=self.hist_slider['max'],
                            step=self.hist_slider['step'],
                            value=self.hist_slider['value'],
                            marks=self.hist_slider['marks'],
                        ), 
                        # html.Br(),
                        # html.Label('Prediction UOI %:', style=dropdown_options_style),
                        # dcc.Slider(
                        #     id=self.uoi_slider['id'],
                        #     min=self.uoi_slider['min'],
                        #     max=self.uoi_slider['max'],
                        #     step=self.uoi_slider['step'],
                        #     value=self.uoi_slider['value'],
                        #     marks=self.uoi_slider['marks'],
                        # ), 
                        html.Br(),
                        html.Label('Graph Width:', style=dropdown_options_style),
                        dcc.Slider(
                            id=self.width_slider['id'],
                            min=self.width_slider['min'],
                            max=self.width_slider['max'],
                            step=self.width_slider['step'],
                            value=self.width_slider['value'],
                            marks=self.width_slider['marks'],
                        ),
                        html.Br(),
                        html.Label('Graph Height:', style=dropdown_options_style),
                        dcc.Slider(
                            id=self.height_slider['id'],
                            min=self.height_slider['min'],
                            max=self.height_slider['max'],
                            step=self.height_slider['step'],
                            value=self.height_slider['value'],
                            marks=self.height_slider['marks'],
                        ),
                        html.Br(),
                    ]),
                    style={'flex': '0 0 20%'}  # Set the width of the column using flex property
                ),
                # Column 2: Graph Output
                dbc.Col(
                    html.Div([
                        html.Br(),
                        # Graph 1 Output
                        html.Div(id=self.output_graph_1['id'], 
                                 style=self.output_graph_1['style']),
                        html.Br(),
                        html.Div(id=self.output_text_1['id'], 
                                 style=self.output_graph_1['style']
                                ),
                        html.Br(),
                    ]),
                    style={'flex': '0 0 70%'}  # Set the width of the column using flex property
                ),
            ], style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'margin-top': '20px',
                'margin-bottom': '20px'
            }),
            # Text Output
            html.Br(),
            html.Br(), 
        ]



 