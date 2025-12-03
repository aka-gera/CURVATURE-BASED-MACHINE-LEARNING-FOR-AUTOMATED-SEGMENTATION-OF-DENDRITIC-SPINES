import os
import numpy as np
import dend_fun_0.help_fun as hf
from dend_fun_0.get_path import get_param,get_name
from dend_fun_0.help_funn import get_color,format_array,loadtxt,loadtxt_count
import pickle

clor=get_color()


def process_appr_vs_true(mp, count_appr, appr_index_data, true_index_data,iou_thre=.2):  
    for ii,appr_index in appr_index_data.items():   
        for jj, true_index in true_index_data.items():
            inter = appr_index.intersection(true_index)
            unio = appr_index.union(true_index) 
            iou=len(inter) / len(unio)
            if iou>=iou_thre:
                if jj not in mp[ii]['true']: 
                    mp[ii]['true'].append(jj)
                if ii not in mp[ii]['appr']:  
                    mp[ii]['appr'].append(ii)
                    
                new_indices = unio.difference(mp[ii]['index'])   
                mp[ii]['index'].extend(new_indices)  
                mp[ii]['iou'].append(iou)
        if len(mp[ii]['iou'])==0:
            # print(ii,'fail',mp[ii]['iou'],len(inter))
            mp[ii]['iou'].append(0)
            mp[ii]['true'].append(-1)
    return mp


def process_appr_vs_appr(mp, count_appr, appr_index_data,iou_thre=.2):  
    for ij in count_appr:
        for ii in count_appr:
            if ij != ii:
                appr_index = appr_index_data[ii]  
                inter = appr_index.intersection(set(mp[ij]['index']))
                unio = appr_index.union(set(mp[ij]['index']))
                iou=len(inter) / len(unio)
                if iou>iou_thre:
                    if ii not in mp[ij]['appr']:  
                        mp[ij]['appr'].append(ii)
    return mp



def process_true_vs_appr(mp, count_appr,count_true, true_index_data,iou_thre=.2):  
    for ij in count_appr:
        for ii,true_index in true_index_data.items(): 
            # true_index = true_index_data[ii]  
            inter = true_index.intersection(set(mp[ij]['index']))
            unio = true_index.union(set(mp[ij]['index']))
            iou=len(inter) / len(unio)
            if iou>iou_thre:
                if ii not in mp[ij]['true']: 
                    mp[ij]['true'].append(ii)  
                new_indices = unio.difference(mp[ij]['index'])   
                mp[ij]['index'].extend(new_indices)  
                # mp[ii]['iou'].append(len(inter) / len(unio))
    return mp


class iou_train(get_name,get_param):
    def __init__(self, 
                 path_true,
                 path_appr,
                 save=False,
                 mp = None,
                 iou_thre=.2,
                 iou_dict=None,
                 dend_first_name=None,
                 file_path=None,
                 path_result=None,
                        ):
        get_name.__init__(self)  
        get_param.__init__(self)  
        self.save=save
        self.iou_thre=iou_thre
        self.iou_dict=iou_dict
        self.dend_first_name=dend_first_name
        self.mp = mp
        self.file_path=file_path
        self.path_result=path_result
        self.appr_index_data,self.true_index_data = {},{}

 
        # count= loadtxt( os.path.join(path_appr,self.txt_spine_count),dtype=int) 
        # count=format_array(count)
        
        count= loadtxt_count(os.path.join(path_appr,self.txt_spine_count))
        mmm=count.ndim
        count=count if mmm==2 else count.reshape(-1,1)
        if len(count)>0:
            self.count_appr={
                ik:f'{count[idx,0]}_{count[idx,1]}' if mmm==2 else f'{count[idx,0]}' for ik,idx in enumerate(range(count.shape[0])) 
            }  
        else:
            self.count_appr={  
            }  


        # for ik,idx in enumerate(range(count.shape[0])): 
        #     ii=count[idx,0]
        #     # ikk= ik  if mmm==2 else ii 
        #     self.count_appr[ik]=f'{ii}_{count[idx,1]}' if mmm==2 else f'{ii}'
 
        # self.count_appr=count_appr=np.loadtxt(os.path.join(path_appr,self.txt_spine_count),dtype=int)


        self.shaft_path_appr=path_appr
        self.count_appr_tmp=self.count_appr 
        # print(self.count_appr_tmp,os.path.join(shaft_path_appr,self.txt_spine_count))
        # for ii in self.count_appr_tmp:
            # print(os.path.join(shaft_path_appr, f'{self.name_spine_index}_{ii}.txt'))
        if len(self.count_appr_tmp)>1:

            self.appr_index_data = {
                ii: set(np.loadtxt(os.path.join(path_appr, f'{self.name_spine_index}_{val}.txt'), dtype=int))
                for ii,val in self.count_appr_tmp.items() if os.path.exists(os.path.join(path_appr, f'{self.name_spine_index}_{val}.txt'))
            } 
        self.count_true= loadtxt(os.path.join(path_true,self.txt_spine_count),dtype=int) 
        if  self.count_true.ndim>0: 
            self.true_index_data = {
                jj: set(np.loadtxt(os.path.join(path_true, f'{self.name_spine_index}_{jj}.txt'), dtype=int))
                for ii,jj in enumerate(self.count_true)
            }


        
    def get_mapping(self,save=None,iou_thre=None):
        save= save or self.save
        iou_thre=iou_thre or self.iou_thre
        mp,count_appr_tmp,count_true, appr_index_data, true_index_data=self.mp, self.count_appr_tmp,self.count_true, self.appr_index_data, self.true_index_data
        if mp is not None:
            if save:
                with open(os.path.join(self.shaft_path_appr,self.pkl_mp), "wb") as file:
                    pickle.dump(mp, file)
        else:
            mp = {
                ii: {
                    'true': [],
                    'appr': [ii],
                    'iou': [],
                    'iou_union':[],
                    'index': [],
                } for ii in self.count_appr_tmp
            }



        mp_results = process_appr_vs_true(mp, count_appr_tmp, appr_index_data, true_index_data,iou_thre)
        den=np.sum(np.array([[len(mp_results[ij]['appr']),len(mp_results[ij]['true'])] for ij in count_appr_tmp]),axis=0)

        chck=True
        while chck:
            mp_results = process_appr_vs_appr(mp_results, count_appr_tmp, appr_index_data,iou_thre)   
            mp_results = process_true_vs_appr(mp_results, count_appr_tmp, count_true, true_index_data,iou_thre)
            den_tmp=np.sum(np.array([[len(mp_results[ij]['appr']),len(mp_results[ij]['true'])] for ij in count_appr_tmp]),axis=0)
            print(den,den_tmp,'den_tmp',den_tmp.ndim)
            if den_tmp.ndim>0:
                if (den[0]==den_tmp[0]) and (den[1]==den_tmp[1]):
                    chck=False
                else:
                    den=den_tmp  
            else:
                chck=False
        self.mp=mp_results
        if save:
            with open(os.path.join(self.shaft_path_appr,self.pkl_mp), "wb") as file:
                pickle.dump(mp_results, file)



    def get_iou_save(self,save=None,iou_thre=.7):
        save= save or self.save
        if self.mp is None:
            self.get_mapping()
        mp_results,count_appr_tmp,count_true, appr_index_data, true_index_data=self.mp, self.count_appr_tmp,self.count_true, self.appr_index_data, self.true_index_data

        true_union_all=[]
        save_iou=[]
        for ii in count_appr_tmp:
            appr_union=[]
            for j,jj in enumerate(mp_results[ii]['appr']):
                if jj in appr_index_data:
                    appr_union.extend((appr_index_data[jj]))
            
            true_union=[]
            
            for j,jj in enumerate(mp_results[ii]['true']):
                #     print(true_index_data )
                if jj>=0: 
                    true_union.extend((true_index_data[jj])) 
                    true_union_all.append(jj)
            if len(true_union)>0:
                iou_union=len(set(appr_union).intersection(set(true_union)))/len(set(appr_union).union(set(true_union))) if (mp_results[ii]['true'][0]>=0) else 0
                # mp_results[ii]['iou_union'].append(iou_union)
                # for yty in mp_results[ii]['true']:
            else:
                iou_union=0
            mp_results[ii]['iou_union'].append(iou_union)
        true_union_all_remain=list(set(self.count_true)-set(true_union_all))
        ii=0
        save_iou=[]
        save_iou_name=[]
        for ii in count_appr_tmp:
            if (len(mp_results[ii]['iou'])>0) and (mp_results[ii]['true'][0] <0) :
                savd=[mp_results[ii]['appr'][0],mp_results[ii]['true'][0],1,1]
            elif (len(mp_results[ii]['true'])>0) and (len(mp_results[ii]['iou'])>0) and (len(mp_results[ii]['iou_union'])>0):
                # print(ii,f'{self.dend_first_name}_sy{ii}',count_appr_tmp,mp_results)
                savd=[mp_results[ii]['appr'][0],mp_results[ii]['true'][0],mp_results[ii]['iou'][0],mp_results[ii]['iou_union'][0]]
            else:
                savd=[0,0,0,0]
            save_iou.append(savd)
            self.iou_dict[f'{self.dend_first_name}_sy{ii}']=savd
        ij=1
        for jj in true_union_all_remain:
            save_iou.append([-1,jj,0,0])
            self.iou_dict[f'{self.dend_first_name}_sy{ii+ij}']=[-1,jj,0,0]
            ij+=1




        self.save_iou=np.array(save_iou)

        if save:
            np.savetxt(os.path.join(self.shaft_path_appr,self.txt_spine_iou),np.array(save_iou), fmt="%4f")
            # if self.iou_dict is not None:
                # m_name=[f'{self.dend_first_name}_sy{name}' for name in count_appr_tmp]
                # if len(save_iou)>0:
                #     for i,ii in enumerate(count_appr_tmp):  
                #         self.iou_dict[f'{self.dend_first_name}_sy{ii}']=save_iou[i]
            # with open(os.path.join( self.shaft_path_appr,'iou.csv'), 'w') as f: 
            #     f.write(',' + ','.join(m_name) + '\n') 
            #     for i, row_name in enumerate(count_appr_tmp):
            #         row_data = ','.join(map(str, save_iou[i]))
            #         row_name=f'{self.dend_first_name}_sy{i}'
            #         f.write(f'{row_name} ,{row_data}\n')


        else:
            self.save_iou=np.array(save_iou,ndmin=2)


    def get_iou_match(self,save=None,iou_thre=.7):
        if (self.file_path is not None)  :
            vertices_0       = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_0), dtype=float) 
            intensity=np.zeros_like(vertices_0[:,0])  
            intensity_true=-1*np.ones_like(vertices_0[:,0]) 
            ht=0
            ol_0=[]
            for val in self.appr_index_data.values():
                intensity[list(val)]=1
                ol_0.extend(list(val))
            ol_1=[]
            for val in self.true_index_data.values():
                intensity[list(val)]=2
                intensity_true[list(val)]=ht
                ol_1.extend(list(val))
                ht+=1
            # print('ooop----------',self.save_iou) 
            if len(self.save_iou)>0:
                vfv=self.save_iou[:,0][(self.save_iou[:,0]>=0)& (self.save_iou[:,1]>=0) &(self.save_iou[:,3]>=iou_thre)  ]
                print('ooop----------',vfv) 
                iidix=[]
                for idx in vfv:
                    iidix.extend(self.appr_index_data[idx])
                intensity[iidix]=3
            # intensity[list(set(ol_0).intersection(set(ol_1)))]=3
            np.savetxt(os.path.join(self.shaft_path_appr,f'spine_match.txt'),intensity, fmt="%d")
            np.savetxt(os.path.join(self.shaft_path_appr,f'spine_annot.txt'),intensity_true, fmt="%d")
        



    def get_graph(self,vertices_0,index,mp=None):
        if mp is None: 
            self.get_mapping()
            mp=self.mp
        count_appr_tmp,count_true, appr_index_data, true_index_data=self.count_appr_tmp,self.count_true, self.appr_index_data, self.true_index_data
        clor=get_color()



        scatter=[]  
        i=0
        # indexi=index+1
        for j,jj in enumerate(mp[index]['appr'] ):   
            ixi=i if i<len(clor) else len(clor)-1
            scatter.append(hf.plotly_scatter(points=vertices_0[list(appr_index_data[jj])], color=clor[ixi], size=3, name=f'appr_{jj}',opacity=0.5))
            i+=1
        for j,jj in enumerate(mp[index]['true']): 
            if mp[index]['true'][0]>=0:
                ixi=i if i<len(clor) else len(clor)-1
                scatter.append(hf.plotly_scatter(points=vertices_0[list(true_index_data[jj])], color=clor[ixi], size=1, name=f'true_{jj}',opacity=.9))
                i+=1
        return scatter