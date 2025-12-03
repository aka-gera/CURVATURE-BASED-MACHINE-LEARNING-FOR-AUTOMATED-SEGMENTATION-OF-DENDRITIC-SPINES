import numpy as np
import trimesh
import tensorflow as tf
from spektral.data import Graph
from spektral.data import Dataset
from spektral.layers import GCNConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
 

from keras.saving import register_keras_serializable
from tensorflow.keras import regularizers
DTYPE = tf.float32

'''
'''

@register_keras_serializable(package="Custom")
class PINN(Model):
    def __init__(self, Dtype=DTYPE, 
                 hidden_layers=4, 
                 neurons_per_layer=32, 
                 n_col=2,
                 activation_init="relu",
                 activation_hidden="relu",
                 activation_last="sigmoid",
                 l1=0.01,
                 l2=0.0000001,
                 **kwargs):
        super(PINN, self).__init__()
        self.hidden_layers = hidden_layers
        self.Dtype = Dtype
        k_regularizers=regularizers.l1_l2(l1=l1,l2=l2)
        self.gcn1 = GCNConv(neurons_per_layer,
                  kernel_regularizer=k_regularizers,
                    activation=activation_hidden)
        self.gcn2 = GCNConv(n_col,  activation=activation_last, dtype=Dtype) # Binary node classification


    def call(self, inputs):
        x, a, mask = inputs['x'], inputs['a'], inputs['mask']
        x = self.gcn1([x, a], mask=mask)
        x = self.gcn2([x, a], mask=mask)
        return x

 
# @register_keras_serializable(package="Custom")
# class PINN(Model):
#     def __init__(self, Dtype=DTYPE, 
#                  hidden_layers=2, 
#                  neurons_per_layer=16, 
#                  n_col=2,
#                  activation_hidden="relu",
#                 #  activation_last="softmax",
#                  activation_last="sigmoid",
#                  l1=0.01,
#                  l2=1e-7,
#                  **kwargs):
#         super().__init__( )
#         self.Dtype = Dtype
#         k_regularizers = regularizers.l1_l2(l1=l1, l2=l2)

#         # Build hidden GCN layers
#         self.hidden_layers_list = [
#             GCNConv(neurons_per_layer,
#                     # kernel_regularizer=k_regularizers,
#                     activation=activation_hidden)
#             for _ in range(hidden_layers)
#         ]
 
#         self.gcn_out = GCNConv(n_col,
#                             #    kernel_regularizer=k_regularizers,
#                                activation=activation_last,
#                                dtype=Dtype)

#     def call(self, inputs):
#         x, a, mask = inputs['x'], inputs['a'], inputs['mask']
#         for gcn in self.hidden_layers_list:
#             x = gcn([x, a], mask=mask)
#         x = self.gcn_out([x, a], mask=mask)
#         return x

#     # def call(self, inputs):
#     #     x, a = inputs['x'], inputs['a']
#     #     # no mask
#     #     for gcn in self.hidden_layers_list:
#     #         x = gcn([x, a])
#     #     x = self.gcn_out([x, a])
#     #     return x



# @register_keras_serializable(package="Custom")
# class PINN(Model):
#     def __init__(self, Dtype=DTYPE, 
#                  hidden_layers=4, 
#                  neurons_per_layer=100, 
#                  n_col=2,
#                  activation_init="relu",
#                  activation_hidden="relu",
#                  activation_last="sigmoid",
#                  l1=0.01,
#                  l2=0.0000001,
#                  **kwargs):
#         super(PINN, self).__init__()
#         self.hidden_layers = hidden_layers
#         self.Dtype = Dtype
#         k_regularizers=regularizers.l1_l2(l1=l1,l2=l2)
        
#         self.ld_init = Dense(neurons_per_layer, activation=activation_init, dtype=Dtype)
#         self.hidden_layers_list = [
#             Dense(neurons_per_layer, 
#                   activation=activation_hidden, 
#                   kernel_regularizer=k_regularizers,
#                   dtype=Dtype) 
#             for _ in range(hidden_layers)
#         ]
#         self.ld_last = Dense(n_col,  activation=activation_last, dtype=Dtype)

#     # def call(self, x):
#     #     x = self.ld_init(x)
#     #     for layer in self.hidden_layers_list:
#     #         x = layer(x)
#     #     x = self.ld_last(x)
#     #     return x

#     # def call(self, inputs):
#     #     x, a, mask = inputs['x'], inputs['a'], inputs['mask']
#     #     x = self.ld_init(x)
#     #     for layer in self.hidden_layers_list:
#     #         x = layer(x)
#     #     x = self.ld_last(x)
#     #     return x



#     def call(self, inputs):
#         x, a, mask = inputs['x'], inputs['a'], inputs['mask']
#         x = self.gcn1([x, a], mask=mask)
#         x = self.gcn2([x, a], mask=mask)
#         return x



def make_graph_input(X: np.ndarray, adj: np.ndarray):
    """
    Returns a dict ready to feed into model.call():
      - 'x':  (N, F) tf.float32
      - 'a':  (N, N) or (E, 2) tf.float32 adjacency
      - 'mask': (N,) all ones (or your real mask)
    """
    mask = np.ones((X.shape[0],), dtype=np.float32)
    return {
        'x': tf.convert_to_tensor(X, dtype=tf.float32),
        'a': tf.convert_to_tensor(adj, dtype=tf.float32),
        'mask': tf.convert_to_tensor(mask, dtype=tf.float32)
    }








class aka_train():
    def __init__(self) :
      pass

    # def get_grad_back_prop(self,fun,model): 

    #     with tf.GradientTape(persistent=True) as tape:  
    #         loss = fun.loss_fun(model) 
    #     grad = tape.gradient(loss,  model.trainable_variables) 
    #     del tape

    #     return loss, grad
    
    # @tf.function
    # def train_PINN(self,optimizer,fun,model):
    #     loss, grads = self.get_grad_back_prop(fun,model)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #     return loss persistent=True

    @tf.function
    def train_PINN(self,optimizer,fun,model):

        with tf.GradientTape() as tape:  
            loss = fun.loss_fun(model) 
        grads = tape.gradient(loss,  model.trainable_variables)  
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        norms = [tf.norm(g) for g in grads if g is not None]
        # tf.print("Loss", loss, "grad norms", norms )
        # tf.print(  model.trainable_variables)

        return loss 
    

class LOSS_simple():
    def __init__(self,rhs,curv): 
        pass
        self.rhs=rhs
        self.curv=curv
    
    def loss_fun(self,model):  
        loss=0
        for cu,rhs in zip(self.curv,self.rhs):
            for rhi in range(rhs.shape[1]):
                loss+= tf.reduce_mean(tf.square(model(cu)-rhs[:,rhi]))  
        return loss 
    

from spektral.data import Graph
import numpy as np
import tensorflow as tf
from spektral.layers import GCNConv
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from spektral.data import Graph, Dataset, Loader




# class MyLoader(Loader):
#     def collate(self, batch):
#         graph = batch[0]  # Since batch_size = 1
#         x = tf.convert_to_tensor(graph.x, dtype=tf.float32)
#         a = tf.convert_to_tensor(graph.a, dtype=tf.float32)
#         y = tf.convert_to_tensor(graph.y, dtype=tf.float32)
#         return (x, a), y

# def loader_to_tf_dataset(loader):
#     (x, a), y = next(iter(loader))
#     x_shape = (None, x.shape[-1])
#     a_shape = (None, None)
#     y_shape = y.shape
#     mask_shape = (None,)  # one value per node

#     def gen():
#         for (x, a), y in loader:
#             mask = np.ones(x.shape[0], dtype=np.float32)  # or your actual mask logic
#             yield ({'x': x, 'a': a, 'mask': mask}, y)

#     return tf.data.Dataset.from_generator(
#         gen,
#         output_signature=(
#             {
#                 'x': tf.TensorSpec(shape=x_shape, dtype=tf.float32),
#                 'a': tf.TensorSpec(shape=a_shape, dtype=tf.float32),
#                 'mask': tf.TensorSpec(shape=mask_shape, dtype=tf.float32),
#             },
#             # tf.TensorSpec(shape=y_shape, dtype=tf.float32)
#         )
#     )
    

'''

def loader_to_tf_dataset(loader):
    (x, a), y = next(iter(loader))
    x_shape = (None, x.shape[-1])
    a_shape = (None, None)
    y_shape = y.shape
    mask_shape = (None,)  # one value per node

    def gen():
        for (x, a), y in loader:
            mask = np.ones(x.shape[0], dtype=np.float32)  # or your actual mask logic
            yield ({'x': x, 'a': a, 'mask': mask}, y)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                'x': tf.TensorSpec(shape=x_shape, dtype=tf.float32),
                'a': tf.TensorSpec(shape=a_shape, dtype=tf.float32),
                'mask': tf.TensorSpec(shape=mask_shape, dtype=tf.float32),
            },
            tf.TensorSpec(shape=y_shape, dtype=tf.float32)
        )
    )'''



class MyLoader(Loader):
    def collate(self, batch):
        graph = batch[0]  # Since batch_size = 1
        x = tf.convert_to_tensor(graph.x, dtype=tf.float32)
        a = tf.convert_to_tensor(graph.a, dtype=tf.float32) 
        return x, a 


def loader_to_tf_dataset(loader):
    x, a = next(iter(loader))
    x_shape = (None, x.shape[-1])
    a_shape = (None, None) 
    mask_shape = (None,)  # one value per node

    def gen():
        for x, a in loader:
            mask = np.ones(x.shape[0], dtype=np.float32)  # or your actual mask logic
            yield ({'x': x, 'a': a, 'mask': mask} )

    return tf.data.Dataset.from_generator(
    gen,
    output_signature={
        'x': tf.TensorSpec(shape=x_shape, dtype=tf.float32),
        'a': tf.TensorSpec(shape=a_shape, dtype=tf.float32),
        'mask': tf.TensorSpec(shape=mask_shape, dtype=tf.float32),
    }
)

from scipy.sparse import coo_matrix

def get_dataset(X,adj):
    # print('oooo99',X.shape,adj  ) 
    # graph = Graph(x=X[adj[1]], a=adj[0], y=y[adj[0]])
    graph = Graph(x=X , a=adj   )
    
    class MyDataset(Dataset):
        def read(self):
            return [graph]

    dataset = MyDataset() 

    loader = MyLoader(dataset, batch_size=1, shuffle=False)
    return loader_to_tf_dataset(loader)


def run_model_on_graph(model, X, adj ):
    tf_dataset = get_dataset(X=X, adj=adj )
    inputs = next(iter(tf_dataset)) 
    return model(inputs)

def model_dataset( X, adj ): 
    print('XXXX',X.shape,adj.shape)
    return next(iter(get_dataset(X=X, adj=adj ))) 


class LOSS():
    def __init__(self, rhs, curv,adj=None, weight=None, loss_mode='mse' ): 
        """
        Args:
            rhs: Target matrix with binary values (0 or 1).
            curv: Input data to the model.
            loss_mode: Type of loss to use ('mse' for mean squared error, 'bce' for binary cross-entropy).
            weight_positive: Weight for the positive class (rhs == 1).
            weight_negative: Weight for the negative class (rhs == 0).
        """
        self.rhs = rhs
        self.curv = curv
        self.adj=adj
        self.loss_mode = loss_mode
        # if weight is None:
        #     siz=self.rhs[0].shape[1]
        #     weight=1/siz*np.ones(siz)
        self.weight = weight 
    
    def loss_fun(self, model,):  

        """
        Compute the loss for the model. *self.weight *self.weight
        """ 
        
        if self.loss_mode == 'mse':  
            loss=0
            # for cu,rh in zip(self.curv,self.rhs):
            
            for cu,rhs in zip(self.curv,self.rhs):
                for rhi in range(rhs.shape[1]):
                    # print(rhs[:,rhi].shape)
                    loss+= tf.reduce_mean(tf.reduce_mean(tf.square(model(cu)[:,rhi]-rhs[:,rhi]))  )  

        elif self.loss_mode == 'bce': 
            bce = tf.keras.losses.BinaryCrossentropy()  
            loss=0
            # for cu,rh  in zip(self.curv,self.rhs):
            #     loss+= tf.reduce_mean(bce(rh, model(cu)) )
            
            loss_tmp,reg_loss=1e10,0
            for weig in self.weight: 
                for cu,rhs in zip(self.curv,self.rhs): 
                    # for rhi in range(rhs.shape[1]):
                    rhs0=model(cu) 
                    # predicted_labels = np.argmax(rhs0, axis=1) 
                    for rhi in range(rhs.shape[1]):
                        # ghg = sum((predicted_labels == rhi))/rhs.shape[0]
                        # ghg=weig[rhi]
                        
                        # weig[rhi]*print(rhs[:,rhi].shape, model(cu).shape)
                        loss+= tf.reduce_mean(bce(rhs[:,rhi ],rhs0[:,rhi ] ) )
                if loss<loss_tmp:
                    loss_tmp=loss
                    reg_loss = tf.add_n(model.losses) if model.losses else 0.0
            
        else:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}") 
        return loss_tmp+reg_loss



        # elif self.loss_mode == 'bce': 
        #     bce = tf.keras.losses.BinaryCrossentropy()  
            
        #     # for cu,rh  in zip(self.curv,self.rhs):
        #     #     loss+= tf.reduce_mean(bce(rh, model(cu)) )
            
        #     # print('losssssss---------------------',self.weight )
                       
        #     loss_tmp=1e10
        #     for weig in self.weight:
        #         loss=0
        #         # print('lossssssscv-------',weig,self.adj)
        #         for cu,rhs,adj in zip(self.curv,self.rhs,self.adj): 
        #             # print('losssssssghv---------------------',rhs.shape[1])
        #             # for rhi in range(rhs.shape[1]): ,y=rhs[adj[1]] 
        #             rhs0=run_model_on_graph(model=model, X=cu[adj[1]] , adj=adj[0])
        #             # rhs0=run_model_on_graph(model=model, X=cu, adj=adj[0])
        #             # predicted_labels = np.argmax(rhs0, axis=1) 
        #             for rhi in range(rhs.shape[1]):
        #                 # ghg = sum((predicted_labels == rhi))/rhs.shape[0] 
                        
        #                 # print('losssssss---------------------',loss,rhs[:,rhi].shape, model(cu).shape)
        #                 loss+= weig[rhi]*tf.reduce_mean(bce(rhs[:,rhi ][adj[1]] ,rhs0[:,rhi ]) )
        #         if loss<loss_tmp:
        #             loss_tmp=loss
            
        # else:
        #     raise ValueError(f"Unsupported loss_mode: {self.loss_mode}") 
        # return loss_tmp

   
    def loss_fun(self, model,):  

        """
        Compute the loss for the model. *self.weight *self.weight
        """ 
        
        if self.loss_mode == 'mse':  
            loss=0
            # for cu,rh in zip(self.curv,self.rhs):
            
            for cu,rhs in zip(self.curv,self.rhs):
                for rhi in range(rhs.shape[1]):
                    # print(rhs[:,rhi].shape)
                    loss+= tf.reduce_mean(tf.reduce_mean(tf.square(model(cu)[:,rhi]-rhs[:,rhi]))  )  

        elif self.loss_mode == 'bce': 
            bce = tf.keras.losses.BinaryCrossentropy()  
            loss=0
            # for cu,rh  in zip(self.curv,self.rhs):
            #     loss+= tf.reduce_mean(bce(rh, model(cu)) )
             
            for cu,rhs in zip(self.curv,self.rhs): 
                # for rhi in range(rhs.shape[1]):
                rhs0=model(cu) 
                # predicted_labels = np.argmax(rhs0, axis=1) 
                for rhi in range(rhs.shape[1]):
                    # ghg = sum((predicted_labels == rhi))/rhs.shape[0]
                    # ghg=weig[rhi]
                    
                    # weig[rhi]*print(rhs[:,rhi].shape, model(cu).shape)
                    loss+= tf.reduce_mean(bce(rhs[:,rhi ],rhs0[:,rhi ] ) )
            return loss


        # elif self.loss_mode == 'bce': 
        #     bce = tf.keras.losses.BinaryCrossentropy()  
            
        #     # for cu,rh  in zip(self.curv,self.rhs):
        #     #     loss+= tf.reduce_mean(bce(rh, model(cu)) )
            
        #     # print('losssssss---------------------',self.weight )
                       
        #     loss_tmp=1e10
        #     for weig in self.weight:
        #         loss=0
        #         # print('lossssssscv-------',weig,self.adj)
        #         for cu,rhs,adj in zip(self.curv,self.rhs,self.adj): 
        #             # print('losssssssghv---------------------',rhs.shape[1])
        #             # for rhi in range(rhs.shape[1]): ,y=rhs[adj[1]] 
        #             rhs0=run_model_on_graph(model=model, X=cu[adj[1]] , adj=adj[0])
        #             # rhs0=run_model_on_graph(model=model, X=cu, adj=adj[0])
        #             # predicted_labels = np.argmax(rhs0, axis=1) 
        #             for rhi in range(rhs.shape[1]):
        #                 # ghg = sum((predicted_labels == rhi))/rhs.shape[0] 
                        
        #                 # print('losssssss---------------------',loss,rhs[:,rhi].shape, model(cu).shape)
        #                 loss+= weig[rhi]*tf.reduce_mean(bce(rhs[:,rhi ][adj[1]] ,rhs0[:,rhi ]) )
        #         if loss<loss_tmp:
        #             loss_tmp=loss
            
        # else:
        #     raise ValueError(f"Unsupported loss_mode: {self.loss_mode}") 
        # return loss_tmp


# def Get_iou(model,curv,dend_saves):
#     iou_s=[]
#     for ii in range(len(curv)):
#         rhs0=model(curv[ii])
#         dend=dend_saves[ii]
#         rhs0=rhs0.numpy()
#         vertices_approx_shaft_index=np.where(rhs0>.5)[0]
#         if len(vertices_approx_shaft_index)>0:
#             ss=set(vertices_approx_shaft_index)
#             sss=set(dend.vertices_true_shaft_index)
#             iou=len(ss.intersection(sss))/len(ss.union(sss))
#         else:
#             iou=0
#         iou_s.append(iou)
#     return iou_s

# def Get_iou(model, curv, lab,adj,get_model_one_hot=None,dend=None):
#     iou_s = {0: [], 1: [], 2: []}
    
#     for ii in range(len(lab)):
#         rhs0=run_model_on_graph(model=model, X=curv[ii][adj[ii][1]], adj=adj[ii][0]) 
#         index = lab[ii]
#         rhs0 = rhs0.numpy()
         
#         predicted_labels = np.argmax(rhs0, axis=1)  

#         for label in [0, 1, 2]: 
#             vertices_approx_index = np.where(predicted_labels == label)[0]
#             # print(vertices_approx_index,label,'hhhhh',index[label])
            
#             if len(vertices_approx_index) > 0:
#                 ss = set(vertices_approx_index)
#                 sss = set(index[label])
#                 iou = len(ss.intersection(sss)) / len(ss.union(sss))
#             else:
#                 iou = 0
            
#             iou_s[label].append(iou)
    
#     return iou_s


def Get_iou(model, curv, lab,adj=None,get_model_one_hot=None,dend=None):
    iou_s = {0: [], 1: [], 2: []}
    
    for ii in range(len(lab)):
        rhs0 = model(curv[ii]).numpy()
        # index = lab[ii] 
         
        # predicted_labels = np.argmax(rhs0, axis=1)  

        for label in range(len(lab[ii])): 
            vertices_approx_index = np.where(np.argmax(rhs0, axis=1)  == label)[0]
            # print(vertices_approx_index,label,'hhhhh',index[label])
            
            if len(vertices_approx_index) > 0:
                ss = set(vertices_approx_index)
                sss = set(lab[ii][label])
                iou = len(ss.intersection(sss)) / len(ss.union(sss))
            else:
                iou = 0
            
            iou_s[label].append(iou)
    
    return iou_s