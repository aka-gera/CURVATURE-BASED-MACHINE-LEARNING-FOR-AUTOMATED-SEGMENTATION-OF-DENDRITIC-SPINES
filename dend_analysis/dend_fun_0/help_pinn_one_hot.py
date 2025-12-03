import numpy as np
import time  
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout
from keras.saving import register_keras_serializable
from tensorflow.keras import regularizers
DTYPE = tf.float32
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

@register_keras_serializable(package="Custom")
class PINN(Model):
    def __init__(self, Dtype=DTYPE, 
                 hidden_layers=4, 
                 neurons_per_layer=100, 
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
        
        self.ld_init = Dense(neurons_per_layer, activation=activation_init, dtype=Dtype)
        self.hidden_layers_list = [
            Dense(neurons_per_layer, 
                  activation=activation_hidden, 
                  kernel_regularizer=k_regularizers,
                  dtype=Dtype) 
            for _ in range(hidden_layers)
        ]
        self.ld_last = Dense(n_col,  activation=activation_last, dtype=Dtype)

    def call(self, x):
        x = self.ld_init(x)
        for layer in self.hidden_layers_list:
            x = layer(x)
        x = self.ld_last(x)
        return x



class PINN_ADV(Model):
    def __init__(self, Dtype=DTYPE, 
                 hidden_layers=4, 
                 neurons_per_layer=100, 
                 n_col=2,
                 activation_init="relu",
                 activation_hidden="relu",
                 activation_last="sigmoid",
                 dropout_prob=.3,
                 l1=0.01,
                 l2=0.001,
                 **kwargs):
        super(PINN_ADV, self).__init__()
        self.hidden_layers = hidden_layers
        self.Dtype = Dtype
        k_regularizers=regularizers.l1_l2(l1=l1,l2=l2)
        
        self.ld_init = Dense(neurons_per_layer, activation=activation_init, dtype=Dtype)
        self.hidden_layers_list = [
            Dense(neurons_per_layer, 
                  activation=activation_hidden, 
                  kernel_regularizer=k_regularizers,
                  dtype=Dtype) 
            for _ in range(hidden_layers)
        ]
        self.dropout = Dropout(dropout_prob)
        self.ld_last = Dense(n_col,  activation=activation_last, dtype=Dtype)

    def call(self, x,training=False):
        x = self.ld_init(x)
        for ii,layer in enumerate(self.hidden_layers_list):
            x = layer(x) 
            if np.mod(ii,10)==0:
                x = self.dropout(x,training=training)


        x = self.ld_last(x)
        return x



 
 


class aka_train():
    def __init__(self) :
      pass

    def get_grad_back_prop(self,fun,model): 

        with tf.GradientTape(persistent=True) as tape:  
            loss = fun.loss_fun(model) 
        grad = tape.gradient(loss,  model.trainable_variables) 
        del tape

        return loss, grad
    
    @tf.function
    def train_PINN(self,optimizer,fun,model):
        loss, grads = self.get_grad_back_prop(fun,model)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

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



class LOSS():
    def __init__(self, rhs, curv, weight=None,
                 dend=None,
                 pid=None,
                 get_model_one_hot=None,
                 adj=None, 
                 loss_mode='mse',
                 dtype=None,rl_par= 0.5, dnn_par= 0.5, 
                        weights=None,  
                        shaft_thre=1/4,
                        smooth_tf=False,
                        neck_lim=0,
                        get_data_txt=False,
                        reconstruction_tf=False, ):  
        self.rhs = rhs
        self.curv = curv
        self.loss_mode = loss_mode
        self.rl_par, self.dnn_par =rl_par, dnn_par  
        self.weight = weight 
        self.dtype=dtype
        self.pid=pid 
        self.weights=weights 
        self.shaft_thre=shaft_thre
        self.smooth_tf=smooth_tf
        self.neck_lim=neck_lim
        self.get_data_txt=get_data_txt
        self.reconstruction_tf=reconstruction_tf
    
    def loss_fun(self, model,):   
        
        if self.loss_mode == 'mse':  
            loss=0 
            
            for cu,rhs in zip(self.curv,self.rhs):
                for rhi in range(rhs.shape[1]): 
                    loss+= tf.reduce_mean(tf.reduce_mean(tf.square(model(cu)[:,rhi]-rhs[:,rhi]))  )  
        
        elif self.loss_mode == 'bce': 
            bce = tf.keras.losses.BinaryCrossentropy()  
            loss=0 
            
            loss_tmp=1e10
            for weig in self.weight:
                
                for cu,rhs in zip(self.curv,self.rhs): 
                    # for rhi in range(rhs.shape[1]):
                    rhs0=model(cu) 
                    # predicted_labels = np.argmax(rhs0, axis=1) 
                    for rhi in range(rhs.shape[1]): 
                        
                        # print(rhs[:,rhi].shape, model(cu).shape)
                        loss+= weig[rhi]*tf.reduce_mean(bce(rhs[:,rhi ],rhs0[:,rhi ] ) )
                if loss<loss_tmp:
                    loss_tmp=loss
            
        else:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}") 
        return loss_tmp

 
def Get_iou(model, curv, lab,adj=None,get_model_one_hot=None,dend=None):
    iou_s = {0: [], 1: [], 2: []}
    
    for ii in range(len(lab)):
        rhs0 = model(curv[ii]).numpy() 
        for label in range(len(lab[ii])): 
            vertices_approx_index = np.where(np.argmax(rhs0, axis=1)  == label)[0] 
            
            if len(vertices_approx_index) > 0:
                ss = set(vertices_approx_index)
                sss = set(lab[ii][label])
                iou = len(ss.intersection(sss)) / len(ss.union(sss))
            else:
                iou = 0
            
            iou_s[label].append(iou)
    
    return iou_s

 