import numpy as np
import time  
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout
from keras.saving import register_keras_serializable
from tensorflow.keras import regularizers
DTYPE = tf.float32


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
                 get_model_one_hot=None,
                 adj=None, 
                 loss_mode='mse',
                 dtype=None,rl_par= 0.5, dnn_par= 0.5 ):  
        self.rhs = rhs
        self.curv = curv
        self.loss_mode = loss_mode
        self.rl_par, self.dnn_par =rl_par, dnn_par  
        self.weight = weight 
        self.dtype=dtype
    
    def loss_fun(self, model,):   
        if self.loss_mode == 'mse':  
            loss=0 
            for cu,rhs in zip(self.curv,self.rhs):
                for rhi in range(rhs.shape[1]): 
                    loss+= tf.reduce_mean(tf.reduce_mean(tf.square(model(cu)[:,rhi]-rhs[:,rhi]))  )  
        
        elif self.loss_mode == 'bce': 
            bce = tf.keras.losses.BinaryCrossentropy()   

            loss = 0.0
            nh = 0

            for cu, rhs in zip(self.curv, self.rhs): 
                logits = model(cu) 
                probs = tf.squeeze(logits, axis=-1) if logits.shape[-1] == 1 else logits
 
                actions = tf.cast(tf.random.uniform(tf.shape(probs)) < probs, dtype=self.dtype)
 
                rewards = tf.cast(tf.equal(actions, rhs), dtype=self.dtype)
 
                baseline = tf.reduce_mean(rewards) 

                log_probs = tf.math.log(
                    tf.where(tf.equal(actions, 1), probs, 1.0 - probs) + 1e-8
                )
 
                rl_loss = -tf.reduce_mean((rewards - baseline) * log_probs)
 
                dnn_loss = bce(rhs, probs)
  
                loss +=  self.rl_par * rl_loss + self.dnn_par * dnn_loss
                nh += 1
 
            # loss = loss / nh
            
        else:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}") 
        return loss 

 

def Get_iou(model, curv, lab,adj=None,get_model_one_hot=None,dend=None):
    iou_s = {0: [], 1: [], 2: []}
    
    for ii in range(len(lab)):
        rhs0 = model(curv[ii])
        index = lab[ii]
        rhs0 = rhs0.numpy()
         
        predicted_labels = np.argmax(rhs0, axis=1)  

        for label in [0, 1, 2]: 
            vertices_approx_index = np.where(predicted_labels == label)[0] 
            
            if len(vertices_approx_index) > 0:
                ss = set(vertices_approx_index)
                sss = set(index[label])
                iou = len(ss.intersection(sss)) / len(ss.union(sss))
            else:
                iou = 0
            
            iou_s[label].append(iou)
    
    return iou_s


