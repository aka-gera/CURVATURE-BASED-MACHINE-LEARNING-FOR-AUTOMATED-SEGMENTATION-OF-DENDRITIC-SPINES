import numpy as np
import time  
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout
from keras.saving import register_keras_serializable
from tensorflow.keras import regularizers
from sklearn.metrics import roc_curve, auc
 
DTYPE = tf.float32
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
import tensorflow as tf
from tensorflow.keras import layers, Model


@register_keras_serializable(package="Custom")
class dnn_PointNet(Model):
    def __init__(self, n_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes

        # Shared MLP (Conv1D with kernel size 1)
        self.mlp1 = layers.Conv1D(64, 1, activation="relu")
        self.mlp2 = layers.Conv1D(128, 1, activation="relu")
        self.mlp3 = layers.Conv1D(1024, 1, activation="relu")

        # Global feature aggregation
        self.global_pool = layers.GlobalMaxPooling1D()

        # Fully connected layers
        self.fc1 = layers.Dense(512, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.dropout = layers.Dropout(0.3)

        # Output classifier
        self.classifier = layers.Dense(n_classes, activation="softmax")

    def call(self, x): 
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        x = self.global_pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return self.classifier(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_classes": self.n_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
 

 

@register_keras_serializable(package="Custom")
class dnn_GINN(Model):
    def __init__(self, Dtype=DTYPE, 
                 hidden_layers=4, 
                 neurons_per_layer=100, 
                 n_classes=2,
                 activation_init="relu",
                 activation_hidden="relu",
                #  activation_last="softmax",
                 activation_last="sigmoid",

                 l1=0.01,
                 l2=0.0000001,
                 **kwargs):
        super(dnn_GINN, self).__init__()
        self.hidden_layers = hidden_layers
        self.Dtype = Dtype
        self.n_classes=n_classes
        k_regularizers=regularizers.l1_l2(l1=l1,l2=l2)
        
        self.ld_init = Dense(neurons_per_layer, activation=activation_init, dtype=Dtype)
        self.hidden_layers_list = [
            Dense(neurons_per_layer, 
                  activation=activation_hidden, 
                  kernel_regularizer=k_regularizers,
                  dtype=Dtype) 
            for _ in range(hidden_layers)
        ]
        self.ld_last = Dense(n_classes,  activation=activation_last, dtype=Dtype)

    def call(self, x):
        x = self.ld_init(x)
        for layer in self.hidden_layers_list:
            x = layer(x)
        x = self.ld_last(x)
        return x


    def get_config(self):
        config = super().get_config()
        config.update({
            "Dtype": self.Dtype,
            "n_classes": self.n_classes,
            "hidden_layers": self.hidden_layers,
            "activation_init": self.ld_init.activation.__name__,
            "activation_hidden": self.hidden_layers_list[0].activation.__name__,
            "activation_last": self.ld_last.activation.__name__,
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)





 

@register_keras_serializable(package="Custom")
class dnn_1GINN_SM(Model):
    def __init__(self, Dtype=DTYPE, 
                 hidden_layers=4, 
                 neurons_per_layer=100, 
                 n_classes=2,
                 activation_init="relu",
                 activation_hidden="relu",
                activation_last="softmax",
                #   activation_last="sigmoid",

                 l1=0.01,
                 l2=0.0000001,
                 **kwargs):
        super(dnn_1GINN_SM, self).__init__()
        self.hidden_layers = hidden_layers
        self.Dtype = Dtype
        self.n_classes=n_classes
        k_regularizers=regularizers.l1_l2(l1=l1,l2=l2)
        
        self.ld_init = Dense(neurons_per_layer, activation=activation_init, dtype=Dtype)
        self.hidden_layers_list = [
            Dense(neurons_per_layer, 
                  activation=activation_hidden, 
                  kernel_regularizer=k_regularizers,
                  dtype=Dtype) 
            for _ in range(hidden_layers)
        ]
        self.ld_last = Dense(n_classes,  activation=activation_last, dtype=Dtype)

    def call(self, x):
        x = self.ld_init(x)
        for layer in self.hidden_layers_list:
            x = layer(x)
        x = self.ld_last(x)
        return x


    def get_config(self):
        config = super().get_config()
        config.update({
            "Dtype": self.Dtype,
            "n_classes": self.n_classes,
            "hidden_layers": self.hidden_layers,
            "activation_init": self.ld_init.activation.__name__,
            "activation_hidden": self.hidden_layers_list[0].activation.__name__,
            "activation_last": self.ld_last.activation.__name__,
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)














class model_choice: 
    def __init__(self, model_type=None, n_classes=2):
        self.model_type = model_type
        self.n_classes = n_classes
 
        self.models = {
            "dnn_PointNet".lower(): {
                "class": dnn_PointNet,
                "params": {"multiple": 16, "margin": 2},
                "filters": 32
            },
            'dnn_GINN'.lower(): {
                "class": dnn_GINN,
                "params": {"multiple": 16, "margin": 2},
                "filters": 32
            }, 
            'dnn_1GINN_SM'.lower(): {
                "class": dnn_1GINN_SM,
                "params": {"multiple": 16, "margin": 2},
                "filters": 32
            }, 
        }

    def get_model(self, model_type=None, n_classes=None, **kwargs):
        model_type = model_type if model_type is not None else self.model_type
        n_classes = n_classes if n_classes is not None else self.n_classes
        model_type = model_type.lower()

        for key, entry in self.models.items():
            if model_type.startswith(key):
 
                filters = kwargs.pop("filters", entry["filters"])

                return entry["class"](filters=filters, n_classes=n_classes, **kwargs)

        raise ValueError(f"Unknown model type: {model_type}")

    def get_cropper_params(self, model_type=None):
        model_type = model_type if model_type is not None else self.model_type
        model_type = model_type.lower()

        for key, entry in self.models.items():
            if model_type.startswith(key):
                return entry["params"]

        raise ValueError(f"No cropper params for model type: {model_type}")

    def get_custom_objects(self, model_type):
        model_type = model_type.lower()

        for key, entry in self.models.items():
            if model_type.startswith(key):
                cls = entry["class"]
                return {cls.__name__: cls}

        raise ValueError(f"Unknown model type: {model_type}")





 


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
    def __init__(self, rhs, curv, 
        weight=None,
        dend=None,
        pid=None,
        get_model_one_hot=None,
        adj=None, 
        loss_mode='mse',
        dtype=None,
        rl_par= 0.5, 
        dnn_par= 0.5, 
        weights=None,  
        shaft_thre=1/4,
        smooth_tf=False,
        neck_lim=0,
        get_data_txt=False,
        reconstruction_tf=False, 
                        ):  
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
                rhs0=model(cu) 
                for rhi in range(rhs.shape[1]): 
                    loss+= tf.reduce_mean(tf.reduce_mean(tf.square(rhs0[:,rhi]-rhs[:,rhi]))  )  
        
        elif self.loss_mode == 'bce': 
            bce = tf.keras.losses.BinaryCrossentropy()  
             
            lossi=tf.zeros(len(self.weight[0]))
            for cu,rhs,weig in zip(self.curv,self.rhs,self.weight):  
                rhs0=model(cu)  
                bce_each =  tf.keras.losses.binary_crossentropy(rhs, rhs0,axis=0)  
                lossi+=tf.reduce_sum(weig*bce_each,axis=-1) 
            loss_tmp=min(lossi) 

 


        elif self.loss_mode == 'bce old': 
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



def get_auc(model, curv, rhs,adj=None,get_model_one_hot=None,dend=None):
    auc_s = {0: [], 1: [], 2: []}
    y_true,score=[],[]
    
    for cuu,rh in zip(curv,rhs):
        score.append(model(cuu).numpy()) 
        y_true.append(rh) 
    if score and y_true:
        yy_true=np.vstack(y_true)
        y_score=np.vstack(score)
        for label,(yy,sc,nm) in enumerate(zip(yy_true.T,y_score.T,['shaft','spine'])):
            # print('=============',yy.shape)
            fpr, tpr, _ = roc_curve(yy, y_score=sc) 
            auc_s[label]=auc(fpr, tpr)
 
    
    return auc_s

 
class model_metric:
    def __init__(self, model, curv, rhs_index=None, ):

        # metric names
        self.mmjj = ['iou', 'auc', 'dice']
        self.mmjjind = [f'{mmm}_ind' for mmm in self.mmjj]
 
        score = [model(cuu).numpy() for cuu in curv]   
        y_score = np.vstack(score)                     
        score_ind = [np.argmax(yy, axis=1) for yy in score] 
        y_pred= sum([list(mm) for mm in score_ind],[])
        y_true= sum([list(mm) for mm in rhs_index],[])
 
        self.n_classes = sorted(list(set(y_true) | set(y_pred)))
 
        self.metrics = {mm: {c: [] for c in self.n_classes}
                        for mm in self.mmjj + self.mmjjind}
 
        global_metrics = self.get_metric(y_score=y_score,
                                         y_true=y_true,
                                         y_pred=y_pred)

        for met in self.mmjj:
            self.metrics[met] = global_metrics[met]
 
        for sco, ytr, ypre in zip(score, rhs_index, score_ind):
            local_metrics = self.get_metric(y_score=sco,
                                            y_true=ytr,
                                            y_pred=ypre)
            for met in self.mmjjind:
                base = met.replace("_ind", "")
                for c in self.n_classes:
                    self.metrics[met][c].append(local_metrics[base][c])


    def get_metric(self, y_score, y_true, y_pred):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        metrics = {mm: {c: None for c in self.n_classes} for mm in self.mmjj}

        for k in self.n_classes: 
            yy = (y_true == k).astype(int)
            sc = y_score[:, k]
 
            if np.sum(yy) > 0:
                fpr, tpr, _ = roc_curve(yy, sc)
                aucc = auc(fpr, tpr)
            else:
                aucc = 0
 
            pred_k = set(np.where(y_pred == k)[0])
            true_k = set(np.where(y_true == k)[0])

            if len(pred_k) > 0 or len(true_k) > 0:
                inter = len(pred_k & true_k)
                union = len(pred_k | true_k)
                iou = inter / union
                dice = 2 * inter / (len(pred_k) + len(true_k))
            else:
                iou, dice = 0, 0

            metrics['iou'][k] = iou
            metrics['auc'][k] = aucc
            metrics['dice'][k] = dice

        return metrics

