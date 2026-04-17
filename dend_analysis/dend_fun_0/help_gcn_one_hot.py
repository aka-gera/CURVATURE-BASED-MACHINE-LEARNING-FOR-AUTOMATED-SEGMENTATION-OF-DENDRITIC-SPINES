import numpy as np 
import tensorflow as tf 
from spektral.layers import GCNConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
 

from keras.saving import register_keras_serializable
from tensorflow.keras import regularizers
DTYPE = tf.float32    
from spektral.data import Graph, Dataset, Loader
 
from tensorflow.keras.layers import Dense, Dropout

from sklearn.metrics import roc_curve, auc
 

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


    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.gcn1.channels,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

 
 



# ---------------------------------------------------------
# 1. Custom GCN Layer
# ---------------------------------------------------------
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, channels, activation=None):
        super().__init__()
        self.channels = channels
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        F = input_shape[0][-1]  # feature dimension
        self.W = self.add_weight(
            shape=(F, self.channels),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        X, A = inputs  # X: (N,F), A: (N,N)

        # Add self-loops
        I = tf.eye(tf.shape(A)[0], dtype=A.dtype)
        A_hat = A + I

        # Compute D^{-1/2}
        D = tf.reduce_sum(A_hat, axis=1)
        D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(D + 1e-10))

        # Normalized adjacency
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

        # GCN propagation
        XW = X @ self.W
        out = A_norm @ XW

        if self.activation:
            out = self.activation(out)

        return out


# ---------------------------------------------------------
# 2. Top-K Pooling Layer
# ---------------------------------------------------------
class TopKPool(tf.keras.layers.Layer):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio

    def build(self, input_shape):
        F = input_shape[0][-1]
        self.score_vector = self.add_weight(
            shape=(F, 1),
            initializer="glorot_uniform",
            trainable=False,
        )

    def call(self, inputs):
        X, A = inputs
        N = tf.shape(X)[0]

        # Compute scores
        scores = tf.squeeze(X @ self.score_vector, axis=1)

        # Top-k selection
        k = tf.cast(tf.math.ceil(self.ratio * tf.cast(N, tf.float32)), tf.int32)
        topk_scores, idx = tf.math.top_k(scores, k=k, sorted=False)

        # Pool features
        X_pooled = tf.gather(X, idx)

        # Pool adjacency
        A_pooled = tf.gather(tf.gather(A, idx, axis=0), idx, axis=1)

        return X_pooled, A_pooled, idx


# ---------------------------------------------------------
# 3. Unpooling
# ---------------------------------------------------------
def unpool(X_pooled, idx, N):
    F = tf.shape(X_pooled)[-1]
    out = tf.zeros((N, F), dtype=X_pooled.dtype)
    return tf.tensor_scatter_nd_update(out, tf.expand_dims(idx, 1), X_pooled)

 
@register_keras_serializable(package="Custom")
class gcn_UNet(tf.keras.Model):
    def __init__(self,
                 filters=32,
                 k1=0.5,
                 k2=0.5,
                 n_classes=2,
                 dropout_rate=0.1,
                 hidden_units=None,  
                 l1=0.01,
                 l2=0.0000001,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.filters = filters
        self.k1 = k1
        self.k2 = k2
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        self.gcn1 = GCNLayer(filters, activation="relu")
        self.pool1 = TopKPool(k1)

        self.gcn2 = GCNLayer(filters, activation="relu")
        self.pool2 = TopKPool(k2)

        self.gcn3 = GCNLayer(filters, activation="relu")
        self.gcn4 = GCNLayer(n_classes, activation="softmax")

        self.dropout = Dropout(dropout_rate)










    def call(self, inputs, training=False):
        X = inputs["x"]
        A = inputs["a"]

        X1 = self.gcn1([X, A])
        X1 = self.dropout(X1, training=training)
        X1_p, A1_p, idx1 = self.pool1([X1, A])

        X2 = self.gcn2([X1_p, A1_p])
        X2 = self.dropout(X2, training=training)
        X2_p, A2_p, idx2 = self.pool2([X2, A1_p])

        N2 = tf.shape(X2)[0]
        X3 = unpool(X2_p, idx2, N2)
        X3 = self.gcn3([X3, A1_p])

        N1 = tf.shape(X1)[0]
        X4 = unpool(X3, idx1, N1)
        out = self.gcn4([X4, A])

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "k1": self.k1,
            "k2": self.k2,
            "n_classes": self.n_classes,
            "dropout_rate": self.dropout_rate,
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
            "gcn_UNet".lower(): {
                "class": gcn_UNet,
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
    def __init__(self, rhs, curv,adj=None, weight=None, loss_mode='mse',dtype=None,dend=None,
                   rl_par= None, 
                   dnn_par= None, ):  
        self.rhs = rhs
        self.curv = curv
        self.adj=adj
        self.loss_mode = loss_mode
        # if weight is None:
        #     siz=self.rhs[0].shape[1]
        #     weight=1/siz*np.ones(siz)
        self.weight = weight 
    
    def loss_fun(self, model,):   
        if self.loss_mode == 'mse':  
            loss=0 
            
            for cu,rhs in zip(self.curv,self.rhs):
                for rhi in range(rhs.shape[1]): 
                    loss+= tf.reduce_mean(tf.reduce_mean(tf.square(model(cu)[:,rhi]-rhs[:,rhi]))  )  
 

        elif self.loss_mode == 'bce': 
            bce = tf.keras.losses.BinaryCrossentropy()  
            lossi=tf.zeros(len(self.weight[0]))
            for cu,rhs,weig in zip(self.curv,self.rhs,self.weight):  
                rhs0=model(cu)  
                bce_each =  tf.keras.losses.binary_crossentropy(rhs, rhs0,axis=0)  
                lossi+=tf.reduce_sum(weig*bce_each,axis=-1) 
            loss_tmp=min(lossi)

        else:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}") 
        return loss_tmp
 

def Get_iou(model, curv, lab,adj,get_model_one_hot=None,dend=None):
    iou_s = {0: [], 1: [], 2: []}
    
    for ii in range(len(lab)): 
        rhs0=model(curv[ii])
        index = lab[ii]
        rhs0 = rhs0.numpy()
         
        predicted_labels = np.argmax(rhs0, axis=1)  

        for label in range(len(index)): 
            vertices_approx_index = np.where(predicted_labels == label)[0]
            # print(len(vertices_approx_index),label,'hhhhh',label)
            
            if len(vertices_approx_index) > 0:
                ss = set(vertices_approx_index) 
                sss = set(index[label])
                iou = len(ss.intersection(sss)) / len(ss.union(sss))
            else:
                iou = 0
            
            iou_s[label].append(iou)
    
    return iou_s 



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



