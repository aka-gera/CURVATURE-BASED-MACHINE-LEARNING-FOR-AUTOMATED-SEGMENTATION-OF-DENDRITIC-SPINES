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
import tensorflow as tf
from tensorflow.keras import layers, Model 


@register_keras_serializable(package="Custom")
class pnet_PointNet(Model):
    def __init__(self, n_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes

        # Shared MLP
        self.mlp1 = layers.Conv1D(64, 1, activation="relu")
        self.mlp2 = layers.Conv1D(128, 1, activation="relu")
        self.mlp3 = layers.Conv1D(1024, 1, activation="relu")

        # Global feature
        self.global_pool = layers.GlobalMaxPooling1D()

        # Segmentation MLP
        self.seg1 = layers.Conv1D(256, 1, activation="relu")
        self.seg2 = layers.Conv1D(128, 1, activation="relu")
        self.seg_out = layers.Conv1D(n_classes, 1, activation="softmax")

    def call(self, x):
        # Per-point features
        f1 = self.mlp1(x)
        f2 = self.mlp2(f1)
        f3 = self.mlp3(f2)

        # Global feature
        global_feat = self.global_pool(f3)
        global_feat = tf.expand_dims(global_feat, 1)
        global_feat = tf.tile(global_feat, [1, tf.shape(x)[1], 1])

        # Concatenate local + global
        x = tf.concat([f1, f2, f3, global_feat], axis=-1)

        # Segmentation head
        x = self.seg1(x)
        x = self.seg2(x)
        return self.seg_out(x)

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)





 



import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------- Basic MLP block ----------
def mlp_block(channels, use_bn=True):
    net = tf.keras.Sequential()
    for c in channels:
        net.add(layers.Dense(c, use_bias=not use_bn))
        if use_bn:
            net.add(layers.BatchNormalization())
        net.add(layers.ReLU())
    return net

# ---------- Farthest Point Sampling ----------
def farthest_point_sample(xyz, npoint):
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]

    centroids = tf.TensorArray(tf.int32, size=npoint)
    distance = tf.ones((B, N)) * 1e10
    farthest = tf.random.uniform((B,), minval=0, maxval=N, dtype=tf.int32)

    def body(i, centroids, distance, farthest):
        centroids = centroids.write(i, farthest)
        centroid_xyz = tf.gather(xyz, farthest, batch_dims=1)
        dist = tf.reduce_sum((xyz - tf.expand_dims(centroid_xyz, 1)) ** 2, axis=-1)
        distance = tf.minimum(distance, dist)
        farthest = tf.argmax(distance, axis=-1, output_type=tf.int32)
        return i + 1, centroids, distance, farthest

    i = tf.constant(0)
    _, centroids, _, _ = tf.while_loop(
        lambda i, *_: i < npoint,
        body,
        [i, centroids, distance, farthest]
    )

    centroids = tf.transpose(centroids.stack(), [1, 0])  # (B, npoint)
    return centroids

# ---------- Ball query ----------
def query_ball_point(radius, nsample, xyz, new_xyz):
    B = tf.shape(xyz)[0]
    xyz_expand = tf.expand_dims(xyz, 1)          # (B, 1, N, 3)
    new_xyz_expand = tf.expand_dims(new_xyz, 2)  # (B, S, 1, 3)
    dist = tf.reduce_sum((xyz_expand - new_xyz_expand) ** 2, axis=-1)  # (B, S, N)
    idx = tf.argsort(dist, axis=-1)[:, :, :nsample]  # (B, S, nsample)
    return idx

# ---------- Grouping ----------
def group_points(points, idx):
    B = tf.shape(points)[0]
    S = tf.shape(idx)[1]
    nsample = tf.shape(idx)[2]
    batch_indices = tf.reshape(tf.range(B), (B, 1, 1))
    batch_indices = tf.tile(batch_indices, (1, S, nsample))
    gather_idx = tf.stack([batch_indices, idx], axis=-1)
    grouped = tf.gather_nd(points, gather_idx)
    return grouped

@register_keras_serializable(package="Custom")
class PointNetSetAbstraction(layers.Layer):
    def __init__(self, npoint, radius, nsample, mlp_channels, use_bn=True, **kwargs):
        super().__init__(**kwargs)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_channels = mlp_channels
        self.use_bn = use_bn
        self.mlp = mlp_block(mlp_channels, use_bn=use_bn)

    def call(self, xyz, features=None):
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = tf.gather(xyz, fps_idx, batch_dims=1)

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = group_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - tf.expand_dims(new_xyz, 2)

        if features is not None:
            grouped_features = group_points(features, idx)
            new_features = tf.concat([grouped_xyz_norm, grouped_features], axis=-1)
        else:
            new_features = grouped_xyz_norm

        B = tf.shape(new_features)[0]
        S = tf.shape(new_features)[1]
        K = tf.shape(new_features)[2]
        C = tf.shape(new_features)[3]

        new_features = tf.reshape(new_features, (B * S * K, C))
        new_features = self.mlp(new_features)
        C2 = tf.shape(new_features)[-1]
        new_features = tf.reshape(new_features, (B, S, K, C2))

        new_features = tf.reduce_max(new_features, axis=2)
        return new_xyz, new_features

    def get_config(self):
        config = super().get_config()
        config.update({
            "npoint": self.npoint,
            "radius": self.radius,
            "nsample": self.nsample,
            "mlp_channels": self.mlp_channels,
            "use_bn": self.use_bn,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable(package="Custom")
class PointNetFeaturePropagation(layers.Layer):
    def __init__(self, mlp_channels, use_bn=True, **kwargs):
        super().__init__(**kwargs)
        self.mlp_channels = mlp_channels
        self.use_bn = use_bn
        self.mlp = mlp_block(mlp_channels, use_bn=use_bn)

    def call(self, xyz1, xyz2, features1, features2):
        B = tf.shape(xyz1)[0]
        N1 = tf.shape(xyz1)[1]
        N2 = tf.shape(xyz2)[1]

        xyz1_expand = tf.expand_dims(xyz1, 2)
        xyz2_expand = tf.expand_dims(xyz2, 1)
        dist = tf.reduce_sum((xyz1_expand - xyz2_expand) ** 2, axis=-1)

        idx = tf.argsort(dist, axis=-1)[:, :, :3]
        dist_sorted = tf.gather(dist, idx, batch_dims=2)
        dist_sorted = tf.maximum(dist_sorted, 1e-10)
        weight = 1.0 / dist_sorted
        weight = weight / tf.reduce_sum(weight, axis=-1, keepdims=True)

        batch_indices = tf.reshape(tf.range(B), (B, 1, 1))
        batch_indices = tf.tile(batch_indices, (1, N1, 3))
        gather_idx = tf.stack([batch_indices, idx], axis=-1)
        interpolated_features = tf.gather_nd(features2, gather_idx)

        interpolated_features = tf.reduce_sum(
            interpolated_features * tf.expand_dims(weight, -1), axis=2
        )

        if features1 is not None:
            new_features = tf.concat([interpolated_features, features1], axis=-1)
        else:
            new_features = interpolated_features

        B = tf.shape(new_features)[0]
        N = tf.shape(new_features)[1]
        C = tf.shape(new_features)[2]
        new_features = tf.reshape(new_features, (B * N, C))
        new_features = self.mlp(new_features)
        C2 = tf.shape(new_features)[-1]
        new_features = tf.reshape(new_features, (B, N, C2))

        return new_features

    def get_config(self):
        config = super().get_config()
        config.update({
            "mlp_channels": self.mlp_channels,
            "use_bn": self.use_bn,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="Custom")
class pnet_2PointNet2(Model):
    def __init__(self, n_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            mlp_channels=[32, 32, 64]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            mlp_channels=[64, 64, 128]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            mlp_channels=[128, 128, 256]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            mlp_channels=[256, 256, 512]
        )

        # Feature Propagation layers
        self.fp4 = PointNetFeaturePropagation([256, 256])
        self.fp3 = PointNetFeaturePropagation([256, 256])
        self.fp2 = PointNetFeaturePropagation([256, 128])
        self.fp1 = PointNetFeaturePropagation([128, 128, 128])

        # Final segmentation head (per-point)
        self.seg_head = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(n_classes)   
        ])

    def call(self, x, training=False):
        # x: (B, N, 3) — pure coordinates, no extra features
        l0_xyz = x
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        logits = self.seg_head(l0_points, training=training)  # (B, N, n_classes)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_classes": self.n_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)













#######################################################


















class model_choice: 
    def __init__(self):
        self.model_factory = {
            "pnet_PointNet".lower(): pnet_PointNet, 
            'pnet_2PointNet2'.lower():pnet_2PointNet2,
        }

    def get_model(self, model_type, n_classes=2, **kwargs):
        model_type = model_type.lower()
 
        for key, model_cls in self.model_factory.items():
            if model_type.startswith(key):
                return model_cls(n_classes=n_classes, **kwargs) 
        raise ValueError(f"Unknown model type: {model_type}")

    def get_custom_objects(self, model_type):
        model_type = model_type.lower()

        for key, model_cls in self.model_factory.items():
            if model_type.startswith(key):
                # Return the CLASS, not an instance
                return {model_cls.__name__: model_cls} 
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
            cce = tf.keras.losses.CategoricalCrossentropy()

            for weig in self.weight:
                for cu, rhs in zip(self.curv, self.rhs):

                    rhs0 = model(cu)          # (1, N, C) 

                    loss += cce(rhs, rhs0) 
                if loss<loss_tmp:
                    loss_tmp=loss
            
        else:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}") 
        return loss_tmp

 
def Get_iou(model, curv, lab,adj=None,get_model_one_hot=None,dend=None):
    iou_s = {0: [], 1: [], 2: []} 
    for ii in range(len(lab)):
        rhs0 = model(curv[ii]).numpy()[0,...] 
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

 