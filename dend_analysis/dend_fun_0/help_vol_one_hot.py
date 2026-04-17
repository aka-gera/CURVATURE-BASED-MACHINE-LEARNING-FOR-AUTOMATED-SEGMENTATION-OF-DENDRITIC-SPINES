 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

 
 

@register_keras_serializable(package="Custom")
class vol_VGG16_FCN3D(Model):
    def __init__(self, filters=64, n_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        # Encoder (same as VGG16 blocks)
        self.block1 = tf.keras.Sequential([
            layers.Conv3D(filters, 3, padding="same", activation="relu"),
            layers.Conv3D(filters, 3, padding="same", activation="relu")
        ])
        self.pool1 = layers.MaxPool3D()

        self.block2 = tf.keras.Sequential([
            layers.Conv3D(filters*2, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*2, 3, padding="same", activation="relu")
        ])
        self.pool2 = layers.MaxPool3D()

        self.block3 = tf.keras.Sequential([
            layers.Conv3D(filters*4, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*4, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*4, 3, padding="same", activation="relu")
        ])
        self.pool3 = layers.MaxPool3D()

        self.block4 = tf.keras.Sequential([
            layers.Conv3D(filters*8, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*8, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*8, 3, padding="same", activation="relu")
        ])
        self.pool4 = layers.MaxPool3D()

        self.block5 = tf.keras.Sequential([
            layers.Conv3D(filters*8, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*8, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*8, 3, padding="same", activation="relu")
        ])
        self.pool5 = layers.MaxPool3D()

        # Decoder (upsampling path)
        self.up5 = layers.Conv3DTranspose(filters*8, 2, strides=2, padding="same")
        self.dec5 = tf.keras.Sequential([
            layers.Conv3D(filters*8, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*8, 3, padding="same", activation="relu")
        ])

        self.up4 = layers.Conv3DTranspose(filters*8, 2, strides=2, padding="same")
        self.dec4 = tf.keras.Sequential([
            layers.Conv3D(filters*8, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*8, 3, padding="same", activation="relu")
        ])

        self.up3 = layers.Conv3DTranspose(filters*4, 2, strides=2, padding="same")
        self.dec3 = tf.keras.Sequential([
            layers.Conv3D(filters*4, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*4, 3, padding="same", activation="relu")
        ])

        self.up2 = layers.Conv3DTranspose(filters*2, 2, strides=2, padding="same")
        self.dec2 = tf.keras.Sequential([
            layers.Conv3D(filters*2, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*2, 3, padding="same", activation="relu")
        ])

        self.up1 = layers.Conv3DTranspose(filters, 2, strides=2, padding="same")
        self.dec1 = tf.keras.Sequential([
            layers.Conv3D(filters, 3, padding="same", activation="relu"),
            layers.Conv3D(filters, 3, padding="same", activation="relu")
        ])

        self.out_layer = layers.Conv3D(n_classes, 1, activation="softmax")

    def call(self, x):
        # x_padded, pads = pad_to_multiple(x, multiple=32)

        c1 = self.block1(x)
        p1 = self.pool1(c1)

        c2 = self.block2(p1)
        p2 = self.pool2(c2)

        c3 = self.block3(p2)
        p3 = self.pool3(c3)

        c4 = self.block4(p3)
        p4 = self.pool4(c4)

        c5 = self.block5(p4)
        p5 = self.pool5(c5)

        u5 = self.up5(p5)
        u5 = self.dec5(u5)

        u4 = self.up4(u5)
        u4 = self.dec4(u4)

        u3 = self.up3(u4)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = self.dec1(u1)

        out = self.out_layer(u1)
        # out = crop_back(out, pads)
        return out




    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def crop_to_match(source, target):
    """
    Crops `source` spatially so it matches `target` shape.
    """
    src_shape = tf.shape(source)
    tgt_shape = tf.shape(target)

    # Compute how much to crop
    crop_d = src_shape[1] - tgt_shape[1]
    crop_h = src_shape[2] - tgt_shape[2]
    crop_w = src_shape[3] - tgt_shape[3]

    crop_d = tf.maximum(crop_d, 0)
    crop_h = tf.maximum(crop_h, 0)
    crop_w = tf.maximum(crop_w, 0)

    return source[:, 
                  :src_shape[1] - crop_d,
                  :src_shape[2] - crop_h,
                  :src_shape[3] - crop_w,
                  :]

def crop_to_match(a, b):
    """
    Returns (a_cropped, b_cropped) so that
    a_cropped.shape[1:4] == b_cropped.shape[1:4]
    by cropping both to the minimum spatial size.
    """
    sa = tf.shape(a)
    sb = tf.shape(b)

    d = tf.minimum(sa[1], sb[1])
    h = tf.minimum(sa[2], sb[2])
    w = tf.minimum(sa[3], sb[3])

    a_c = a[:, :d, :h, :w, :]
    b_c = b[:, :d, :h, :w, :]
    return a_c, b_c

def crop_to_match(a, b):
    """
    Returns (a_cropped, b_cropped) so that
    a_cropped.shape[1:4] == b_cropped.shape[1:4]
    by cropping both to the minimum spatial size.
    """
    sa = tf.shape(a)
    sb = tf.shape(b)

    d = tf.minimum(sa[1], sb[1])
    h = tf.minimum(sa[2], sb[2])
    w = tf.minimum(sa[3], sb[3])

    a_c = a[:, :d, :h, :w, :]
    b_c = b[:, :d, :h, :w, :]
    return a_c, b_c


def pad_to_multiple(x, multiple=16):
    shape = tf.shape(x)
    d = shape[1]
    h = shape[2]
    w = shape[3]

    pad_d = (multiple - d % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    paddings = [[0,0],
                [0, pad_d],
                [0, pad_h],
                [0, pad_w],
                [0,0]]
    return tf.pad(x, paddings), (pad_d, pad_h, pad_w)


def crop_back(pred, pads):
    pad_d, pad_h, pad_w = pads
    if pad_d > 0:
        pred = pred[:, :-pad_d, :, :, :]
    if pad_h > 0:
        pred = pred[:, :, :-pad_h, :, :]
    if pad_w > 0:
        pred = pred[:, :, :, :-pad_w, :]
    return pred



@register_keras_serializable(package="Custom")
class vol_UNet3D(tf.keras.Model):
    def __init__(self, filters=32, n_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.filters = filters
        # define layers...


# class UNet3D(Model):
#     def __init__(self, filters=32, n_classes=1):
#         super().__init__()

        self.conv1 = tf.keras.Sequential([
            layers.Conv3D(filters, 3, padding="same", activation="relu"),
            layers.Conv3D(filters, 3, padding="same", activation="relu")
        ])
        self.pool1 = layers.MaxPool3D()

        self.conv2 = tf.keras.Sequential([
            layers.Conv3D(filters*2, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*2, 3, padding="same", activation="relu")
        ])
        self.pool2 = layers.MaxPool3D()

        self.conv3 = tf.keras.Sequential([
            layers.Conv3D(filters*4, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*4, 3, padding="same", activation="relu")
        ])
        self.pool3 = layers.MaxPool3D()

        self.conv4 = tf.keras.Sequential([
            layers.Conv3D(filters*8, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*8, 3, padding="same", activation="relu")
        ])
        self.pool4 = layers.MaxPool3D()

        self.middle = tf.keras.Sequential([
            layers.Conv3D(filters*16, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*16, 3, padding="same", activation="relu")
        ])

        self.up4 = layers.Conv3DTranspose(filters*8, 2, strides=2, padding="same")
        self.dec4 = tf.keras.Sequential([
            layers.Conv3D(filters*8, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*8, 3, padding="same", activation="relu")
        ])

        self.up3 = layers.Conv3DTranspose(filters*4, 2, strides=2, padding="same")
        self.dec3 = tf.keras.Sequential([
            layers.Conv3D(filters*4, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*4, 3, padding="same", activation="relu")
        ])

        self.up2 = layers.Conv3DTranspose(filters*2, 2, strides=2, padding="same")
        self.dec2 = tf.keras.Sequential([
            layers.Conv3D(filters*2, 3, padding="same", activation="relu"),
            layers.Conv3D(filters*2, 3, padding="same", activation="relu")
        ])

        self.up1 = layers.Conv3DTranspose(filters, 2, strides=2, padding="same")
        self.dec1 = tf.keras.Sequential([
            layers.Conv3D(filters, 3, padding="same", activation="relu"),
            layers.Conv3D(filters, 3, padding="same", activation="relu")
        ])

        self.out_layer = layers.Conv3D(n_classes, 1, activation="softmax")

    def call(self, x):
        # 1) pad input first
        x_padded, pads = pad_to_multiple(x, multiple=16)

        c1 = self.conv1(x_padded)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        m = self.middle(p4)

        u4 = self.up4(m)
        u4, c4 = crop_to_match(u4, c4)
        u4 = tf.concat([u4, c4], axis=-1)
        u4 = self.dec4(u4)

        u3 = self.up3(u4)
        u3, c3 = crop_to_match(u3, c3)
        u3 = tf.concat([u3, c3], axis=-1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2, c2 = crop_to_match(u2, c2)
        u2 = tf.concat([u2, c2], axis=-1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1, c1 = crop_to_match(u1, c1)
        u1 = tf.concat([u1, c1], axis=-1)
        u1 = self.dec1(u1)

        mm = self.out_layer(u1)

        # 2) crop back to original size
        pred = crop_back(mm, pads)
        return pred


    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

 
import tensorflow as tf 
from tensorflow.keras.layers import Conv3D, MaxPool3D, Dropout, Conv3DTranspose, Concatenate
from tensorflow.keras.initializers import TruncatedNormal, VarianceScaling
from tensorflow.keras.regularizers import L2

@register_keras_serializable(package="Custom")
class SegModel(Model):
    def __init__(self, filters=32, n_classes=3, activation="relu",
                 kernel_init="glorot_uniform", kernel_reg=None, dropout_prob=0.0, **kwargs):
        super().__init__( **kwargs)

        self.filters = filters
        self.n_classes = n_classes
        self.activation = activation
        self.dropout_prob = dropout_prob

        # Initializers
        if kernel_init == "normal":
            self.base_init = TruncatedNormal(stddev=0.1)
        elif kernel_init == "He":
            self.base_init = VarianceScaling()
        else:
            self.base_init = kernel_init

        # Regularizer
        self.reg_init = L2(kernel_reg) if kernel_reg else None

        self.dropout = Dropout(rate=self.dropout_prob)
        self.concat = Concatenate(axis=-1)

    # ---- rewritten conv3D ----
    def conv3D(self, filters, kernel=3, stride=1, name=None):
        return Conv3D(
            filters,
            kernel_size=kernel,
            strides=stride,
            padding="same",                     # always SAME
            activation=self.activation,
            kernel_initializer=self.base_init,
            kernel_regularizer=self.reg_init,
            name=name
        )

    # ---- rewritten conv3DT ----
    def conv3DT(self, filters, kernel=2, stride=2, name=None):
        return Conv3DTranspose(
            filters,
            kernel_size=kernel,
            strides=stride,
            padding="same",                     # always SAME
            kernel_initializer=self.base_init,
            kernel_regularizer=self.reg_init,
            name=name
        )
 
    def concat_tensors(self, enc, dec):
        enc = enc[:, :dec.shape[1], :dec.shape[2], :dec.shape[3], :]
        return tf.concat([enc, dec], axis=-1)
    
    def pad_to_multiple(self,x, multiple=16):
        shape = tf.shape(x)
        d, h, w = shape[1], shape[2], shape[3]

        def pad_amount(size):
            remainder = size % multiple
            return (multiple - remainder) if remainder != 0 else 0

        pd = pad_amount(d)
        ph = pad_amount(h)
        pw = pad_amount(w)

        paddings = [[0, 0],
                    [0, pd],
                    [0, ph],
                    [0, pw],
                    [0, 0]]

        x_padded = tf.pad(x, paddings, mode="CONSTANT")
        return x_padded, (pd, ph, pw)


    def crop_back(self,x, pads):
        pd, ph, pw = pads
        if pd > 0:
            x = x[:, :-pd, :, :, :]
        if ph > 0:
            x = x[:, :, :-ph, :, :]
        if pw > 0:
            x = x[:, :, :, :-pw, :]
        return x



@register_keras_serializable(package="Custom")
class vol_2UNet3D2(SegModel):
    def __init__(self, filters=32, n_classes=3, **kwargs):
        super().__init__(filters=filters, n_classes=n_classes, **kwargs)

        # Encoder
        self.conv_0_1 = self.conv3D(filters, name='conv_0_1')
        self.conv_0_2 = self.conv3D(filters * 2, name='conv_0_2')
        self.max_0_1 = MaxPool3D(2, 2)

        self.conv_1_1 = self.conv3D(filters * 2, name='conv_1_1')
        self.conv_1_2 = self.conv3D(filters * 4, name='conv_1_2')
        self.max_1_2 = MaxPool3D(2, 2)

        self.conv_2_1 = self.conv3D(filters * 4, name='conv_2_1')
        self.conv_2_2 = self.conv3D(filters * 8, name='conv_2_2')
        self.max_2_3 = MaxPool3D(2, 2)

        self.conv_3_1 = self.conv3D(filters * 8, name='conv_3_1')
        self.conv_3_2 = self.conv3D(filters * 16, name='conv_3_2')

        # Decoder
        self.up_conv_3_4 = self.conv3DT(filters * 8, name='up_conv_3_4')
        self.conv_4_1 = self.conv3D(filters * 8, name='conv_4_1')
        self.conv_4_2 = self.conv3D(filters * 8, name='conv_4_2')

        self.up_conv_4_5 = self.conv3DT(filters * 4, name='up_conv_4_5')
        self.conv_5_1 = self.conv3D(filters * 4, name='conv_5_1')
        self.conv_5_2 = self.conv3D(filters * 4, name='conv_5_2')

        self.up_conv_5_6 = self.conv3DT(filters * 2, name='up_conv_5_6')
        self.conv_6_1 = self.conv3D(filters * 2, name='conv_6_1')
        self.conv_6_2 = self.conv3D(filters * 2, name='conv_6_2')

        # Output
        self.conv_out = Conv3D(
            n_classes, 1, padding='same', activation="softmax", name='conv_out'
        )

    def call(self, x ):
        # Encoder
        x, pads = self.pad_to_multiple(x, multiple=16)
        x0 = self.conv_0_1(x)
        x0 = self.conv_0_2(x0)
        p0 = self.max_0_1(x0)

        x1 = self.conv_1_1(p0)
        x1 = self.conv_1_2(x1)
        p1 = self.max_1_2(x1)

        x2 = self.conv_2_1(p1)
        x2 = self.conv_2_2(x2)
        p2 = self.max_2_3(x2)

        x3 = self.conv_3_1(p2)
        x3 = self.conv_3_2(x3)

        # Decoder
        u4 = self.up_conv_3_4(x3)
        u4 = self.concat_tensors(x2, u4)
        u4 = self.conv_4_1(u4)
        u4 = self.conv_4_2(u4)

        u5 = self.up_conv_4_5(u4)
        u5 = self.concat_tensors(x1, u5)
        u5 = self.conv_5_1(u5)
        u5 = self.conv_5_2(u5)

        u6 = self.up_conv_5_6(u5)
        u6 = self.concat_tensors(x0, u6)
        u6 = self.conv_6_1(u6)
        u6 = self.conv_6_2(u6)

        out= self.conv_out(u6)
        return crop_back(out, pads)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "n_classes": self.n_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)







@register_keras_serializable(package="Custom")
class vol_3UNet3D3(SegModel):
    def __init__(self, filters=32, n_classes=3, **kwargs):
        super().__init__(filters=filters, n_classes=n_classes, **kwargs)

        # Encoder
        self.conv_0_1 = self.conv3D(filters, name='conv_0_1')
        self.conv_0_2 = self.conv3D(filters * 2, name='conv_0_2')
        self.max_0_1 = MaxPool3D(2, 2)

        self.conv_1_1 = self.conv3D(filters * 2, name='conv_1_1')
        self.conv_1_2 = self.conv3D(filters * 4, name='conv_1_2')
        self.max_1_2 = MaxPool3D(2, 2)

        self.conv_2_1 = self.conv3D(filters * 4, name='conv_2_1')
        self.conv_2_2 = self.conv3D(filters * 8, name='conv_2_2')
        self.max_2_3 = MaxPool3D(2, 2)

        self.conv_3_1 = self.conv3D(filters * 8, name='conv_3_1')
        self.conv_3_2 = self.conv3D(filters * 16, name='conv_3_2')

        # Decoder
        self.up_conv_3_4 = self.conv3DT(filters * 8, name='up_conv_3_4')
        self.conv_4_1 = self.conv3D(filters * 8, name='conv_4_1')
        self.conv_4_2 = self.conv3D(filters * 8, name='conv_4_2')

        self.up_conv_4_5 = self.conv3DT(filters * 4, name='up_conv_4_5')
        self.conv_5_1 = self.conv3D(filters * 4, name='conv_5_1')
        self.conv_5_2 = self.conv3D(filters * 4, name='conv_5_2')

        self.up_conv_5_6 = self.conv3DT(filters * 2, name='up_conv_5_6')
        self.conv_6_1 = self.conv3D(filters * 2, name='conv_6_1')
        self.conv_6_2 = self.conv3D(filters * 2, name='conv_6_2')

        # Output
        self.conv_out = Conv3D(
            n_classes, 1, padding='same', activation="softmax", name='conv_out'
        )

    def call(self, x ):
        # Encoder
        # x, pads = self.pad_to_multiple(x, multiple=16)
        x0 = self.conv_0_1(x)
        x0 = self.conv_0_2(x0)
        p0 = self.max_0_1(x0)

        x1 = self.conv_1_1(p0)
        x1 = self.conv_1_2(x1)
        p1 = self.max_1_2(x1)

        x2 = self.conv_2_1(p1)
        x2 = self.conv_2_2(x2)
        p2 = self.max_2_3(x2)

        x3 = self.conv_3_1(p2)
        x3 = self.conv_3_2(x3)

        # Decoder
        u4 = self.up_conv_3_4(x3)
        u4 = self.concat_tensors(x2, u4)
        u4 = self.conv_4_1(u4)
        u4 = self.conv_4_2(u4)

        u5 = self.up_conv_4_5(u4)
        u5 = self.concat_tensors(x1, u5)
        u5 = self.conv_5_1(u5)
        u5 = self.conv_5_2(u5)

        u6 = self.up_conv_5_6(u5)
        u6 = self.concat_tensors(x0, u6)
        # u6=crop_back(u6, pads)
        u6 = self.conv_6_1(u6)
        u6 = self.conv_6_2(u6)
 
        return  self.conv_out(u6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "n_classes": self.n_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)












@register_keras_serializable(package="Custom")
class vol_VoxNetSeg(Model): 
    def __init__(self, filters=32, n_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.n_classes = n_classes 

        # 3D convolutions
        self.conv1 = layers.Conv3D(filters, 5, strides=1, padding="same",
                                   activation=tf.nn.leaky_relu)
        self.conv2 = layers.Conv3D(filters, 3, strides=1, padding="same",
                                   activation=tf.nn.leaky_relu)
        self.conv3 = layers.Conv3D(filters*2, 3, padding="same",
                                   activation=tf.nn.leaky_relu)
 
        self.out_conv = layers.Conv3D(n_classes, 1, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.out_conv(x)


    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



@register_keras_serializable(package="Custom")
class vol_FastFCN3D(Model):
    def __init__(self, filters=32, n_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.n_classes = n_classes 
        self.net = tf.keras.Sequential([
            layers.Conv3D(2, 1, activation="relu"),
            layers.Conv3D(2, 1, activation="relu"),
            layers.Conv3D(n_classes, 1, activation="softmax"),
        ])

    def call(self, x):
        return self.net(x)


    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
 

class model_choice: 
    def __init__(self, model_type=None, n_classes=3):
        self.model_type = model_type
        self.n_classes = n_classes

        self.models = {
            "vol_unet3d": {
                "class": vol_UNet3D,
                "params": {"multiple": 16, "margin": 2},
                "filters": 32
            },
            "vol_fastfcn3d": {
                "class": vol_FastFCN3D,
                "params": {"multiple": 16, "margin": 2},
                "filters": 32
            },
            "vol_vgg16_fcn3d": {
                "class": vol_VGG16_FCN3D,
                "params": {"multiple": 32, "margin": 2},
                "filters": 16
            },
            "vol_voxnetseg": {
                "class": vol_VoxNetSeg,
                "params": {"multiple": 16, "margin": 2},
                "filters": 32
            },
            "vol_2unet3d2": {
                "class": vol_2UNet3D2,
                "params": {"multiple": 16, "margin": 2},
                "filters": 32
            },
            "vol_3unet3d3": {
                "class": vol_3UNet3D3,
                "params": {"multiple": 16, "margin": 2},
                "filters": 32
            }
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









def voxel_generator(volumes, masks):
    for vol, mask in zip(volumes, masks):
        vol = vol.astype("float32")
        mask = mask.astype("float32")

        vol = vol[..., None]   # [D,H,W,1]
        mask = mask[..., None] # [D,H,W,1]

        yield vol, mask

def get_dataset(volumes, masks, batch_size=1):
    ds = tf.data.Dataset.from_generator(
        lambda: voxel_generator(volumes, masks),
        output_signature=(
            tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
        )
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def dice_loss(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.sigmoid(y_pred)  

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2. * intersection + eps) / (union + eps)
    return 1.0 - dice

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)



class aka_train:
    def __init__(self):
        pass

    def get_grad_back_prop(self, fun, model):
        with tf.GradientTape() as tape:
            loss = fun.loss_fun(model)
        grads = tape.gradient(loss, model.trainable_variables)
        return loss, grads

    @tf.function
    def train_PINN(self, optimizer, fun, model):
        loss, grads = self.get_grad_back_prop(fun, model)
        # print([g is None for g in grads])
        # print(grads)
        # for i, g in enumerate(grads):
        #     tf.print("grad", i, "norm:", tf.norm(g))

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss 
    
 

def weighted_categorical_crossentropy(class_weights,y_true, y_pred):
    class_weights = tf.constant(class_weights, dtype=tf.float32)


    y_true = tf.cast(y_true, tf.float32)
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(ce * weights) 


class LOSS:
    def __init__(self, volumes, masks,idx_originals, loss_mode='bce_dice', weight=None):
        self.volumes = volumes
        self.masks = masks
        self.loss_mode = loss_mode
        self.weight = weight
        self.idx_originals=idx_originals

    def loss_fun(self, model):
        loss = 0.0

        for vol, mask,idx_original,wei in zip(self.volumes, self.masks,self.idx_originals,self.weight):
 
            pred = model(vol)

            if self.loss_mode == 'mse':
                # loss += tf.reduce_mean(idx_original*tf.square(pred - mask))
                ce = tf.keras.losses.categorical_crossentropy(mask, pred)
                # print(ce.shape,'=====')
                loss += tf.reduce_mean( idx_original *ce)
            elif self.loss_mode == 'wbce': 
                y_true = tf.cast(mask, tf.float32)
                weightss = tf.reduce_sum(wei* y_true, axis=-1)
                ce = tf.keras.losses.categorical_crossentropy(y_true, pred)
                loss+= tf.reduce_mean(ce * weightss) 

            elif self.loss_mode == 'bce':
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                loss += tf.reduce_mean(idx_original*bce(mask, pred))

            elif self.loss_mode == 'dice':
                loss += tf.reduce_mean(idx_original*dice_loss(mask, pred))

            elif self.loss_mode == 'bce_dice':
                loss += tf.reduce_mean(idx_original*bce_dice_loss(mask, pred))

            else:
                raise ValueError(f"Unsupported loss_mode: {self.loss_mode}")

        return loss


 
def Get_iou(model, lab,mskls=None, ):
    iou_s = {0: [], 1: [], 2: []}
    
    for ii in range(len(lab)): 
        rhs0=mskls[ii].get_pred_rhs_crop(model,)[:,[1,0]]  
        for label in range(rhs0.shape[1]): 
            vertices_approx_index = np.where(np.argmax(rhs0, axis=1)  == label)[0] 
            
            
            if len(vertices_approx_index) > 0:
                ss = set(vertices_approx_index)
                sss = set(lab[ii][label])
                iou = len(ss.intersection(sss)) / len(ss.union(sss))
            else:
                iou = 0
            
            iou_s[label].append(iou)
    
    return iou_s

 



from sklearn.metrics import roc_curve, auc

def get_auc(model, mskls, rhs,adj=None,curv=None,get_model_one_hot=None,dend=None):
    auc_s = {0: [], 1: [], 2: []}
    y_true,score=[],[]
    
    for mskl,rh in zip(mskls,rhs):
        rhs0=mskl.get_pred_rhs_crop(model,)[:,[1,0]]  
        score.append(rhs0) 
        y_true.append(rh) 
    if score and y_true:
        yy_true=np.vstack(y_true)
        y_score=np.vstack(score)
        for label,(yy,sc,nm) in enumerate(zip(yy_true.T,y_score.T,['shaft','spine'])): 
            fpr, tpr, _ = roc_curve(yy, y_score=sc) 
            auc_s[label]=auc(fpr, tpr)
 
    
    return auc_s








class model_metric:
    def __init__(self, model, mskls, rhs_index=None, ):

        # metric names
        self.mmjj = ['iou', 'auc', 'dice']
        self.mmjjind = [f'{mmm}_ind' for mmm in self.mmjj]

        # model predictions
        score = [mskl.get_pred_rhs_crop(model,)[:,[1,0]] for mskl in mskls]    
        y_score = np.vstack(score)                  

        # predicted class indices
        score_ind = [np.argmax(yy, axis=1) for yy in score]
        # y_pred = np.concatenate(score_ind)

        # # true class indices
        # y_true = np.concatenate(rhs_index)
        y_pred= sum([list(mm) for mm in score_ind],[])
        y_true= sum([list(mm) for mm in rhs_index],[])

        # all classes present
        self.n_classes = sorted(list(set(y_true) | set(y_pred)))

        # initialize metrics dictionary
        self.metrics = {mm: {c: [] for c in self.n_classes}
                        for mm in self.mmjj + self.mmjjind}

        # global metrics
        global_metrics = self.get_metric(y_score=y_score,
                                         y_true=y_true,
                                         y_pred=y_pred)

        for met in self.mmjj:
            self.metrics[met] = global_metrics[met]

        # per‑sample metrics
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

            # AUC
            if np.sum(yy) > 0:
                fpr, tpr, _ = roc_curve(yy, sc)
                aucc = auc(fpr, tpr)
            else:
                aucc = 0

            # IoU and Dice
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









