import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, LayerNormalization, ZeroPadding2D

ac = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

def attention(x,H,W,dim,num_heads,sr_ratio,qkv_bias=False,attn_drop=0.0,proj_drop=0.0):
    dim = int(dim)
    head_dim = dim // num_heads
    units = num_heads * head_dim
    sqrt_of_units = math.sqrt(head_dim)
    q = Dense(units)(x)
    q = tf.reshape(q, shape=(tf.shape(q)[0], -1, num_heads, int(head_dim)))
    q = tf.transpose(q, perm=[0, 2, 1, 3])

    if sr_ratio > 1:
        x = tf.reshape(x, (tf.shape(x)[0], H, W, dim))
        x = Conv2D(filters=dim, kernel_size=sr_ratio, strides=sr_ratio)(x)
        x = tf.reshape(x, (tf.shape(x)[0], -1, dim))
        x = LayerNormalization(epsilon=1e-05)(x)

    k = Dense(units)(x)
    k = tf.reshape(k, shape=(tf.shape(k)[0], -1, num_heads, int(head_dim)))
    k = tf.transpose(k, perm=[0, 2, 1, 3])

    v = Dense(units)(x)
    v = tf.reshape(v, shape=(tf.shape(v)[0], -1, num_heads, int(head_dim)))
    v = tf.transpose(v, perm=[0, 2, 1, 3])


    attn = tf.matmul(q, k, transpose_b=True)
    scale = tf.cast(sqrt_of_units, dtype=attn.dtype)
    attn = tf.divide(attn, scale)

    attn = tf.nn.softmax(logits=attn, axis=-1)
    attn = tf.keras.layers.Dropout(attn_drop)(attn)
    x = tf.matmul(attn, v)
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, (tf.shape(x)[0], -1, units))
    x = Dense(dim)(x)
    x = Dropout(proj_drop)(x)

    return x


def dwconv(x, H, W, filters=768):
    x = tf.reshape(x, (tf.shape(x)[0], H, W, x.shape[-1]))
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding="same",
        groups=filters
    )(x)
    x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], x.shape[-1]))

    return x


def mlp(x,H,W,in_features,hidden_features=None,out_features=None,drop=0.0):
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    x = Dense(hidden_features)(x)
    x = dwconv(x, H, W, hidden_features)
    x = tf.keras.activations.gelu(x)
    x = Dropout(drop)(x)
    x = Dense(out_features)(x)
    x = Dropout(drop)(x)

    return x


def Droppath(x, drop_path, training=None):
    if training:
        keep_prob = 1 - drop_path
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
        random_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * random_tensor
    return x


def block(x,H,W,dim,num_heads,mlp_ratio=4.0,qkv_bias=False,drop=0.0,attn_drop=0.0,drop_path=0.0,sr_ratio=1):
    attn = attention(LayerNormalization(epsilon=1e-05)(x),H,W,dim,num_heads,sr_ratio,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop)
    drop_path_layer = (Droppath(attn, drop_path) if drop_path > 0.0 else tf.keras.layers.Layer()(attn))
    x = x + drop_path_layer
    norm2 = LayerNormalization(epsilon=1e-05)
    mlp_hidden_dim = int(dim * mlp_ratio)
    mlp_output = mlp(norm2(x), H, W, dim, mlp_hidden_dim, out_features=dim, drop=drop)
    drop_path_layer2 = (Droppath(mlp_output, drop_path) if drop_path > 0.0 else tf.keras.layers.Layer()(mlp_output))
    x = x + drop_path_layer2
    return x


def overlap_patch_embed(x, img_size=224, patch_size=7, stride=4, filters=768):
    x = ZeroPadding2D(padding=patch_size // 2)(x)
    x = Conv2D(filters=filters, kernel_size=patch_size,strides=stride, padding="VALID")(x)
    get_shapes = x.shape
    H = get_shapes[1]
    W = get_shapes[2]
    C = get_shapes[3]
    x = tf.reshape(x, (-1, H * W, C))
    x = LayerNormalization(epsilon=1e-05)(x)

    return (x, H, W)
