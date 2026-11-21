from os import fspath

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Lambda, MaxPooling2D,
    Permute, ReLU, Reshape, UpSampling2D, add, concatenate, multiply,
)
from tensorflow.keras.models import Model

from .wavelet import dwt_transform
from .attention import ac, block, overlap_patch_embed

k = 32

padding = 'same'

bias = False




def local_affinity(x, dilations=[1]):
    """LocalAffinity 操作

    Args:
        x: 输入张量，形状为 [B, C, H, W]
        dilations: 空洞率列表

    Returns:
        张量，形状为 [B, C, P, H, W]
    """
    # Equivalent fixed 8-neighbour shifts. Slicing preserves the original
    # zero-padded forward values and supports dilated gradients on TF 2.10 CPU,
    # whose native dilated Conv2DBackpropInput is unavailable on Windows.
    h, w = x.shape[1:3]
    neighbours = []
    for d in dilations:
        padded = tf.pad(x, [[0, 0], [d, d], [d, d], [0, 0]])
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbours.append(padded[:, d + dy*d:d + dy*d + h,
                                      d + dx*d:d + dx*d + w, :])
    return tf.stack(neighbours, axis=-1)


def fbf_module(feature, num_iter=5, dilations=None):
    """FBFModule 操作

    Args:
        feature: 输入特征，形状为 [B, C, H, W]
        num_iter: 迭代次数
        dilations: 空洞率列表

    Returns:
        张量，形状为 [B, C, H, W]
    """
    if dilations is None:
        dilations = [1]
    n, h, w, c = feature.shape

    for _ in range(num_iter):
        f = local_affinity(feature, dilations)  # [B, C, P, H, W]

        abs_diff = tf.abs(tf.expand_dims(feature, axis=4) - f)
        aff = tf.exp(-tf.reduce_mean(abs_diff, axis=3, keepdims=True))

        aff = aff / tf.reduce_sum(aff, axis=4, keepdims=True)  # [B, 1, P, H, W]

        feature = tf.reduce_sum(f * aff, axis=4)

    return feature

def sobel_kernels(channels):
    sobel_x = np.zeros((3, 3, channels, channels), dtype=np.float32)
    sobel_y = np.zeros((3, 3, channels, channels), dtype=np.float32)
    sobel_45 = np.zeros((3, 3, channels, channels), dtype=np.float32)
    sobel_135 = np.zeros((3, 3, channels, channels), dtype=np.float32)

    for i in range(channels):
        sobel_x[:, :, i, i] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y[:, :, i, i] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        sobel_45[:, :, i, 0] = [[0, 2, 1],
                                [-2, 0, 2],
                                [-1, -2, 0]]

        sobel_135[:, :, i, 0] = [[1, 2, 0],
                                 [2, 0, -2],
                                 [0, -2, -1]]

    return tf.constant(sobel_x, dtype=tf.float32), tf.constant(sobel_y, dtype=tf.float32), tf.constant(sobel_45, dtype=tf.float32), tf.constant(sobel_135, dtype=tf.float32)

def sobel_filter(image):
    channels = image.shape[-1]
    sobel_x, sobel_y, sobel_45, sobel_135 = sobel_kernels(channels)

    edges_x = tf.nn.conv2d(image, sobel_x, strides=1, padding="SAME")
    edges_y = tf.nn.conv2d(image, sobel_y, strides=1, padding="SAME")

    edges = (tf.abs(edges_x) + tf.abs(edges_y)) / 2.0
    return edges

def erode(image, kernel_size=3):
    eroded_image = -tf.nn.max_pool2d(-image, ksize=kernel_size, strides=1, padding='SAME')
    return eroded_image

def dilate(image, kernel_size=3):
    dilated_image = tf.nn.max_pool2d(image, ksize=kernel_size, strides=1, padding='SAME')
    return dilated_image

def closing_operation(image, kernel_size=3):
    dilated = dilate(image, kernel_size)
    return erode(dilated, kernel_size)

def fp_module(input_tensor, in_c, out_c, num_iter, dilations):
    fbf_output = fbf_module(input_tensor, num_iter=num_iter, dilations=dilations)

    edges = sobel_filter(input_tensor)
    edges = tf.keras.activations.tanh(edges)

    closed_edges = closing_operation(edges, kernel_size=5)

    out = fbf_output + input_tensor * closed_edges

    return out,closed_edges,fbf_output

def edgemodule(input_tensor):
    edges = sobel_filter(input_tensor)

    edges = tf.keras.activations.tanh(edges)
    closed_edges = closing_operation(edges, kernel_size=5)
    out = input_tensor * closed_edges

    return out

def edge_enhance_module(input,
                        channel,
                        d=[1, 3, 5],
                        depth=1,
                        window=True,
                        filter_mode="normal"):
    """
    filter_mode:
        "normal" -> 正常
        "detach" -> filtering branch stop_gradient
        "zero"   -> filtering branch置零
    """
    cc1 = Conv2D(channel, 3, activation='relu', padding=padding,
                 kernel_initializer='he_normal', use_bias=bias)(input)
    cc1 = BatchNormalization()(cc1)
    cc1 = Conv2D(channel, 3, activation='relu', padding=padding,
                 kernel_initializer='he_normal', use_bias=bias)(cc1)
    cc1 = BatchNormalization()(cc1)

    cc2, edge, fbf = fp_module(input, channel, channel, 1, dilations=d)

    if filter_mode == "detach":
        cc2 = tf.stop_gradient(cc2)
    elif filter_mode == "zero":
        cc2 = tf.zeros_like(cc2)

    if window:
        cc3 = WTrans(input, embed_dim=channel, num_heads=8, ff_dim=channel, window_size=8)
    else:
        cc3, H, W = overlap_patch_embed(input, patch_size=3, stride=1, filters=channel)
        cc3 = block(cc3, H, W, dim=channel,
                    num_heads=2,
                    mlp_ratio=2.0,
                    qkv_bias=False,
                    drop=0.0,
                    attn_drop=0.0,
                    sr_ratio=1)
        cc3 = tf.keras.layers.LayerNormalization(epsilon=1e-05)(cc3)
        cc3 = tf.reshape(cc3, (-1, H, W, cc3.shape[-1]))

    c = concatenate([cc1, cc2, cc3], axis=3)
    c = Conv2D(2 * channel, 1, activation='relu', padding=padding,
               kernel_initializer='he_normal', use_bias=bias)(c)
    c = BatchNormalization()(c)

    debug = {
        "conv": cc1,
        "filter": cc2,
        "edge": edge,
        "fbf": fbf,
        "trans": cc3,
        "fused": c
    }

    return c, debug

def multiscale_decoder(cc2,cc3,cc4,channel):

    s4, H, W = overlap_patch_embed(cc4, patch_size=3, stride=1, filters=channel)
    s4 = block(s4, H, W, dim=channel,
                num_heads=2,
                mlp_ratio=2.0,
                qkv_bias=False,
                drop=0.0,
                attn_drop=0.0,
                sr_ratio=1)
    s4 = tf.keras.layers.LayerNormalization(epsilon=1e-05)(s4)
    s4 = tf.reshape(s4, (-1, H, W, s4.shape[-1]))
    s3, H, W = overlap_patch_embed(cc3, patch_size=3, stride=2, filters=channel/2)
    s3 = block(s3, H, W, dim=channel / 2,
                num_heads=2,
                mlp_ratio=2.0,
                qkv_bias=False,
                drop=0.0,
                attn_drop=0.0,
                sr_ratio=1)
    s3 = tf.keras.layers.LayerNormalization(epsilon=1e-05)(s3)
    s3 = tf.reshape(s3, (-1, H, W, s3.shape[-1]))


    s = concatenate([s4,s3],axis=-1)
    s_deep = UpSampling2D(size=(2, 2), interpolation='bilinear')(s)
    cc3_e = edgemodule(cc3)
    s = concatenate([s_deep, cc3_e],axis=-1)


    s2 = channel_attention(cc2)
    s2 = spatial_attention(s2)
    s2 = WavestarDown(s2)

    s2_e = edgemodule(s2)
    s = concatenate([s,s2_e],axis=-1)
    s_before = Conv2D(channel / 2, 1, activation='relu', padding=padding, kernel_initializer=ac, use_bias=bias)(s)

    decoder_fp_out, decoder_fp_edge, decoder_fp_fbf = fp_module(
        s_before,
        int(channel / 2),
        int(channel / 2),
        num_iter=1,
        dilations=[1, 3, 5, 7]
    )

    s = add([s_before, decoder_fp_out])

    decoder_fp_debug = {
        "input": s_before,
        "edge": decoder_fp_edge,
        "fbf": decoder_fp_fbf,
        "filter": decoder_fp_out
    }

    return s, s_deep, concatenate([cc3_e, s2_e], axis=-1), s_before, decoder_fp_debug

def WTrans(x, embed_dim, num_heads, ff_dim, window_size=8):
    """
    基于窗口的 Transformer 层
    Args:
        x: 输入张量，形状为 (batch_size, height, width, channels)
        embed_dim: 特征嵌入维度
        num_heads: 自注意力机制的头数
        ff_dim: 前馈网络的隐藏层维度
        window_size: 每个窗口的尺寸（默认为 8x8）
    Returns:
        输出张量，形状与输入相同
    """
    batch_size, height, width, channels = x.shape[0], x.shape[1], x.shape[2], x.shape[3]


    windows = tf.reshape(x, [
        tf.shape(x)[0],
        height // window_size, window_size,
        width // window_size, window_size,
        channels
    ])

    windows = tf.transpose(windows, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, [-1, window_size * window_size, channels])

    attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    attn_output = attention_layer(windows, windows)
    out1 = layers.LayerNormalization(epsilon=1e-6)(windows + attn_output)  # 残差连接和归一化

    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),  # 激活函数
        layers.Dense(embed_dim),
    ])
    ffn_output = ffn(out1)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)  # 残差连接和归一化

    out2 = tf.reshape(out2, [
        tf.shape(x)[0],
        height // window_size, width // window_size,
        window_size, window_size, channels
    ])

    out2 = tf.transpose(out2, perm=[0, 1, 3, 2, 4, 5])
    out2 = tf.reshape(out2, [tf.shape(x)[0], height, width, channels])

    return out2

def channel_attention(input_feature, ratio=4):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding=padding,
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def cov2d(input,channel, size=3, activation='relu', padding=padding, kernel_initializer=ac, use_bias=bias):
    c = Conv2D(channel, size, activation=None, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias)(input)
    c = BatchNormalization(momentum=0.99, epsilon=1e-3, trainable=True)(c)
    c = ReLU()(c)
    return c

def WavestarDown(input, ratio=3):
    B, H, W, C = input.shape
    x11 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input)
    x22 = dwt_transform(input)
    x1 = concatenate([x11, (x22[1]+ x22[2]+ x22[3])], axis=-1)
    x1 = cov2d(x1,C*ratio,1)
    x2 = cov2d(x22[0],C*ratio,1)
    x = x1 * x2
    x = cov2d(x,C,1)
    x = Conv2D(filters=C, kernel_size=7, strides=1, padding='same', groups=C)(x)
    x = BatchNormalization()(x)
    x = x + x11 +x22[0]
    return x

def GDNet2(scale=256, num_class=6, k=k, filter_mode="normal", return_debug=False):
    inputs = Input((scale, scale, 3), name="input_image")

    input_feat = Conv2D(k, 3, activation='relu', padding=padding,
                        kernel_initializer='he_normal', use_bias=bias, strides=2)(inputs)
    cc1 = BatchNormalization()(input_feat)

    cc2, dbg2 = edge_enhance_module(
        cc1, k, [1, 3, 5, 7], depth=2, filter_mode=filter_mode
    )

    cc3 = WavestarDown(cc2)
    cc3, dbg3 = edge_enhance_module(
        cc3, 2 * k, [1, 3, 5, 7], depth=2, filter_mode=filter_mode
    )

    cc4 = WavestarDown(cc3)
    cc4, dbg4 = edge_enhance_module(
        cc4, 4 * k, [1, 3, 5, 7], depth=2, filter_mode=filter_mode
    )

    cc5, s_deep, s_e, s_before, decoder_fp_debug = multiscale_decoder(cc2, cc3, cc4, 8 * k)
    cc = UpSampling2D(size=(4, 4), interpolation='bilinear')(cc5)

    if num_class > 2:
        pred = Conv2D(num_class, 1, activation=None, name="logits")(cc)
        pred = softmax(pred)
    else:
        pred = Conv2D(2, 1, activation='sigmoid', name="logits")(cc)

    if not return_debug:
        return Model(inputs=inputs, outputs=pred, name=f"GDNet2_{filter_mode}")

    debug_outputs = {
        "pred": pred,

        "stage2_conv": dbg2["conv"],
        "stage2_filter": dbg2["filter"],
        "stage2_edge": dbg2["edge"],
        "stage2_fbf": dbg2["fbf"],
        "stage2_trans": dbg2["trans"],
        "stage2_fused": dbg2["fused"],

        "stage3_conv": dbg3["conv"],
        "stage3_filter": dbg3["filter"],
        "stage3_edge": dbg3["edge"],
        "stage3_fbf": dbg3["fbf"],
        "stage3_trans": dbg3["trans"],
        "stage3_fused": dbg3["fused"],

        "stage4_conv": dbg4["conv"],
        "stage4_filter": dbg4["filter"],
        "stage4_edge": dbg4["edge"],
        "stage4_fbf": dbg4["fbf"],
        "stage4_trans": dbg4["trans"],
        "stage4_fused": dbg4["fused"],

        "decoder_s_deep": s_deep,
        "decoder_s_e": s_e,
        "decoder_s_before": s_before,

        "decoder_fp_input": decoder_fp_debug["input"],
        "decoder_fp_edge": decoder_fp_debug["edge"],
        "decoder_fp_fbf": decoder_fp_debug["fbf"],
        "decoder_fp_filter": decoder_fp_debug["filter"],
    }

    return Model(inputs=inputs, outputs=debug_outputs, name=f"GDNet2_debug_{filter_mode}")

def SGFWSeg(scale=256, num_class=6, k=k, filter_mode="normal", return_debug=False):
    """Paper-facing alias; the underlying graph is the original GDNet2 graph."""
    return GDNet2(scale=scale, num_class=num_class, k=k,
                  filter_mode=filter_mode, return_debug=return_debug)


def load_sgfwseg(weights_path, *, scale=256, num_class=None):
    """Build SGF-WSeg and load all weights in their saved layer order.

    The provided legacy HDF5 checkpoints store the class count in the final
    convolution's shape. For other checkpoint formats, pass num_class explicitly.
    """
    weights_path = fspath(weights_path)
    if num_class is None:
        with h5py.File(weights_path, "r") as checkpoint:
            group = checkpoint.get("model_weights", checkpoint)
            for layer_name in reversed(group.attrs.get("layer_names", ())):
                layer = group[layer_name]
                weight_names = layer.attrs.get("weight_names", ())
                if not len(weight_names):
                    continue
                shapes = [layer[name].shape for name in weight_names]
                if len(shapes) == 2:
                    kernel, bias_shape = shapes
                    if len(kernel) == 4 and kernel[:2] == (1, 1) and bias_shape == kernel[-1:]:
                        num_class = kernel[-1]
                break
        if num_class is None:
            raise ValueError("Cannot infer the class count from this checkpoint. Pass num_class explicitly.")

    model = SGFWSeg(scale=scale, num_class=num_class)
    model.load_weights(weights_path)
    return model
