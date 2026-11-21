def dwt_transform(image):
    """
    实现离散小波变换 (DWT)。
    输入:
        image: 输入张量, 大小为 [batch_size, height, width, channels]
    输出:
        dwt_output: 包含 4 个分量的张量, 大小为 [batch_size, height // 2, width // 2, channels * 4]
    """
    if len(image.shape) != 4:
        raise ValueError(f"Input must have 4 dimensions [batch, height, width, channels], but got {image.shape}")

    # 分块操作
    x01 = image[:, 0::2, :, :] / 2
    x02 = image[:, 1::2, :, :] / 2
    x1 = x01[:, :, 0::2, :]
    x2 = x02[:, :, 0::2, :]
    x3 = x01[:, :, 1::2, :]
    x4 = x02[:, :, 1::2, :]

    # 计算四个分量
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    # 拼接分量
    # dwt_output = tf.concat([x_LL, x_HL, x_LH, x_HH], axis=-1)
    dwt_output = []
    dwt_output.append(x_LL)
    dwt_output.append(x_HL)
    dwt_output.append(x_LH)
    dwt_output.append(x_HH)
    return dwt_output
