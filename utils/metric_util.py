import torch

import numpy as np


def calc_ber(watermark_decoded_tensor, watermark_tensor, threshold=0.5):
    watermark_decoded_binary = watermark_decoded_tensor >= threshold
    watermark_binary = watermark_tensor >= threshold
    ber_tensor = 1 - (watermark_decoded_binary == watermark_binary).to(torch.float32).mean()
    return ber_tensor


def to_equal_length(original, signal_watermarked):
    if original.shape != signal_watermarked.shape:
        print("警告！输入内容长度不一致", len(original), len(signal_watermarked))
        min_length = min(len(original), len(signal_watermarked))
        original = original[0:min_length]
        signal_watermarked = signal_watermarked[0:min_length]
    assert original.shape == signal_watermarked.shape
    return original, signal_watermarked


def signal_noise_ratio(original, signal_watermarked):
    # 数值越高越好，最好的结果为无穷大
    original, signal_watermarked = to_equal_length(original, signal_watermarked)
    noise_strength = np.sum((original - signal_watermarked) ** 2)
    if noise_strength == 0:  # 说明原始信号并未改变
        return np.inf
    signal_strength = np.sum(original ** 2)
    ratio = signal_strength / noise_strength

    # np.log10(1) == 0
    # 当噪声比信号强度还高时，信噪比就是负的
    # 如果ratio是0,那么 np.log10(0) 就是负无穷 -inf
    # 这里限定一个最小值,以免出现负无穷情况
    ratio = max(1e-10, ratio)
    return 10 * np.log10(ratio)


def batch_signal_noise_ratio(original, signal_watermarked):
    signal = original.detach().cpu().numpy()
    signal_watermarked = signal_watermarked.detach().cpu().numpy()
    tmp_list = []
    for s, swm in zip(signal, signal_watermarked):
        out = signal_noise_ratio(s, swm)
        tmp_list.append(out)
    return np.mean(tmp_list)


def calc_bce_acc(predictions, ground_truth, threshold=0.5):
    assert predictions.shape == ground_truth.shape

    # 将预测值转换为类别标签
    predicted_labels = (predictions >= threshold).float()

    # 计算准确率
    accuracy = ((predicted_labels == ground_truth).float().mean().item())
    return accuracy


def resample_to16k(data, old_sr):
    # 对数据进行重采样
    new_fs = 16000
    new_data = data[::int(old_sr / new_fs)]
    return new_data


import pypesq


def pesq(signal1, signal2, sr):
    signal1, signal2 = to_equal_length(signal1, signal2)

    # Perceptual Evaluation of Speech Quality
    # [−0.5 to 4.5], PESQ>3.5 时音频质量较好，>4.0基本上就听不到了
    # 函数只支持16k或8k的输入，因此在输入前校验采样率。由于这个指标计算的是可感知性，因此这里改变采样率和水印鲁棒性是无关的
    if sr != 16000:
        signal1 = resample_to16k(signal1, sr)
        signal2 = resample_to16k(signal2, sr)

    try:
        pesq = pypesq.pesq(signal1, signal2, 16000)
        # 可能会有错误：ValueError: ref is all zeros, processing error!
    except Exception as e:
        pesq = 0
        print("pesq计算错误:", e)

    return pesq
