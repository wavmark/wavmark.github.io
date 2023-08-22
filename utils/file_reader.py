import os
import soundfile
import librosa
import resampy


def is_wav_file(filename):
    # 获取文件扩展名
    file_extension = os.path.splitext(filename)[1]

    # 判断文件扩展名是否为'.wav'或'.WAV'
    return file_extension.lower() == ".wav"


import numpy as np


def read_as_single_channel_16k(audio_file, def_sr, verbose=True, aim_second=None):
    assert os.path.exists(audio_file)

    file_extension = os.path.splitext(audio_file)[1].lower()

    if file_extension == ".mp3":
        data, origin_sr = librosa.load(audio_file, sr=None)
    elif file_extension in [".wav", ".flac"]:
        data, origin_sr = soundfile.read(audio_file)
    else:
        raise Exception("unsupported file:" + file_extension)

    # 通道数
    if len(data.shape) == 2:
        left_channel = data[:, 0]
        if verbose:
            print("Warning! the input audio has multiple chanel, this tool only use the first channel!")
        data = left_channel

    # 采样率
    if origin_sr != def_sr:
        data = resampy.resample(data, origin_sr, def_sr)
        if verbose:
            print("Warning! The original samplerate is not 16Khz; the watermarked audio will be re-sampled to 16KHz")

    sr = def_sr
    audio_length_second = 1.0 * len(data) / sr
    # if verbose:
    #     print("input length :%d second" % audio_length_second)

    if aim_second is not None:
        signal = data
        assert len(signal) > 0
        current_second = len(signal) / sr
        if current_second < aim_second:
            repeat_count = int(aim_second / current_second) + 1
            signal = np.repeat(signal, repeat_count)
        data = signal[0:sr * aim_second]

    return data, sr, audio_length_second


def read_as_single_channel(file, aim_sr):
    if file.endswith(".mp3"):
        data, sr = librosa.load(file, sr=aim_sr)  # 这里默认就是会转换为输入的sr
    else:
        data, sr = soundfile.read(file)

    if len(data.shape) == 2:  # 双声道
        data = data[:, 0]  # 只要第一个声道

    # 然后再切换sr,因为soundfile可能读取出一个双通道的东西
    if sr != aim_sr:
        data = resampy.resample(data, sr, aim_sr)
    return data
