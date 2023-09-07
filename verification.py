from utils import wm_add_util, file_reader, wm_decode_util, my_parser, metric_util, path_util
import torch
import numpy as np
import soundfile
from utils import model_util, prob_util
import math


def add_watermark(signal, audio_length_second, watermark_text):
    watermark = np.array([int(i) for i in watermark_text])
    assert len(watermark) == 32
    signal_wmd, info = wm_add_util.add_watermark(watermark, signal, 16000, 0.1, device, model, args.min_snr,
                                                 args.max_snr)
    info["snr"] = metric_util.signal_noise_ratio(signal, signal_wmd)
    path_util.mk_parent_dir_if_necessary(args.output)
    soundfile.write(args.output, signal_wmd, 16000)
    print("Audio Length:%ds,Time Cost:%ds, Speed:x%.1f" % (audio_length_second, info["time_cost"],
                                                           audio_length_second / info["time_cost"]))

    print("Added %d watermark segments, skipped %d muted segments" % (
        info["encoded_sections"], info["skip_sections"]))

    if info["encoded_sections"] == 0:
        print("Warning! No watermarked added!! You can setup a lower min_snr value")


def calculate_probability(bit_length, not_equal_count):
    total_cases = 2 ** (bit_length * 2)

    combinations = math.comb(bit_length, not_equal_count)

    equal_cases = combinations * (2 ** not_equal_count) * (2 ** (bit_length - not_equal_count))

    probability = equal_cases / total_cases
    return probability


def decode_watermark(signal, audio_length_second, watermark_text):
    watermark = np.array([int(i) for i in watermark_text])
    assert len(watermark) == 32

    info = wm_decode_util.extract_watermark_v2(
        signal,
        watermark,
        0.1,
        16000,
        model,
        device, args.decode_batch_size)
    print("Audio length:%ds, Time Cost:%ds, Speed:x%.1f" % (audio_length_second, info["time_cost"],
                                                            audio_length_second / info["time_cost"]))

    results = info["results"]
    if len(results) == 0:
        print("No Watermark Found")
        return

    # sort...
    results.sort(key=lambda x: x['prob'])
    results_0 = results[0]
    print("以%f的出错概率认定音频中存在水印" % results_0['prob'])


if __name__ == "__main__":
    parser = my_parser.MyParser()
    parser.custom({
        "mode": "encode",  # encode\decode
        "input": "",
        "output": "",
        "watermark": "00111111100101101111110101110100",
        "max_snr": 38,
        "min_snr": 20,
        "decode_batch_size": 10,
        "min_time_length": 4,
    })
    args = parser.parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_util.load_model(device)
    assert args.mode in ["encode", "decode"], "wrong mode"

    # input check
    assert len(args.input) > 0, "you should setup an input path"
    signal, sr, audio_length_second = file_reader.read_as_single_channel_16k(args.input, 16000)
    assert audio_length_second > args.min_time_length, "audio time length should larger than %d seconds" % args.min_time_length

    # watermark check
    watermark_text = args.watermark
    assert len(watermark_text) == 32, "watermark length should be %d, current is %d" \
                                      % (32, len(watermark_text))
    assert set(watermark_text) == {'1', '0'}, "watermark should only has 0 and 1"

    if args.mode == "encode":
        assert len(args.output) > 0, "you should setup an output path"
        assert args.output.lower().endswith(".wav"), "output should be a .wav filename"
        add_watermark(signal, audio_length_second, args.watermark)
    elif args.mode == "decode":
        decode_watermark(signal, audio_length_second, args.watermark)
