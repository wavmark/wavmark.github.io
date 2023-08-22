import pdb
import time
import os
from utils import wm_add_v2, file_reader, model_util, wm_decode_v2, bin_util, my_parser, metric_util, path_util
from models import my_model
import torch
import uuid
import datetime
import numpy as np
import soundfile
from huggingface_hub import hf_hub_download


def load_model():
    resume_path = hf_hub_download(repo_id="M4869/WavMark",
                                  filename="step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl",
                                  )
    model = my_model.Model(16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8).to(device)
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    model_ckpt = checkpoint
    model.load_state_dict(model_ckpt, strict=True)
    model.eval()
    return model


def add_watermark(signal, audio_length_second, watermark_text):
    watermark_npy = np.array([int(i) for i in watermark_text])

    pattern_bit = wm_add_v2.fix_pattern[0:args.pattern_bit_length]

    watermark = np.concatenate([pattern_bit, watermark_npy])
    assert len(watermark) == 32
    signal_wmd, info = wm_add_v2.add_watermark(watermark, signal, 16000, 0.1, device, model, args.min_snr, args.max_snr)
    info["snr"] = metric_util.signal_noise_ratio(signal, signal_wmd)
    path_util.mk_parent_dir_if_necessary(args.output)
    soundfile.write(args.output, signal_wmd, 16000)

    print("Audio Length:%ds,Time Cost:%ds, Speed:x%.1f" % (audio_length_second, info["time_cost"],
                                                           audio_length_second / info["time_cost"]))

    print("Added %d watermark sections, skipped %d muted sections" % (
        info["encoded_sections"], info["skip_sections"]))

    if info["encoded_sections"] == 0:
        print("Warning! No watermarked added!! You can setup a lower min_snr value")


def decode_watermark(signal, audio_length_second):
    len_start_bit = args.pattern_bit_length
    start_bit = wm_add_v2.fix_pattern[0:len_start_bit]
    mean_result, info = wm_decode_v2.extract_watermark_v3_batch(
        signal,
        start_bit,
        0.1,
        16000,
        model,
        device, args.decode_batch_size)
    print("Audio length:%ds, Time Cost:%ds, Speed:x%.1f" % (audio_length_second, info["time_cost"],
                                                            audio_length_second / info["time_cost"]))

    if mean_result is None:
        print("No Watermark Found")
        return

    payload = mean_result[len_start_bit:]
    payload_str = "".join([str(i) for i in payload])
    print("Decoded Result:", payload_str)
    print("This result is found in the following time point:")
    for obj in info["results"]:
        print("%.1fs" % obj["start_time_position"])


if __name__ == "__main__":
    parser = my_parser.MyParser()
    parser.custom({
        "mode": "encode",  # encode\decode
        "input": "",
        "output": "",
        "watermark": "0010101010100111",

        "pattern_bit_length": 16,

        # if the watermarked audio has SNR > max_snr, we think the watermark is insufficient and perform "Repeated Encoding"
        "max_snr": 38,

        # if the watermarked audio has SNR < min_snr, we think it affects perception, thus skip this section
        "min_snr": 20,

        "decode_batch_size": 10,

        "min_time_length": 4,
    })
    args = parser.parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    assert args.mode in ["encode", "decode"], "wrong mode"

    # input check
    assert len(args.input) > 0, "you should setup an input path"
    signal, sr, audio_length_second = file_reader.read_as_single_channel_16k(args.input, 16000)
    assert audio_length_second > args.min_time_length, "audio time length should larger than %d seconds" % args.min_time_length

    # watermark check
    watermark_text = args.watermark
    if len(watermark_text) > 0 or args.mode == "encode":
        payload_length = 32 - args.pattern_bit_length
        assert len(watermark_text) == payload_length, "watermark length should be %d, current is %d" \
                                                      % (payload_length, len(watermark_text))
        assert set(watermark_text) == {'1', '0'}, "watermark should only has 0 and 1"

    if args.mode == "encode":
        # output check
        assert len(args.output) > 0, "you should setup an output path"
        assert args.output.lower().endswith(".wav"), "output should be a .wav filename"
        add_watermark(signal, audio_length_second, args.watermark)
    elif args.mode == "decode":
        decode_watermark(signal, audio_length_second)
