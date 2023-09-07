import torch
import numpy as np
import tqdm
import time
from utils import prob_util


def decode_trunck(trunck, model, device):
    with torch.no_grad():
        signal = torch.FloatTensor(trunck).to(device).unsqueeze(0)
        message = (model.decode(signal) >= 0.5).int()
        message = message.detach().cpu().numpy().squeeze()
    return message


def extract_watermark(data, start_bit, shift_range, num_point, model, device, batch_size=10,
                      shift_range_p=0.5):
    start_time = time.time()
    shift_step = int(shift_range * num_point * shift_range_p)

    # determine where to detect
    total_detections = (len(data) - num_point) // shift_step
    total_detect_points = [i * shift_step for i in range(total_detections)]

    total_batch_counts = len(total_detect_points) // batch_size + 1
    results = []
    for i in tqdm.tqdm(range(total_batch_counts)):
        detect_points = total_detect_points[i * batch_size:i * batch_size + batch_size]
        if len(detect_points) == 0:
            break
        current_batch = np.array([data[p:p + num_point] for p in detect_points])
        with torch.no_grad():
            signal = torch.FloatTensor(current_batch).to(device)
            batch_message = (model.decode(signal) >= 0.5).int().detach().cpu().numpy()
            for p, bit_array in zip(detect_points, batch_message):
                decoded_start_bit = bit_array[0:len(start_bit)]
                ber_start_bit = 1 - np.mean(start_bit == decoded_start_bit)
                num_equal_bits = np.sum(start_bit == decoded_start_bit)
                if ber_start_bit > 0:  # not exact match
                    continue
                results.append({
                    "sim": 1 - ber_start_bit,
                    "num_equal_bits": num_equal_bits,
                    "msg": bit_array,
                    "start_position": p,
                    "start_time_position": p / 16000
                })

    end_time = time.time()
    time_cost = end_time - start_time

    info = {
        "time_cost": time_cost,
        "results": results,
    }

    if len(results) == 0:
        return None, info

    results_1 = [i["msg"] for i in results if np.isclose(i["sim"], 1.0)]
    mean_result = (np.array(results_1).mean(axis=0) >= 0.5).astype(int)
    return mean_result, info


def extract_watermark_v2(data, watermark, shift_range, num_point, model, device, batch_size=10,
                         shift_range_p=0.5):
    start_time = time.time()
    shift_step = int(shift_range * num_point * shift_range_p)

    # determine where to detect
    total_detections = (len(data) - num_point) // shift_step
    total_detect_points = [i * shift_step for i in range(total_detections)]

    total_batch_counts = len(total_detect_points) // batch_size + 1
    results = []
    for i in tqdm.tqdm(range(total_batch_counts)):
        detect_points = total_detect_points[i * batch_size:i * batch_size + batch_size]
        if len(detect_points) == 0:
            break
        current_batch = np.array([data[p:p + num_point] for p in detect_points])
        with torch.no_grad():
            signal = torch.FloatTensor(current_batch).to(device)
            batch_message = (model.decode(signal) >= 0.5).int().detach().cpu().numpy()
            for p, bit_array in zip(detect_points, batch_message):
                num_not_equal_bits = np.sum(bit_array != watermark)
                prob = prob_util.calculate_probability(len(watermark), num_not_equal_bits)
                results.append({
                    "prob": prob,
                    "start_position": p,
                    "start_time_position": p / 16000
                })

    end_time = time.time()
    time_cost = end_time - start_time
    info = {
        "time_cost": time_cost,
        "results": results,
    }
    return info
