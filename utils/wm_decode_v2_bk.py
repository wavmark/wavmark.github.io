import torch
import numpy as np
import tqdm
import time


def decode_trunck(trunck, model, device):
    with torch.no_grad():
        signal = torch.FloatTensor(trunck).to(device).unsqueeze(0)
        message = (model.decode(signal) >= 0.5).int()
        message = message.detach().cpu().numpy().squeeze()
    return message


def extract_watermark_v3_batch(data, start_bit, shift_range, num_point, model, device, batch_size=10,
                               shift_range_p=0.5):
    start_time = time.time()
    # 1.确定步长
    shift_step = int(shift_range * num_point * shift_range_p)

    # 2.确定在哪里执行采样
    total_detections = (len(data) - num_point) // shift_step
    total_detect_points = [i * shift_step for i in range(total_detections)]

    # 3.构建batch
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

                if ber_start_bit > 0.2:
                    continue
                results.append({
                    "sim": 1 - ber_start_bit,
                    "num_equal_bits": num_equal_bits,
                    "msg": bit_array,
                    "start_position": p,
                })

    end_time = time.time()
    time_cost = end_time - start_time

    info = {
        "time_cost": time_cost,
        "results": results,
        "perfectly_matched_section": 0,
        "not_perfect_match_sim": 0,
        "not_perfect_match_prob": 0,
    }

    if len(results) == 0:
        return None, info

    #
    best_val = sorted(results, key=lambda x: x["sim"], reverse=True)[0]
    if np.isclose(1.0, best_val["sim"]):
        # 那么对所有为1.0的进行求平均
        results_1 = [i["msg"] for i in results if np.isclose(i["sim"], 1.0)]
        info["perfectly_matched_section"] = len(results_1)
        mean_result = (np.array(results_1).mean(axis=0) >= 0.5).astype(int)
    else:
        mean_result = best_val["msg"]
        info["not_perfect_match_sim"] = best_val["sim"]
        info["not_perfect_match_prob"] = best_val["num_equal_bits"]

    return mean_result, info
