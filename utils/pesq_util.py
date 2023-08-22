import pypesq
import numpy as np


def batch_pesq(batch_signal, batch_signal_wmd):
    batch_signal1 = batch_signal.detach().cpu().numpy()
    batch_signal2 = batch_signal_wmd.detach().cpu().numpy()
    pesq_array = []
    for signal1, signal2 in zip(batch_signal1, batch_signal2):
        try:
            pesq = pypesq.pesq(signal1, signal2, 16000)
            #可能会有错误：ValueError: ref is all zeros, processing error!

        except Exception as e:
            print(e)

            continue
        if np.isnan(pesq):
            print("pesq is nan!")
            continue
        pesq_array.append(pesq)

    if len(pesq_array) > 0:
        return np.mean(pesq_array)
    return -1
