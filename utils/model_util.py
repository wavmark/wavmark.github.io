from models import my_model
import torch
from huggingface_hub import hf_hub_download


def load_model(device):
    resume_path = hf_hub_download(repo_id="M4869/WavMark",
                                  filename="step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl",
                                  )
    model = my_model.Model(16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8).to(device)
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    model_ckpt = checkpoint
    model.load_state_dict(model_ckpt, strict=True)
    model.eval()
    return model
