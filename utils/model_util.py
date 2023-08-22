import torch
import os
import json
import sys
from utils import pickle_util

history_array = []


def save_model(epoch, model, optimizer, file_save_path):
    dirpath = os.path.abspath(os.path.join(file_save_path, os.pardir))
    if not os.path.exists(dirpath):
        print("mkdir:", dirpath)
        os.makedirs(dirpath)

    opti = None
    if optimizer is not None:
        opti = optimizer.state_dict()

    torch.save(obj={
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': opti,
    }, f=file_save_path)

    history_array.append(file_save_path)


def save_model_v4(epoch, model, optimizer, file_save_path, discriminator):
    dirpath = os.path.abspath(os.path.join(file_save_path, os.pardir))
    if not os.path.exists(dirpath):
        print("mkdir:", dirpath)
        os.makedirs(dirpath)

    opti = None
    if optimizer is not None:
        opti = optimizer.state_dict()

    torch.save(obj={
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': opti,
        "discriminator": discriminator,
    }, f=file_save_path)

    history_array.append(file_save_path)


def delete_last_saved_model():
    if len(history_array) == 0:
        return
    last_path = history_array.pop()
    if os.path.exists(last_path):
        os.remove(last_path)
        print("delete model:", last_path)

    if os.path.exists(last_path + ".json"):
        os.remove(last_path + ".json")


def load_model(resume_path, model, optimizer=None, strict=True):
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("checkpoint loaded!")
    return start_epoch


def save_model_v2(model, args, model_save_name):
    model_save_path = os.path.join(args.model_save_folder, args.project, args.name, model_save_name)
    save_model(0, model, None, model_save_path)
    print("save:", model_save_path)


def save_project_info(args):
    run_info = {
        "cmd_str": ' '.join(sys.argv[1:]),
        "args": vars(args),
    }

    name = "run_info.json"
    folder = os.path.join(args.model_save_folder, args.project, args.name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    json_file_path = os.path.join(folder, name)
    with open(json_file_path, "w") as f:
        json.dump(run_info, f)

    print("save_project_info:", json_file_path)


def get_pkl_json(folder):
    names = [i for i in os.listdir(folder) if ".pkl.json" in i]
    assert len(names) == 1
    json_path = os.path.join(folder, names[0])
    obj = pickle_util.read_json(json_path)
    return obj


# 并行

def is_data_parallel_checkpoint(state_dict):
    return any(key.startswith('module.') for key in state_dict.keys())


def map_state_dict(state_dict):
    if is_data_parallel_checkpoint(state_dict):
        # 处理 DataParallel 添加的前缀 'module.'
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除前缀 'module.'
            new_state_dict[name] = v
        return new_state_dict
    return state_dict
