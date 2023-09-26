import os
import pickle

import torch
import numpy as np

TARGET_FLOAT = torch.float32
TARGET_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALL_FEATS_LEN = 64 + 1049 + 32 + 32 + 187

FEATS_SIZE = {
    'feat_summary': torch.Size([64, 768]),
    'feat_content': torch.Size([1049, 768]),
    'feat_2d': torch.Size([32, 1536]),
    'feat_3d': torch.Size([32, 2048]),
    'feat_audio': torch.Size([187, 512]),
}
def read_feature_from_file(meta_data, key):
    feature_files = meta_data[key]
    size = FEATS_SIZE[key]
    length = size[0]
    feats = []
    for f in feature_files:
        if os.path.exists(f):
            feat = pickle.load(open(f, 'rb'))
            feat = align_features(feat)
            feats.append(feat)
        else:
            feats.append(torch.zeros(size))
    feats = sequence_padding(feats, length)
    return feats

def align_features(feature_list):
    tensor_list = []
    for feature in feature_list:
        feature_type = type(feature)
        if feature_type is torch.Tensor:
            feature = feature.detach()
        elif feature_type is float or feature_type is int:
            feature = torch.FloatTensor(feature)
        else:
            feature = torch.from_numpy(feature)
        if feature.dtype != TARGET_FLOAT:
            feature = feature.to(dtype=TARGET_FLOAT)
        tensor_list.append(feature)
    return torch.stack(tensor_list, dim=0)

def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    shape = np.shape(inputs[0])
    pad_width = [(0, 0) for _ in shape]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = x.detach().numpy()
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        x = torch.from_numpy(x).to(device=TARGET_DEVICE)
        outputs.append(x)
    outputs = torch.stack(outputs, dim=0)
    return outputs

KEY_FEAT_SUMMARY = "feat_summary"
KEY_FEAT_CONTENT = "feat_content"
KEY_FEAT_2D = "feat_2d"
KEY_FEAT_3D = "feat_3d"
KEY_FEAT_AUDIO = "feat_audio"

def check_fill_feats(args, meta_data, inputs):
    if args.use_fusion:
        inputs[KEY_FEAT_SUMMARY] = read_feature_from_file(meta_data, KEY_FEAT_SUMMARY)
        inputs[KEY_FEAT_CONTENT] = read_feature_from_file(meta_data, KEY_FEAT_CONTENT)
        inputs[KEY_FEAT_2D] = read_feature_from_file(meta_data, KEY_FEAT_2D)
        inputs[KEY_FEAT_3D] = read_feature_from_file(meta_data, KEY_FEAT_3D)
        inputs[KEY_FEAT_AUDIO] = read_feature_from_file(meta_data, KEY_FEAT_AUDIO)