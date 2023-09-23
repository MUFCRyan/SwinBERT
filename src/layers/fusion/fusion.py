import math
import sys

sys.path.append('../')

import torch
from torch import nn


class ActivateFun(nn.Module):
    def __init__(self, configs):
        super(ActivateFun, self).__init__()
        self.activate_fun = configs.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class Linear(nn.Module):
    def __init__(self, config, input_dim):
        super(Linear, self).__init__()
        dim = config.dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim, bias=False),
            ActivateFun(config)
        )

    def forward(self, features):
        return self.encoder(features)


class TransformerEncoder(nn.Module):
    def __init__(self, config, num_layers):
        super(TransformerEncoder, self).__init__()
        dim = config.dim
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=dim // 64, dim_feedforward=dim * 2)
        transformer_encoder_layer.linear1.bias.requires_grad = False
        transformer_encoder_layer.linear2.bias.requires_grad = False
        encoder_norm = nn.LayerNorm(dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                         num_layers=num_layers,
                                                         norm=encoder_norm)

    def forward(self, features):
        return self.transformer_encoder(features)


class TextModel(nn.Module):
    def __init__(self, config):
        super(TextModel, self).__init__()
        dim = config.dim
        self.summary_encoder = Linear(config, 768)
        self.content_encoder = Linear(config, 768)
        self.transformer_encoder = TransformerEncoder(config, num_layers=config.text_tran_num_layers)
        self.bi_lstm = nn.LSTM(input_size=dim, hidden_size=dim // 2, batch_first=True, bidirectional=True)

    def forward(self, feature_summary, feature_content):
        feat_list = []
        if feature_summary is not None:
            feature_summary = self.summary_encoder(feature_summary)
            feat_list.append(feature_summary)
        feature_content = self.content_encoder(feature_content)
        feat_list.append(feature_content)
        feature_output = torch.cat(feat_list, dim=1)
        # feature_output, _ = self.bi_lstm(feature_output)
        feature_output = self.transformer_encoder(feature_output)
        return feature_output


class VisualModel(nn.Module):
    # 2d dim = n * 1536, 3d dim = n * 2048
    def __init__(self, config):
        super(VisualModel, self).__init__()
        self.encoder_2d = Linear(config, 1536)
        self.encoder_3d = Linear(config, 2048)

        self.transformer_encoder = TransformerEncoder(config, num_layers=config.visual_tran_num_layers)

    def forward(self, feature_2d, feature_3d):
        feat_list = []
        if feature_2d is not None:
            feature_2d = self.encoder_2d(feature_2d)
            feat_list.append(feature_2d)
        if feature_3d is not None:
            feature_3d = self.encoder_3d(feature_3d)
            feat_list.append(feature_3d)
        feature_output = None
        if len(feat_list) > 0:
            feature_output = torch.cat(feat_list, dim=1)
            feature_output = self.transformer_encoder(feature_output)
        return feature_output


class AudioModel(nn.Module):
    def __init__(self, config):
        super(AudioModel, self).__init__()
        self.encoder = Linear(config, 512)
        self.transformer_encoder = TransformerEncoder(config, num_layers=config.audio_tran_num_layers)

    def forward(self, feature):
        feature_output = None
        if feature is not None:
            feature_output = self.encoder(feature)
            feature_output = self.transformer_encoder(feature_output)
        return feature_output


class FuseModel(nn.Module):
    def __init__(self, config):
        super(FuseModel, self).__init__()
        self.text_model = TextModel(config)
        self.visual_model = VisualModel(config)
        self.audio_model = AudioModel(config)
        self.transformer_encoder = TransformerEncoder(config, num_layers=config.fuse_tran_num_layers)

    def forward(self, feat_summary, feat_content, feat_2d, feat_3d, feat_audio):
        feat_list = []
        feat_text = self.text_model(feat_summary, feat_content)
        feat_list.append(feat_text)
        feat_visual = self.visual_model(feat_2d, feat_3d)
        if feat_visual is not None:
            feat_list.append(feat_visual)
        feat_audio = self.audio_model(feat_audio)
        if feat_audio is not None:
            feat_list.append(feat_audio)
        all_features = torch.cat(feat_list, dim=1)
        # shape = all_features.shape
        # mask = torch.ones((shape[0], shape[1]), dtype=torch.float32)
        feature_output = self.transformer_encoder(all_features)
        return feature_output