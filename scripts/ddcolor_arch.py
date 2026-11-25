"""Standalone Apache-2.0 PyTorch architecture for official piddnad/DDColor checkpoints.

Matches the exact state_dict module hierarchy of HuggingFace checkpoints:
  - ``piddnad/ddcolor_modelscope`` (convnext-l)
  - ``piddnad/ddcolor_paper_tiny`` (convnext-t)
  - ``piddnad/ddcolor_artistic`` (convnext-l)

without requiring ``basicsr`` or an external ``ddcolor`` wheel install.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

_CONVNEXT_CONFIGS: dict[str, tuple[list[int], list[int]]] = {
    "convnext-t": ([3, 3, 9, 3], [96, 192, 384, 768]),
    "convnext-s": ([3, 3, 27, 3], [96, 192, 384, 768]),
    "convnext-b": ([3, 3, 27, 3], [128, 256, 512, 1024]),
    "convnext-l": ([3, 3, 27, 3], [192, 384, 768, 1536]),
}


class _LayerNorm2d(nn.Module):
    """LayerNorm supporting either channels_last or channels_first tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm2d(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class _ConvNeXtBackbone(nn.Module):
    def __init__(self, depths: list[int], dims: list[int]) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            _LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    _LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[_ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.norm0 = _LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first")
        self.norm1 = _LayerNorm2d(dims[1], eps=1e-6, data_format="channels_first")
        self.norm2 = _LayerNorm2d(dims[2], eps=1e-6, data_format="channels_first")
        self.norm3 = _LayerNorm2d(dims[3], eps=1e-6, data_format="channels_first")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        norms = (self.norm0, self.norm1, self.norm2, self.norm3)
        feats: list[torch.Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(norms[i](x))
        return feats


class _Encoder(nn.Module):
    def __init__(self, encoder_name: str = "convnext-l") -> None:
        super().__init__()
        depths, dims = _CONVNEXT_CONFIGS.get(encoder_name, _CONVNEXT_CONFIGS["convnext-l"])
        self.arch = _ConvNeXtBackbone(depths=depths, dims=dims)
        self.dims = dims

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.arch(x)


class _PixelShuffleICNR(nn.Module):
    def __init__(self, ni: int, nf: int, scale: int = 2, bias: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ni, nf * (scale**2), kernel_size=1, bias=bias)),
            nn.BatchNorm2d(nf * (scale**2)),
            nn.ReLU(inplace=True),
        )
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.shuf(self.conv(x))
        if self.scale == 2:
            out = self.blur(self.pad(out))
        return out


class _LastPixelShuffleICNR(nn.Module):
    def __init__(self, ni: int, nf: int, scale: int = 4) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ni, nf * (scale**2), kernel_size=1, bias=True)),
            nn.ReLU(inplace=True),
        )
        self.shuf = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuf(self.conv(x))


class _UnetBlockWide(nn.Module):
    def __init__(self, up_in_c: int, x_in_c: int, n_out: int) -> None:
        super().__init__()
        self.shuf = _PixelShuffleICNR(up_in_c, n_out, scale=2, bias=False)
        self.bn = nn.BatchNorm2d(x_in_c)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(n_out + x_in_c, n_out, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_out),
        )

    def forward(self, up_in: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        up_out = self.shuf(up_in)
        cat_x = torch.cat([up_out, self.bn(s)], dim=1)
        return self.conv(cat_x)


class _SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        return self.norm(tgt + tgt2)


class _CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, pos: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory)[0]
        return self.norm(tgt + tgt2)


class _FFNLayer(nn.Module):
    def __init__(self, d_model: int = 256, dim_feedforward: int = 2048) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        return self.norm(tgt + tgt2)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if idx < len(self.layers) - 1 else layer(x)
        return x


class _PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class _MultiScaleColorDecoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        num_scales: int = 3,
    ) -> None:
        super().__init__()
        self.pe_layer = _PositionEmbeddingSine(hidden_dim // 2)
        self.num_layers = dec_layers
        self.num_feature_levels = num_scales
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(_SelfAttentionLayer(hidden_dim, nheads))
            self.transformer_cross_attention_layers.append(_CrossAttentionLayer(hidden_dim, nheads))
            self.transformer_ffn_layers.append(_FFNLayer(hidden_dim, dim_feedforward))

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(num_scales, hidden_dim)
        self.input_proj = nn.ModuleList([nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_channels])
        self.color_embed = _MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    def forward(self, x: list[torch.Tensor], img_features: torch.Tensor) -> torch.Tensor:
        src = []
        pos = []
        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i]).flatten(2).permute(2, 0, 1))
            src.append((self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]).permute(2, 0, 1))

        bs = x[0].shape[0]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            output = self.transformer_cross_attention_layers[i](output, src[level_index], pos[level_index], query_embed)
            output = self.transformer_self_attention_layers[i](output, query_embed)
            output = self.transformer_ffn_layers[i](output)

        decoder_output = self.decoder_norm(output).transpose(0, 1)
        color_embed = self.color_embed(decoder_output)
        return torch.einsum("bqc,bchw->bqhw", color_embed, img_features)


class _CustomDecoder(nn.Module):
    def __init__(self, dims: list[int], num_queries: int = 100, num_scales: int = 3, dec_layers: int = 9) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _UnetBlockWide(dims[3], dims[2], 512),
                _UnetBlockWide(512, dims[1], 512),
                _UnetBlockWide(512, dims[0], 256),
            ]
        )
        self.last_shuf = _LastPixelShuffleICNR(256, 256, scale=4)
        self.color_decoder = _MultiScaleColorDecoder(
            in_channels=[512, 512, 256],
            hidden_dim=256,
            num_queries=num_queries,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=dec_layers,
            num_scales=num_scales,
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        out0 = self.layers[0](feats[3], feats[2])
        out1 = self.layers[1](out0, feats[1])
        out2 = self.layers[2](out1, feats[0])
        img_features = self.last_shuf(out2)
        return self.color_decoder([out0, out1, out2], img_features)


class DDColorOfficial(nn.Module):
    """Official piddnad/DDColor inference module mapping normalized Lab-gray RGB to 2-channel ab."""

    def __init__(
        self,
        encoder_name: str = "convnext-l",
        num_queries: int = 100,
        num_scales: int = 3,
        dec_layers: int = 9,
        num_output_channels: int = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder = _Encoder(encoder_name=encoder_name)
        self.decoder = _CustomDecoder(
            dims=self.encoder.dims,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )
        self.refine_net = nn.Sequential(
            nn.Sequential(
                spectral_norm(nn.Conv2d(num_queries + 3, num_output_channels, kernel_size=1)),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        dec_out = self.decoder(feats)
        return self.refine_net(torch.cat([dec_out, x], dim=1))


def infer_config_from_checkpoint(checkpoint_path: Path, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Infer ConvNeXt variant and decoder parameters from sidecar config.json or state_dict shapes."""
    for candidate in (checkpoint_path.with_suffix(".config.json"), checkpoint_path.parent / "config.json"):
        if candidate.is_file():
            try:
                cfg = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(cfg, dict) and cfg.get("encoder_name"):
                    return cfg
            except Exception:
                pass

    stem_w = state_dict.get("encoder.arch.downsample_layers.0.0.weight")
    c0 = int(stem_w.shape[0]) if isinstance(stem_w, torch.Tensor) else 192
    stage2_blocks = {
        int(k.split(".")[3])
        for k in state_dict
        if k.startswith("encoder.arch.stages.2.") and len(k.split(".")) > 4 and k.split(".")[3].isdigit()
    }
    if c0 == 192:
        enc = "convnext-l"
    elif c0 == 128:
        enc = "convnext-b"
    elif len(stage2_blocks) >= 27:
        enc = "convnext-s"
    else:
        enc = "convnext-t"

    q_w = state_dict.get("decoder.color_decoder.query_feat.weight")
    num_queries = int(q_w.shape[0]) if isinstance(q_w, torch.Tensor) else 100
    dec_idxs = {
        int(k.split(".")[3])
        for k in state_dict
        if k.startswith("decoder.color_decoder.transformer_self_attention_layers.") and k.split(".")[3].isdigit()
    }
    dec_layers = (max(dec_idxs) + 1) if dec_idxs else 9
    return {
        "encoder_name": enc,
        "num_queries": num_queries,
        "num_scales": 3,
        "dec_layers": dec_layers,
        "num_output_channels": 2,
    }


def build_official_ddcolor(checkpoint_path: Path, state_dict: dict[str, Any]) -> DDColorOfficial:
    """Construct `DDColorOfficial` and strictly load state_dict weights."""
    cfg = infer_config_from_checkpoint(checkpoint_path, state_dict)
    model = DDColorOfficial(
        encoder_name=str(cfg.get("encoder_name", "convnext-l")),
        num_queries=int(cfg.get("num_queries", 100)),
        num_scales=int(cfg.get("num_scales", 3)),
        dec_layers=int(cfg.get("dec_layers", 9)),
        num_output_channels=int(cfg.get("num_output_channels", 2)),
    )
    model.load_state_dict(state_dict, strict=True)
    return model
