# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor
import torch


class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        ct_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation, add_crosstalk=True)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers, ct_layer=ct_layer)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search, exc):
        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        # exc = exc.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search, exc=exc,
                                                  src1_key_padding_mask=mask_temp,
                                                  src2_key_padding_mask=mask_search,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search)
        hs = self.decoder(memory_search, memory_temp,
                          tgt_key_padding_mask=mask_search,
                          memory_key_padding_mask=mask_temp,
                          pos_enc=pos_temp, pos_dec=pos_search)
        return hs.unsqueeze(0).transpose(1, 2), exc  # .unsqueeze(0).transpose(1, 2)


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers, ct_layer, num_fb=1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(copy.deepcopy(ct_layer))
            else:
                layers.append(copy.deepcopy(featurefusion_layer))
        self.layers = nn.ModuleList(layers)
        # self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers
        self.num_fb = num_fb

    def forward(self, src1, src2, exc,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        output1 = src1
        output2 = src2

        for idx, layer in enumerate(self.layers):
            if idx >= len(self.layers) - self.num_fb:
                output1, output2 = layer(output1, output2, exc, src1_mask=src1_mask,
                                         src2_mask=src2_mask,
                                         src1_key_padding_mask=src1_key_padding_mask,
                                         src2_key_padding_mask=src2_key_padding_mask,
                                         pos_src1=pos_src1, pos_src2=pos_src2)
            else:
                output1, output2 = layer(output1, output2, None, src1_mask=src1_mask,
                                         src2_mask=src2_mask,
                                         src1_key_padding_mask=src1_key_padding_mask,
                                         src2_key_padding_mask=src2_key_padding_mask,
                                         pos_src1=pos_src1, pos_src2=pos_src2)

        return output1, output2


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # self.mix_norm2 = nn.LayerNorm(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):

        # q = self.mix_q(torch.cat([q, self.mix_norm(exc)], -1))
        # Alternative is to add q
        preq = self.with_pos_embed(tgt, pos_dec)
        tgt2 = self.multihead_attn(query=preq,
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)


class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", add_crosstalk=False):
        super().__init__()
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)
        self.add_crosstalk = add_crosstalk
        if self.add_crosstalk:
            self.circuit_dw = nn.Sequential(
                    nn.Conv2d(2 * d_model, 2 * d_model, kernel_size=3, padding=3//2),  # nn.Linear(d_model * 2, d_model)
                    nn.ReLU(),
                    nn.Conv2d(2 * d_model, d_model, kernel_size=1, padding=1//2))
            # self.excnorm = nn.LayerNorm(d_model)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

        # self.mix_q1 = nn.Linear(d_model, d_model)
        # self.mix_q2 = nn.Linear(d_model, d_model)
        # self.mix_q3 = nn.Linear(d_model * 2, d_model)
        # self.mix_q4 = nn.Linear(d_model, d_model)
        # self.mix_norm1 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2, exc,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):
        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
                               key_padding_mask=src1_key_padding_mask)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        if self.add_crosstalk:
            res_src2 = src2.permute(1, 2, 0).view(-1, src2.shape[2], 32, 32)
            mask = self.circuit_dw(torch.cat([res_src2, exc], 1)).sigmoid()  # Dynamic weighting of exc
            masked_res_src2 = (1 - mask) * res_src2 + (mask * exc)
            src2 = masked_res_src2.flatten(2).permute(2, 0, 1)
            # mask = self.dw(exc).sigmoid().flatten(2).permute(2, 0, 1)
            # src2 = src2 * mask.flatten(2).permute(2, 0, 1)
        q2 = k2 = self.with_pos_embed(src2, pos_src2)
        src22 = self.self_attn2(q2, k2, value=src2, attn_mask=src2_mask,
                               key_padding_mask=src2_key_padding_mask)[0]
        # if self.add_crosstalk:  # exc is not None:
            # from matplotlib import pyplot as plt
            # plt.subplot(141);plt.imshow(mask[0].mean(0).detach().cpu());plt.title("mask");plt.subplot(142);plt.imshow(src2.view(32, 32, 4, 256)[:,:,0].mean(-1).detach().cpu());plt.title("src");plt.subplot(143);plt.imshow(exc[0].mean(0).detach().cpu());plt.title("circuit");plt.subplot(144);plt.imshow(res_src2[0].mean(0).detach().cpu());plt.title("original_src");plt.show()
            # # plt.subplot(151);plt.title("Key");plt.imshow((k2[:, ].reshape(32, 32, -1) ** 2).mean(-1).detach().cpu());plt.subplot(152);plt.title("Raw exc");plt.imshow((exc ** 2).mean(0)[0].detach().cpu())
            # # plt.subplot(153);plt.title("Circuit");plt.imshow(((mask * exc)** 2).mean(0)[0].detach().cpu())
            # # plt.subplot(154);plt.title("proc-src");plt.imshow((src22[:, ].reshape(32, 32, -1) ** 2).mean(-1).detach().cpu())
            # # plt.subplot(155);plt.imshow((src2[:, ].reshape(32, 32, -1) ** 2).mean(-1).detach().cpu());plt.title("skipnorm-src");plt.show()

        # if exc is not None:
        #     # Old version
        #     src2 = src2 + self.dropout21(src22) + exc
        # else:
        #     src2 = src2 + self.dropout21(src22)

        # # New version
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)
        # #

        # if exc is not None:
        #     from matplotlib import pyplot as plt
        #     plt.subplot(151);plt.title("Key");plt.imshow((k2[:, ].reshape(32, 32, -1) ** 2).mean(-1).detach().cpu());plt.subplot(152);plt.title("Raw exc");plt.imshow((exc[:, ].reshape(32, 32, -1) ** 2).mean(-1).detach().cpu());plt.subplot(153);plt.title("Circuit");plt.imshow(((dw * exc)[:, ].reshape(32, 32, -1) ** 2).mean(-1).detach().cpu());plt.subplot(154);plt.title("proc-src");plt.imshow((src22[:, ].reshape(32, 32, -1) ** 2).mean(-1).detach().cpu());plt.subplot(155);plt.imshow((((1 - dw) * src2)[:, ].reshape(32, 32, -1) ** 2).mean(-1).detach().cpu());plt.title("skipnorm-src");plt.show()

        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                   key=self.with_pos_embed(src2, pos_src2),
                                   value=src2, attn_mask=src2_mask,
                                   key_padding_mask=src2_key_padding_mask)[0]

        # Template K/V and Source Q
        mergeq = self.with_pos_embed(src2, pos_src2)
        src22 = self.multihead_attn2(query=mergeq,
                                   key=self.with_pos_embed(src1, pos_src1),
                                   value=src1, attn_mask=src1_mask,
                                   key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)

        return src1, src2

    def forward(self, src1, src2, exc,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, exc, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
    return FeatureFusionNetwork(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        num_featurefusion_layers=settings.featurefusion_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

