import torch
import torch.nn as nn
import numpy as np
import sys

import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(project_dir, 'decomposition'))
# sys.path.insert(1, "/vol/bitbucket/apg416/project/decomposition")
from hparams import hparams
from model import Generator_3 as Generator
from utils import quantize_f0_torch


class Decomposition(nn.Module):
    def __init__(self, ckpt_path, freeze):
        super(Decomposition, self).__init__()
        g_checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.G = Generator(hparams).eval()
        self.G.load_state_dict(g_checkpoint["model"])
        # Freeze the model
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x_real_org, emb_org, f0_org):
        x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
        f0_org = quantize_f0_torch(x_f0[:, :, -1])[0]
        x_f0_org = torch.cat((x_f0[:, :, :-1], f0_org), dim=-1)

        x_1 = x_f0_org.transpose(2, 1)
        codes_x, codes_f0 = self.G.encoder_1(x_1)

        x_2 = x_real_org.transpose(2, 1)
        codes_2 = self.G.encoder_2(x_2, None)

        C = x_1.size(-1)

        return C, emb_org, codes_x, codes_f0, codes_2

    def decode(self, C, emb_org, codes_x, codes_f0, codes_2):

        x_1_size_last = C
        c_trg = emb_org

        code_exp_1 = codes_x.repeat_interleave(hparams.freq, dim=1)
        code_exp_3 = codes_f0.repeat_interleave(hparams.freq_3, dim=1)
        code_exp_2 = codes_2.repeat_interleave(hparams.freq_2, dim=1)

        encoder_outputs = torch.cat(
            (
                code_exp_1,
                code_exp_2,
                code_exp_3,
                c_trg.unsqueeze(1).expand(-1, x_1_size_last, -1),
            ),
            dim=-1,
        )

        mel_outputs = self.G.decoder(encoder_outputs)

        return mel_outputs
