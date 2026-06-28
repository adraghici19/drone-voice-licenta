import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DSConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride_f: int = 2):
        super().__init__()
        self._pad_t = kernel - 1
        self._pad_f = kernel // 2
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel,
            stride=(1, stride_f), groups=in_ch, bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._pad_f, self._pad_f, self._pad_t, 0))
        x = self.dw(x)
        return F.relu(self.bn(self.pw(x)), inplace=True)


class DSConvTransposeBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=3, stride=(1, 2),
            padding=(1, 1), groups=in_ch, bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


def _cat_skip(dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    f = min(dec.shape[-1], skip.shape[-1])
    return torch.cat([dec[..., :f], skip[..., :f]], dim=1)


class LiteVoiceNet(nn.Module):

    def __init__(
        self,
        num_features: int = 5,
        enc_ch: Optional[List[int]] = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        num_kws_classes: int = 4,
        kws_window: int = 188,
        n_freq_bins: int = 257,
    ):
        super().__init__()
        if enc_ch is None:
            enc_ch = [32, 64, 96, 128]
        assert len(enc_ch) == 4, "enc_ch must have exactly 4 stages"

        self.n_freq_bins = n_freq_bins
        self.kws_window = kws_window

        in_chs = [num_features] + enc_ch[:-1]
        self.encoder = nn.ModuleList(
            [DSConvBlock(in_chs[i], enc_ch[i]) for i in range(4)]
        )

        self.gru = nn.GRU(
            input_size=enc_ch[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.2 if gru_layers > 1 else 0.0,
        )

        self.dec4 = DSConvTransposeBlock(gru_hidden + enc_ch[3], enc_ch[2])
        self.dec3 = DSConvTransposeBlock(enc_ch[2] * 2,          enc_ch[1])
        self.dec2 = DSConvTransposeBlock(enc_ch[1] * 2,          enc_ch[0])
        self.dec1 = DSConvTransposeBlock(enc_ch[0] * 2,          16)
        self.mask_proj = nn.Conv2d(16, 1, kernel_size=1)

        self.vad_head = nn.Linear(gru_hidden, 1)

        self.kws_head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_kws_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skips: List[torch.Tensor] = []
        feat = x
        for block in self.encoder:
            feat = block(feat)
            skips.append(feat)

        bottleneck = feat.mean(dim=-1).permute(0, 2, 1)
        gru_out, h_out = self.gru(bottleneck, hx)

        vad = torch.sigmoid(self.vad_head(gru_out))

        win = min(gru_out.shape[1], self.kws_window)
        kws_feat = gru_out[:, -win:, :].mean(dim=1)
        kws = F.log_softmax(self.kws_head(kws_feat), dim=-1)

        F_bot = skips[3].shape[-1]
        dec = gru_out.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, F_bot)
        dec = torch.cat([dec, skips[3]], dim=1)

        dec = self.dec4(dec)
        dec = _cat_skip(dec, skips[2])
        dec = self.dec3(dec)
        dec = _cat_skip(dec, skips[1])
        dec = self.dec2(dec)
        dec = _cat_skip(dec, skips[0])
        dec = self.dec1(dec)

        if dec.shape[-1] > self.n_freq_bins:
            dec = dec[..., : self.n_freq_bins]
        elif dec.shape[-1] < self.n_freq_bins:
            dec = F.pad(dec, (0, self.n_freq_bins - dec.shape[-1]))

        mask = torch.sigmoid(self.mask_proj(dec))

        return mask, vad, kws, h_out


def build_model(cfg) -> LiteVoiceNet:
    return LiteVoiceNet(
        num_features=cfg.array.num_features,
        enc_ch=cfg.model.enc_ch,
        gru_hidden=cfg.model.gru_hidden,
        gru_layers=cfg.model.gru_layers,
        num_kws_classes=cfg.model.num_kws_classes,
        kws_window=cfg.model.kws_window_frames,
        n_freq_bins=cfg.signal.n_freq_bins,
    )
