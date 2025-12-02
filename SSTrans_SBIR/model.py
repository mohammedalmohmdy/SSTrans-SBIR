# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

class DCTSpectralBranch(nn.Module):
    def __init__(self, in_ch=3, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((8,8))
        )
    def forward(self, x):
        f = self.conv(x)
        B,C,H,W = f.shape
        tokens = f.view(B, C, H*W).permute(0,2,1)
        return tokens

class SpatialBackbone(nn.Module):
    def __init__(self, out_dim=256, backbone='resnet50'):
        super().__init__()
        res = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(res.children())[:-2])
        self.proj = nn.Conv2d(2048, out_dim, kernel_size=1)
    def forward(self,x):
        f = self.features(x)
        f = self.proj(f)
        B,C,H,W = f.shape
        tokens = f.view(B, C, H*W).permute(0,2,1)
        return tokens

class SpectralSpatialFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
    def forward(self, spec_tokens, spat_tokens):
        spec_mean = spec_tokens.mean(dim=1, keepdim=True)
        spat_mean = spat_tokens.mean(dim=1, keepdim=True)
        fused = torch.cat([spec_mean.expand(-1,spat_tokens.size(1),-1), spat_tokens], dim=-1)
        fused = self.fc(fused)
        return fused

class CrossDomainTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, query_tokens, key_value_tokens):
        x = torch.cat([query_tokens, key_value_tokens], dim=1)
        out = self.transformer(x)
        return out[:, :query_tokens.size(1), :]

class SSTransSBIR(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.spec = DCTSpectralBranch(out_dim=64)
        self.spat = SpatialBackbone(out_dim=d_model)
        self.fusion = SpectralSpatialFusion(d_model=d_model)
        self.cdt = CrossDomainTransformer(d_model=d_model, nhead=4, num_layers=4)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 128))
    def forward(self, sketch_img, photo_img):
        spec_s = self.spec(sketch_img)
        spat_s = self.spat(sketch_img)
        fused_s = self.fusion(spec_s, spat_s)

        spec_p = self.spec(photo_img)
        spat_p = self.spat(photo_img)
        fused_p = self.fusion(spec_p, spat_p)

        contextual = self.cdt(fused_s, fused_p)
        emb = contextual.mean(dim=1)
        emb = self.proj(emb)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb
