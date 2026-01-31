## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from models.axial_rope import make_axial_pos
from einops import rearrange
# from models.VIT_INR.style import StyleLayer


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            # nn.ReLU(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        # b * pixels * features +coord -> b * pixels *rgb

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        # print(x.shape)
        if(self.with_attn):
            x = self.attn(x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class idm(nn.Module):
    def __init__(self, dim,feat_unfold=False, local_ensemble=False, cell_decode=False):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        if self.cell_decode:
            self.imnet = nn.Sequential(nn.Linear(dim + 2 + 2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, dim))
        else:
            self.imnet = nn.Sequential(nn.Linear(dim + 2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, dim))

    def forward(self, x, shape):
        coord = make_coord(shape).repeat(x.shape[0], 1, 1).to(x.device)  # .to(dist.get_rank())
        cell = torch.ones_like(coord).to(x.device)
        cell[:, 0] *= 2 / shape[-2]
        cell[:, 1] *= 2 / shape[-1]
        return self.query_rgb(x, coord, cell)

    def query_rgb(self, x_feat, coord, cell=None):

        feat = x_feat   # .cuda()
        # print(feat.device)
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            # print(feat.device)

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]).to(x_feat.device)

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                # print(rel_coord)
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret


def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


class Linear(nn.Linear):
    def forward(self, x):
        # flops.op(flops.op_linear, x.shape, self.weight.shape)
        return super().forward(x)


class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x, t):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, t):
        x = x + self.attn(self.norm1(x, t))
        x = x + self.ffn(self.norm2(x, t))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=5, stride=1, padding=5//2, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        pad_w = (2 - (x.size(-1) % 2)) % 2  # 计算宽度需要填充的像素数
        pad_h = (2 - (x.size(-2) % 2)) % 2  # 计算高度需要填充的像素数

        # 使用 F.pad 进行填充（这里用反射填充，也可以选择零填充或其他方式）
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Resmodule -----------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual  # 添加残差连接
        out = self.relu(out)
        return out


##########################################################################
##---------- Restormer -----------------------
class Transformer_INR(nn.Module):
    def __init__(self, config):
        inp_channels = config.vitinr.inp_channels
        out_channels = config.vitinr.out_channels
        inner_channel = config.vitinr.inner_channel
        embed_dim = config.vitinr.embed_dim
        num_blocks = config.vitinr.num_blocks
        heads = config.vitinr.heads
        ffn_expansion_factor = config.vitinr.ffn_expansion_factor
        bias = config.vitinr.bias
        LayerNorm_type = config.vitinr.LayerNorm_type     ## Other option 'BiasFree'
        global_residual = config.vitinr.global_residual
        noise_level_channel = config.vitinr.noise_level_channel
        norm_groups = config.vitinr.norm_groups
        dropout = config.vitinr.dropout
        pre_channel = config.vitinr.pre_channel

        super(Transformer_INR, self).__init__()
        self.global_residual = global_residual
        # print('GR: ', self.global_residual)

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        # self.gen_feature = Extrator(in_channels=out_channels, depth=4, feature_dim=64)

        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=embed_dim)

        self.resba0 = ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        #
        self.encoder_level1 = TransformerBlock(dim=pre_channel, num_heads=heads[0],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba1 = ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.down1_2 = Downsample(pre_channel)  ## From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock(dim=int(pre_channel * 2 ** 1), num_heads=heads[1],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba2 = ResnetBlocWithAttn(int(pre_channel * 2 ** 1), int(pre_channel * 2 ** 1),
                                         noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.down2_3 = Downsample(int(pre_channel * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock(dim=int(pre_channel * 2 ** 2), num_heads=heads[2],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba3 = ResnetBlocWithAttn(int(pre_channel * 2 ** 2), int(pre_channel * 2 ** 2),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.down3_4 = Downsample(int(pre_channel * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = TransformerBlock(dim=int(pre_channel * 2 ** 3), num_heads=heads[3],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba4 = ResnetBlocWithAttn(int(pre_channel * 2 ** 3), int(pre_channel * 2 ** 3),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.down4_5 = Downsample(int(pre_channel * 2 ** 3))  ## From Level 3 to Level 4
        self.encoder_level5 = TransformerBlock(dim=int(pre_channel * 2 ** 4), num_heads=heads[3],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba5 = ResnetBlocWithAttn(int(pre_channel * 2 ** 4), int(pre_channel * 2 ** 4),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(int(pre_channel * 2 ** 4), int(pre_channel * 2 ** 4),
                               noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(int(pre_channel * 2 ** 4), int(pre_channel * 2 ** 4),
                               noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        self.resba6 = ResnetBlocWithAttn(int(pre_channel * 2 ** 4) * 2, int(pre_channel * 2 ** 4),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        self.up5_4 = idm(int(pre_channel * 2 ** 4))

        self.resba7 = ResnetBlocWithAttn(int(pre_channel * 2 ** 4) + 256, int(pre_channel * 2 ** 3),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        self.up4_3 = idm(int(pre_channel * 2 ** 3))

        self.resba8 = ResnetBlocWithAttn(int(pre_channel * 2 ** 3) + 128, int(pre_channel * 2 ** 2),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        self.up3_2 = idm(int(pre_channel * 2 ** 2))

        self.resba9 = ResnetBlocWithAttn(int(pre_channel * 2 ** 2) + 64, int(pre_channel * 2 ** 1),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        self.up2_1 = idm(int(pre_channel * 2 ** 1))

        self.final_conv = Block(int(pre_channel * 2 ** 1), out_channels, groups=norm_groups)

    def check_image_size(self, x, mode_base=16):
        _, _, h, w = x.size()
        mod_pad_h = (mode_base - h % 16) % mode_base
        mod_pad_w = (mode_base - w % 16) % mode_base
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, img, time):
        H, W = img.shape[2:]
        # timestep embedding
        t = self.noise_level_mlp(time)
        # print(t.shape)
        # print(img.shape)

        # sv_feature = self.gen_feature(sv_img)
        # inp_img = torch.cat([sv_feature, fv_img], dim=1)
        inp_img = self.check_image_size(img)

        patch_embed = self.patch_embed(inp_img)
        feature = self.resba0(patch_embed, t)

        feats = []
        out_enc_level1 = self.encoder_level1(feature, t)
        out_enc_level1 = self.resba1(out_enc_level1, t)
        # print('out_enc_level1.shape=', out_enc_level1.shape)
        feats.append(out_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, t)
        out_enc_level2 = self.resba2(out_enc_level2, t)
        feats.append(out_enc_level2)
        # print('out_enc_level2.shape=', out_enc_level2.shape)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, t)
        out_enc_level3 = self.resba3(out_enc_level3, t)
        feats.append(out_enc_level3)
        # print('out_enc_level3.shape=', out_enc_level3.shape)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.encoder_level4(inp_enc_level4, t)
        out_enc_level4 = self.resba4(out_enc_level4, t)
        feats.append(out_enc_level4)
        # print(out_enc_level4.shape)
        inp_enc_level5 = self.down4_5(out_enc_level4)
        out_enc_level5 = self.encoder_level5(inp_enc_level5, t)
        out_enc_level5 = self.resba5(out_enc_level5, t)
        feats.append(out_enc_level5)
        # print(out_enc_level5.shape)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                mid = layer(out_enc_level5, t)
            else:
                mid = layer(out_enc_level5)

        # print('mid.shape=', mid.shape)
        # print('torch.cat([feats.pop(), mid], dim=1).shape=', torch.cat([feats.pop(), mid], dim=1).shape)
        inp_dec_level5 = self.resba6(torch.cat([feats.pop(), mid], dim=1), t)
        # print('inp_dec_level5.shape=', inp_dec_level5.shape)
        out_dec_level5 = self.up5_4(inp_dec_level5, feats[-1].shape[2:])
        # print('out_dec_level4.shape=', out_dec_level4.shape)
        out_dec_level5 = rearrange(out_dec_level5, 'b (h w) c -> b c h w', h=feats[-1].shape[-2])
        # print('out_dec_level5.shape=', out_dec_level5.shape)
        inp_dec_level4 = self.resba7(torch.cat([feats.pop(), out_dec_level5], dim=1), t)
        # print('inp_dec_level4.shape=', inp_dec_level4.shape)
        out_dec_level4 = self.up4_3(inp_dec_level4, feats[-1].shape[2:])
        # print('out_dec_level3.shape=', out_dec_level3.shape)
        out_dec_level4 = rearrange(out_dec_level4, 'b (h w) c -> b c h w', h=feats[-1].shape[-2])
        # print('out_dec_level4.shape=', out_dec_level4.shape)
        inp_dec_level3 = self.resba8(torch.cat([feats.pop(), out_dec_level4], dim=1), t)
        # print('inp_dec_level3.shape=', inp_dec_level3.shape)
        out_dec_level3 = self.up3_2(inp_dec_level3, feats[-1].shape[2:])
        # print('out_dec_level3.shape=', out_dec_level3.shape)
        out_dec_level3 = rearrange(out_dec_level3, 'b (h w) c -> b c h w', h=feats[-1].shape[-2])
        # print('out_dec_level3.shape=', out_dec_level3.shape)
        inp_dec_level2 = self.resba9(torch.cat([feats.pop(), out_dec_level3], dim=1), t)
        # print('inp_dec_level3.shape=', inp_dec_level3.shape)
        # print('feats[-1].shape[2:]=', feats[-1].shape[2:])
        out_dec_level2 = self.up2_1(inp_dec_level2, feats[-1].shape[2:])
        # print('out_dec_level3.shape=', out_dec_level3.shape)
        out_dec_level2 = rearrange(out_dec_level2, 'b (h w) c -> b c h w', h=feats[-1].shape[-2])
        # print('out_dec_level2.shape=', out_dec_level2.shape)

        out = self.final_conv(out_dec_level2)

        return out[:, :, :H, :W]









##########################################################################
##---------- Restormer -----------------------
class Restormer_Backbone(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 inner_channel=32,
                 embed_dim=32,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 global_residual=False,
                 noise_level_channel=32,
                 norm_groups=32,
                 dropout=0.0,
                 pre_channel=32,
                 ):

        super(Restormer_Backbone, self).__init__()
        self.global_residual = global_residual
        # print('GR: ', self.global_residual)

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        # self.gen_feature = Extrator(in_channels=out_channels, depth=4, feature_dim=64)

        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=embed_dim)

        self.resba0 = ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        #
        self.encoder_level1 = TransformerBlock(dim=pre_channel, num_heads=heads[0],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba1 = ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.down1_2 = Downsample(pre_channel)  ## From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock(dim=int(pre_channel * 2 ** 1), num_heads=heads[1],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba2 = ResnetBlocWithAttn(int(pre_channel * 2 ** 1), int(pre_channel * 2 ** 1),
                                         noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.down2_3 = Downsample(int(pre_channel * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock(dim=int(pre_channel * 2 ** 2), num_heads=heads[2],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba3 = ResnetBlocWithAttn(int(pre_channel * 2 ** 2), int(pre_channel * 2 ** 2),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.down3_4 = Downsample(int(pre_channel * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = TransformerBlock(dim=int(pre_channel * 2 ** 3), num_heads=heads[3],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba4 = ResnetBlocWithAttn(int(pre_channel * 2 ** 3), int(pre_channel * 2 ** 3),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.down4_5 = Downsample(int(pre_channel * 2 ** 3))  ## From Level 3 to Level 4
        self.encoder_level5 = TransformerBlock(dim=int(pre_channel * 2 ** 4), num_heads=heads[3],
                                               ffn_expansion_factor=ffn_expansion_factor,
                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.resba5 = ResnetBlocWithAttn(int(pre_channel * 2 ** 4), int(pre_channel * 2 ** 4),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(int(pre_channel * 2 ** 4), int(pre_channel * 2 ** 4),
                               noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(int(pre_channel * 2 ** 4), int(pre_channel * 2 ** 4),
                               noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        self.resba6 = ResnetBlocWithAttn(int(pre_channel * 2 ** 4) * 2, int(pre_channel * 2 ** 4),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        self.up5_4 = idm(int(pre_channel * 2 ** 4))

        self.resba7 = ResnetBlocWithAttn(int(pre_channel * 2 ** 4) + 256, int(pre_channel * 2 ** 3),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        self.up4_3 = idm(int(pre_channel * 2 ** 3))

        self.resba8 = ResnetBlocWithAttn(int(pre_channel * 2 ** 3) + 128, int(pre_channel * 2 ** 2),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        self.up3_2 = idm(int(pre_channel * 2 ** 2))

        self.resba9 = ResnetBlocWithAttn(int(pre_channel * 2 ** 2) + 64, int(pre_channel * 2 ** 1),
                                         noise_level_emb_dim=noise_level_channel,
                                         norm_groups=norm_groups,
                                         dropout=dropout, with_attn=False)
        self.up2_1 = idm(int(pre_channel * 2 ** 1))

        self.final_conv = Block(int(pre_channel * 2 ** 1), out_channels, groups=norm_groups)

    def check_image_size(self, x, mode_base=8):
        _, _, h, w = x.size()
        mod_pad_h = (mode_base - h % 8) % mode_base
        mod_pad_w = (mode_base - w % 8) % mode_base
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, img, time):
        H, W = img.shape[2:]
        # timestep embedding
        t = self.noise_level_mlp(time)
        # print(t.shape)
        # print(img.shape)

        # sv_feature = self.gen_feature(sv_img)
        # inp_img = torch.cat([sv_feature, fv_img], dim=1)
        inp_img = self.check_image_size(img)

        patch_embed = self.patch_embed(inp_img)
        feature = self.resba0(patch_embed, t)

        feats = []
        out_enc_level1 = self.encoder_level1(feature, t)
        out_enc_level1 = self.resba1(out_enc_level1, t)
        # print('out_enc_level1.shape=', out_enc_level1.shape)
        feats.append(out_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, t)
        out_enc_level2 = self.resba2(out_enc_level2, t)
        feats.append(out_enc_level2)
        # print('out_enc_level2.shape=', out_enc_level2.shape)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, t)
        out_enc_level3 = self.resba3(out_enc_level3, t)
        feats.append(out_enc_level3)
        # print('out_enc_level3.shape=', out_enc_level3.shape)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.encoder_level4(inp_enc_level4, t)
        out_enc_level4 = self.resba4(out_enc_level4, t)
        feats.append(out_enc_level4)
        # print(out_enc_level4.shape)
        inp_enc_level5 = self.down4_5(out_enc_level4)
        out_enc_level5 = self.encoder_level5(inp_enc_level5, t)
        out_enc_level5 = self.resba5(out_enc_level5, t)
        feats.append(out_enc_level5)
        # print(out_enc_level5.shape)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                mid = layer(out_enc_level5, t)
            else:
                mid = layer(out_enc_level5)

        # print('mid.shape=', mid.shape)
        # print('torch.cat([feats.pop(), mid], dim=1).shape=', torch.cat([feats.pop(), mid], dim=1).shape)
        inp_dec_level5 = self.resba6(torch.cat([feats.pop(), mid], dim=1), t)
        # print('inp_dec_level5.shape=', inp_dec_level5.shape)
        out_dec_level5 = self.up5_4(inp_dec_level5, feats[-1].shape[2:])
        # print('out_dec_level4.shape=', out_dec_level4.shape)
        out_dec_level5 = rearrange(out_dec_level5, 'b (h w) c -> b c h w', h=feats[-1].shape[-2])
        # print('out_dec_level5.shape=', out_dec_level5.shape)
        inp_dec_level4 = self.resba7(torch.cat([feats.pop(), out_dec_level5], dim=1), t)
        # print('inp_dec_level4.shape=', inp_dec_level4.shape)
        out_dec_level4 = self.up4_3(inp_dec_level4, feats[-1].shape[2:])
        # print('out_dec_level3.shape=', out_dec_level3.shape)
        out_dec_level4 = rearrange(out_dec_level4, 'b (h w) c -> b c h w', h=feats[-1].shape[-2])
        # print('out_dec_level4.shape=', out_dec_level4.shape)
        inp_dec_level3 = self.resba8(torch.cat([feats.pop(), out_dec_level4], dim=1), t)
        # print('inp_dec_level3.shape=', inp_dec_level3.shape)
        out_dec_level3 = self.up3_2(inp_dec_level3, feats[-1].shape[2:])
        # print('out_dec_level3.shape=', out_dec_level3.shape)
        out_dec_level3 = rearrange(out_dec_level3, 'b (h w) c -> b c h w', h=feats[-1].shape[-2])
        # print('out_dec_level3.shape=', out_dec_level3.shape)
        inp_dec_level2 = self.resba9(torch.cat([feats.pop(), out_dec_level3], dim=1), t)
        # print('inp_dec_level3.shape=', inp_dec_level3.shape)
        # print('feats[-1].shape[2:]=', feats[-1].shape[2:])
        out_dec_level2 = self.up2_1(inp_dec_level2, feats[-1].shape[2:])
        # print('out_dec_level3.shape=', out_dec_level3.shape)
        out_dec_level2 = rearrange(out_dec_level2, 'b (h w) c -> b c h w', h=feats[-1].shape[-2])
        # print('out_dec_level2.shape=', out_dec_level2.shape)

        out = self.final_conv(out_dec_level2)

        return out[:, :, :H, :W]












if __name__ == "__main__":
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    model = Restormer_Backbone()
    params = get_parameter_number(model)
    print(params)
    x1 = torch.randn((1, 1, 512, 512))
    # x2 = torch.randn((1, 1, 720, 729))
    b = x1.size(0)
    noise_level = torch.FloatTensor([0.000001]).repeat(b, 1)
    print(noise_level.shape)
    x = model(x1, noise_level)
    print(x.shape)



