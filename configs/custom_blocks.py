import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class StripConv(nn.Module):
    """分解条带卷积：1×k + k×1 两路分解，等效十字形感受野"""
    def __init__(self, c_in, c_out, k=5, stride=1):
        super().__init__()
        c_in = int(c_in)
        c_out = int(c_out)
        k = int(k)
        # 水平条带 1×k
        self.conv_h = Conv(c_in, c_out, (1, k), stride)
        # 垂直条带 k×1
        self.conv_v = Conv(c_in, c_out, (k, 1), stride)

    def forward(self, x):
        return self.conv_h(x) + self.conv_v(x)


class DualBranchBottleneck(nn.Module):
    """双分支瓶颈结构：各向同性3×3 + 各向异性条带卷积，1:1通道比例"""
    def __init__(self, c_in, c_out, shortcut=True, e=0.5, strip_k=5):
        super().__init__()
        c_in = int(c_in)
        c_out = int(c_out)
        c_mid = int(c_out * e)
        self.shortcut = shortcut and c_in == c_out

        # 各向同性分支（标准3×3）
        self.iso_branch = nn.Sequential(
            Conv(c_in, c_mid // 2, 1),
            Conv(c_mid // 2, c_mid // 2, 3),
        )

        # 各向异性分支（条带卷积）
        self.aniso_branch = nn.Sequential(
            Conv(c_in, c_mid // 2, 1),
            StripConv(c_mid // 2, c_mid // 2, k=strip_k),
        )

        self.proj = Conv(c_mid, c_out, 1)

    def forward(self, x):
        identity = x
        x_iso = self.iso_branch(x)
        x_aniso = self.aniso_branch(x)
        x = torch.cat([x_iso, x_aniso], dim=1)
        x = self.proj(x)
        if self.shortcut:
            x = x + identity
        return x


class C2f_DIA(nn.Module):
    """C2f + 双分支DIA瓶颈，接口与标准C2f完全一致"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(int((2 + n) * self.c), c2, 1)
        self.m = nn.ModuleList(
            DualBranchBottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(int(n))
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class StripBottleneck(nn.Module):
    """纯条带单分支瓶颈，用于全替换对照实验"""
    def __init__(self, c_in, c_out, shortcut=True, e=0.5, strip_k=5):
        super().__init__()
        c_in = int(c_in)
        c_out = int(c_out)
        c_mid = int(c_out * e)
        self.shortcut = shortcut and c_in == c_out
        self.conv = nn.Sequential(
            Conv(c_in, c_mid, 1),
            StripConv(c_mid, c_mid, k=strip_k),
        )
        self.proj = Conv(c_mid, c_out, 1)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.proj(x)
        if self.shortcut:
            x = x + identity
        return x


class C2f_Strip(nn.Module):
    """C2f + 纯条带瓶颈，全替换对照组"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(int((2 + n) * self.c), c2, 1)
        self.m = nn.ModuleList(
            StripBottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(int(n))
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
