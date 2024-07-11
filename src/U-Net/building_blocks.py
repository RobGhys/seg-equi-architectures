"""
Simple implementation from scratch of a U-Net

This code is strongly inspired from "https://github.com/milesial/Pytorch-UNet/tree/master"
"""

""" Parts of the U-Net model """

from typing import Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from BesselConv.BesselConv2d import BesselConv2d
from BesselConv.GaussianBlur2d import GaussianBlur2d
from BesselConv.AttentiveNorm2d import AttentiveNorm2d
from e2cnn import gspaces
from e2cnn import nn as e2_nn


class DoubleConv_vanilla(nn.Module):
    """(convolution => [BN] => acti) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, acti=nn.LeakyReLU,
                 acti_kwargs={'negative_slope': 0.1}, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(mid_channels),
            acti(**acti_kwargs),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            acti(**acti_kwargs)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_bcnn(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 reflex_inv=False, scale_inv=False, cutoff='strong', kernel_size=5,
                 acti=nn.Tanh, acti_kwargs={}, bn=False, TensorCorePad=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if not bn:
            self.double_conv = nn.Sequential(
                BesselConv2d(C_in=in_channels, C_out=mid_channels, k=kernel_size, padding='same', bias=False,
                             reflex_inv=reflex_inv, scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad),
                #nn.BatchNorm2d(mid_channels),
                acti(**acti_kwargs),
                BesselConv2d(C_in=mid_channels, C_out=out_channels, k=kernel_size, padding='same', bias=False,
                             reflex_inv=reflex_inv, scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad),
                #nn.BatchNorm2d(out_channels),
                acti(**acti_kwargs)
            )
        else:
            self.double_conv = nn.Sequential(
                BesselConv2d(C_in=in_channels, C_out=mid_channels, k=kernel_size, padding='same', bias=False,
                             reflex_inv=reflex_inv, scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad),
                #nn.BatchNorm2d(mid_channels),
                AttentiveNorm2d(mid_channels),
                acti(**acti_kwargs),
                BesselConv2d(C_in=mid_channels, C_out=out_channels, k=kernel_size, padding='same', bias=False,
                             reflex_inv=reflex_inv, scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad),
                #nn.BatchNorm2d(out_channels),
                AttentiveNorm2d(out_channels),
                acti(**acti_kwargs)
            )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_e2cnn(e2_nn.EquivariantModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, first=False, acti=e2_nn.ELU, stride=1,
                 acti_kwargs={'alpha': 0.1, 'inplace': True}, gspace=gspaces.Rot2dOnR2(N=4), kernel_size=5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.gspace = gspace
        self.in_channels = in_channels

        layers = []
        if first:
            in_type = e2_nn.FieldType(gspace, in_channels * [gspace.trivial_repr])
        else:
            in_type = e2_nn.FieldType(gspace, in_channels * [gspace.regular_repr])
        out_type = e2_nn.FieldType(gspace, mid_channels * [gspace.regular_repr])
        layers += [e2_nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)]
        layers += [e2_nn.InnerBatchNorm(out_type)]
        layers += [acti(in_type=out_type, **acti_kwargs)]
        in_type = out_type
        out_type = e2_nn.FieldType(gspace, out_channels * [gspace.regular_repr])
        layers += [e2_nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
                                stride=stride)]
        layers += [e2_nn.InnerBatchNorm(out_type)]
        layers += [acti(in_type=out_type, **acti_kwargs)]

        self.double_conv = nn.Sequential(
            *layers
        )
        self.out_type = out_type

    def forward(self, x):
        if not isinstance(x, e2_nn.geometric_tensor.GeometricTensor):
            x = e2_nn.GeometricTensor(
                x, e2_nn.FieldType(self.gspace, self.in_channels * [self.gspace.trivial_repr])
            )
        out = self.double_conv(x)
        return out, self.out_type

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass


class Down_vanilla(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, acti=nn.LeakyReLU,
                 acti_kwargs={'negative_slope': 0.1}, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_vanilla(in_channels, out_channels, acti=acti,
                               acti_kwargs=acti_kwargs, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down_bcnn(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, acti=e2_nn.ELU, acti_kwargs={'alpha': 0.1, 'inplace': True},
                 reflex_inv=False, scale_inv=False, cutoff='strong', kernel_size=5, TensorCorePad=False,
                 bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            GaussianBlur2d(C_in=in_channels, sigma=0.66),
            nn.AvgPool2d(2),
            DoubleConv_bcnn(in_channels, out_channels, acti=acti, acti_kwargs=acti_kwargs, reflex_inv=reflex_inv,
                            scale_inv=scale_inv, cutoff=cutoff, kernel_size=kernel_size, TensorCorePad=TensorCorePad,
                            bn=bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down_e2cnn(e2_nn.EquivariantModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, acti=e2_nn.ELU, acti_kwargs={'alpha': 0.1, 'inplace': True},
                 gspace=gspaces.Rot2dOnR2(N=4), kernel_size=5):
        super().__init__()

        in_type = e2_nn.FieldType(gspace, in_channels * [gspace.regular_repr])

        self.maxpool_conv = nn.Sequential(
            e2_nn.PointwiseAvgPoolAntialiased(in_type, sigma=0.66, stride=2),
            DoubleConv_e2cnn(in_channels, out_channels, first=False, acti=acti, stride=1,
                             acti_kwargs=acti_kwargs, gspace=gspace, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass


class Up_vanilla(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, acti=nn.LeakyReLU,
                 acti_kwargs={'negative_slope': 0.1}, kernel_size=3):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv_vanilla(in_channels, out_channels, in_channels // 2, acti=acti,
                                           acti_kwargs=acti_kwargs, kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_vanilla(in_channels, out_channels, acti=acti,
                                           acti_kwargs=acti_kwargs, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up_bcnn(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, acti=nn.Tanh, acti_kwargs={}, bn=False,
                 reflex_inv=False, scale_inv=False, cutoff='strong', kernel_size=5, TensorCorePad=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_bcnn(in_channels, out_channels, in_channels // 2, reflex_inv=reflex_inv,
                                        scale_inv=scale_inv, cutoff=cutoff, kernel_size=kernel_size, bn=bn,
                                        acti=acti, acti_kwargs=acti_kwargs, TensorCorePad=TensorCorePad)
        else:
            raise NotImplementedError('Only bilinear upsampling is implemented for BCNNs')

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up_e2cnn(e2_nn.EquivariantModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, acti=e2_nn.ELU,
                 acti_kwargs={'alpha': 0.1, 'inplace': True}, gspace=gspaces.Rot2dOnR2(N=4), kernel_size=5):
        super().__init__()

        in_type = e2_nn.FieldType(gspace, (in_channels // 2) * [gspace.regular_repr])

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = e2_nn.R2Upsampling(in_type=in_type, scale_factor=2)
            self.conv = DoubleConv_e2cnn(in_channels, out_channels, in_channels // 2, first=False, acti=acti,
                                         acti_kwargs=acti_kwargs, gspace=gspace, kernel_size=kernel_size)
        else:
            raise NotImplementedError('Only bilinear upsampling is implemented for E2CNNs')

    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x = e2_nn.tensor_directsum([x2, x1])

        return self.conv(x)[0]

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class paramActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0]))
        self.w2 = nn.Parameter(torch.tensor([1.0]))
        self.w3 = nn.Parameter(torch.tensor([1.0]))
        self.w4 = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.w1 * torch.sigmoid(x) + self.w2 * torch.tanh(x) + self.w3 * F.relu(x) + self.w4 * x


class ConvertToTensor(e2_nn.EquivariantModule):
    def __init__(self,
                 in_type: Type[e2_nn.FieldType]):
        super(ConvertToTensor, self).__init__()
        self.out_type = in_type
        if self.out_type.gspace.dimensionality == 2:
            self.gpool = e2_nn.GroupPooling(in_type)
        else:
            self.gpool = e2_nn.NormPool(in_type)

    def forward(self, x):
        x = self.gpool(x)
        return x.tensor

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass
