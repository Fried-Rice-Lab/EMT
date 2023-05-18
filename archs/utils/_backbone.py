import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as f

from archs.utils._conv import Conv2d1x1, Conv2d3x3

__all__ = [ 'TransformerGroup',  'Upsampler']



class TransformerGroup(nn.Module):
    r"""

    Args:
        sa_list:
        mlp_list:
        conv_list:

    """

    def __init__(self, sa_list: list, mlp_list: list, conv_list: list = None) -> None:
        super(TransformerGroup, self).__init__()

        assert len(sa_list) == len(mlp_list)

        self.sa_list = nn.ModuleList(sa_list)
        self.mlp_list = nn.ModuleList(mlp_list)
        self.conv = nn.Sequential(*conv_list if conv_list is not None else [nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list):
            x = x + sa(x)
            x = x + mlp(x)
        return self.conv(x)


class _EncoderTail(nn.Module):
    def __init__(self, planes: int) -> None:
        super(_EncoderTail, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels=planes, out_channels=2 * planes,
                                            kernel_size=(2, 2), stride=(2, 2), bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _DecoderHead(nn.Module):
    def __init__(self, planes: int) -> None:
        super(_DecoderHead, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels=planes, out_channels=2 * planes,
                                            kernel_size=(1, 1), bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsampler(nn.Sequential):
    r"""Tail of the image restoration network.

    Args:
        upscale (int):
        in_channels (int):
        out_channels (int):
        upsample_mode (str):

    """

    def __init__(self, upscale: int, in_channels: int,
                 out_channels: int, upsample_mode: str = 'csr') -> None:

        layer_list = list()
        if upsample_mode == 'csr':  # classical
            if (upscale & (upscale - 1)) == 0:  # 2^n?
                for _ in range(int(math.log(upscale, 2))):
                    layer_list.append(Conv2d3x3(in_channels, 4 * in_channels))
                    layer_list.append(nn.PixelShuffle(2))
            elif upscale == 3:
                layer_list.append(Conv2d3x3(in_channels, 9 * in_channels))
                layer_list.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'Upscale {upscale} is not supported.')
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        elif upsample_mode == 'lsr':  # lightweight
            layer_list.append(Conv2d3x3(in_channels, out_channels * (upscale ** 2)))
            layer_list.append(nn.PixelShuffle(upscale))
        elif upsample_mode == 'denoising' or upsample_mode == 'deblurring' or upsample_mode == 'deraining':
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        else:
            raise ValueError(f'Upscale mode {upscale} is not supported.')

        super(Upsampler, self).__init__(*layer_list)


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    module = DistBackbone(planes=20, dist_num=8, dist_rate=0.123,
                          rema_layer=Conv2d1x1, rema_layer_kwargs={'bias': True},
                          dist_layer=Conv2d1x1, dist_layer_kwargs={'bias': False},
                          act_layer=nn.LeakyReLU, act_layer_kwargs={'negative_slope': 0.05, 'inplace': True})
    print(count_parameters(module))

    data = torch.randn(1, 20, 10, 10)
    print(module(data).size())


    class Conv(nn.Conv2d):
        def __init__(self, in_channels: int, kernel_size: tuple, padding: tuple = (0, 0),
                     dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                     **kwargs) -> None:
            super(Conv, self).__init__(in_channels=in_channels, out_channels=in_channels,
                                       kernel_size=kernel_size, stride=(1, 1), padding=padding,
                                       dilation=dilation, groups=groups, bias=bias, **kwargs)


    u = UBackbone(planes=4,
                  encoder=Conv, encoder_kwargs={'kernel_size': 1, 'padding': 0}, encoder_nums=[2, 2, 2],
                  middler=Conv, middler_kwargs={'kernel_size': 3, 'padding': 1}, middler_num=2,
                  decoder=Conv, decoder_kwargs={'kernel_size': 5, 'padding': 2}, decoder_nums=[2, 2, 2])
    print(count_parameters(u), u)

    data = torch.randn(1, 4, 49, 49)
    print(u(data).size())
