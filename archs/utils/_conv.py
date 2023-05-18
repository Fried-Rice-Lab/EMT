import torch
from torch import nn as nn
from torch.nn import functional as f


__all__ = ['Conv2d1x1', 'Conv2d3x3', 'MeanShift',
           'ShiftConv2d1x1'
           ]


class Conv2d1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class Conv2d3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d3x3, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class MeanShift(nn.Conv2d):
    r"""

    Args:
        rgb_range (int):
        sign (int):
        data_type (str):

    """

    def __init__(self, rgb_range: int, sign: int = -1, data_type: str = 'DIV2K') -> None:
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))

        rgb_std = (1.0, 1.0, 1.0)
        if data_type == 'DIV2K':
            # RGB mean for DIV2K 1-800
            rgb_mean = (0.4488, 0.4371, 0.4040)
        elif data_type == 'DF2K':
            # RGB mean for DF2K 1-3450
            rgb_mean = (0.4690, 0.4490, 0.4036)
        else:
            raise NotImplementedError(f'Unknown data type for MeanShift: {data_type}.')

        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ShiftConv2d1x1(nn.Conv2d):
    r"""

    Args:
        in_channels (int):
        out_channels (int):
        stride (tuple):
        dilation (tuple):
        bias (bool):
        shift_mode (str):
        val (float):

    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), bias: bool = True, shift_mode: str = '+', val: float = 1.,
                 **kwargs) -> None:
        super(ShiftConv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                             dilation=dilation, groups=1, bias=bias, **kwargs)

        assert in_channels % 5 == 0, f'{in_channels} % 5 != 0.'

        channel_per_group = in_channels // 5
        self.mask = nn.Parameter(torch.zeros((in_channels, 1, 3, 3)), requires_grad=False)
        if shift_mode == '+':
            self.mask[0 * channel_per_group:1 * channel_per_group, 0, 1, 2] = val
            self.mask[1 * channel_per_group:2 * channel_per_group, 0, 1, 0] = val
            self.mask[2 * channel_per_group:3 * channel_per_group, 0, 2, 1] = val
            self.mask[3 * channel_per_group:4 * channel_per_group, 0, 0, 1] = val
            self.mask[4 * channel_per_group:, 0, 1, 1] = val
        elif shift_mode == 'x':
            self.mask[0 * channel_per_group:1 * channel_per_group, 0, 0, 0] = val
            self.mask[1 * channel_per_group:2 * channel_per_group, 0, 0, 2] = val
            self.mask[2 * channel_per_group:3 * channel_per_group, 0, 2, 0] = val
            self.mask[3 * channel_per_group:4 * channel_per_group, 0, 2, 2] = val
            self.mask[4 * channel_per_group:, 0, 1, 1] = val
        else:
            raise NotImplementedError(f'Unknown shift mode for ShiftConv2d1x1: {shift_mode}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.conv2d(input=x, weight=self.mask, bias=None,
                     stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=self.in_channels)
        x = f.conv2d(input=x, weight=self.weight, bias=self.bias,
                     stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        return x





if __name__ == '__main__':
    a = torch.arange(1, 21).reshape(2, 10, 1, 1).float()
    a = f.pad(a, (1, 1, 1, 1))
