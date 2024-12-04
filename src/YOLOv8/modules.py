import numpy as np
import torch as th

from torch.nn import (
    Conv2d,
    BatchNorm2d,
    SiLU,
    MaxPool2d,
    Module,
    Sequential,
    ModuleList,
    ModuleDict,
    Upsample
)


class Conv(Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            stride: int = 1
    ) -> None:
        
        super().__init__()
        self._model = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            BatchNorm2d(num_features=out_channels),
            SiLU()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._model(inputs)

class Sppf(Module):
    
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads_n: int = 3
    ) -> None:
        
        super().__init__()
        self._conv = ModuleDict({
            "in_conv": Conv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0
            ),
            "out_conv": Conv(
                        in_channels=out_channels * (heads_n + 1),
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    ) 
        })
        self._pool = ModuleList([
            MaxPool2d(kernel_size=3, stride=1, padding=1)
            for _ in range(heads_n)
        ])
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        in_conv = self._conv["in_conv"](inputs)
        x = in_conv
        cat_layers = []
        for layer in self._pool:

            x = layer(x)
            cat_layers.append(x)

        cat = th.cat(cat_layers + [in_conv, ], dim=1)
        return self._conv["out_conv"](cat)


class BottleNeck(Module):

    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 3
    ) -> None:
        
        super().__init__()
        self._in_model = Sequential(
            Conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
        )
        self._out_model = Sequential(
            Conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        in_out = self._in_model(inputs)
        add = th.add(inputs, in_out)
        return self._out_model(add)
    

class Split(Module):

    def __init__(
        self,
        in_channels: int
    ):
        super().__init__()
        self.in_channels = in_channels
        assert self.in_channels % 2 == 0, "in_channels must be devideble on 2!"
        
    def __call__(self, inputs: th.Tensor) -> list[th.Tensor, th.Tensor] | tuple[th.Tensor, th.Tensor]:

        return [
            inputs[:, :self.in_channels // 2, :, :],
            inputs[:, self.in_channels // 2:, :, :]
        ]

class C2f(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hiden_layers: int = 3
    ) -> None:
        
        super().__init__()
        self._in_conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self._split = Split(in_channels=out_channels)
        self._bottle_neck = ModuleList([
            BottleNeck(in_channels=out_channels // 2)
            for _ in range(hiden_layers)
        ])
        self._out_conv = Conv(
            in_channels=(out_channels // 2) * (hiden_layers + 1),
            out_channels=out_channels
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        in_conv = self._in_conv(inputs)
        splits = self._split(in_conv)
        x = splits[1]

        cat_layers = [splits[0]]
        for layer in self._bottle_neck:

            x = layer(x)
            cat_layers.append(x)

        cat = th.cat(cat_layers, dim=1)
        return self._out_conv(cat)


class Detect(Module):

    def __init__(
            self,
            in_channels: int,
            classes_n: int = 3,
            bbox_n: int = 5,
            grids_n: int = 7
    ) -> None:
        
        super().__init__()
        self._bbox_head = Sequential(
            Conv(in_channels=in_channels, out_channels=in_channels, stride=2),
            Conv(in_channels=in_channels, out_channels=in_channels, stride=2),
            Upsample(size=grids_n),
            Conv2d(
                in_channels=in_channels,
                out_channels=bbox_n,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self._cll_head = Sequential(
            Conv(in_channels=in_channels, out_channels=in_channels, stride=2),
            Conv(in_channels=in_channels, out_channels=in_channels, stride=2),
            Upsample(size=grids_n),
            Conv2d(
                in_channels=in_channels,
                out_channels=classes_n,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    
    def __call__(self, inputs: th.Tensor) -> list[
        th.Tensor,
        th.Tensor
    ]:
        return [
            self._bbox_head(inputs),
            self._cll_head(inputs)
        ]
    
if __name__ == "__main__":

    inputs = th.Tensor(np.random.normal(0, 1.120, (32, 48, 32, 32)))
    out = C2f(in_channels=48, out_channels=64)(inputs)
    bbox, cll = Detect(in_channels=64)(out)

    print(bbox.size(), cll.size())

