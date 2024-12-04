import numpy as np
import torch as th 

from torch.nn import (
    Module,
    ModuleDict,
    Upsample
)
from modules import (
    Conv,
    Sppf,
    C2f,
    Detect
)

class BackBone(Module):

    def __init__(
            self, 
            in_channels: int,
            out_channels: list[int, int, int]
    ) -> None:
        
        super().__init__()
        self._model = ModuleDict({
            "con0v": Conv(in_channels=in_channels, out_channels=256, stride=1),
            "conv1": Conv(in_channels=256, out_channels=256, stride=1),
            "c2f0": C2f(in_channels=256, out_channels=128, hiden_layers=1),
            "conv2": Conv(in_channels=128, out_channels=128, stride=2),
            "c2f1": C2f(in_channels=128, out_channels=out_channels[0], hiden_layers=1),
            "conv3": Conv(in_channels=out_channels[0], out_channels=out_channels[0], stride=2),
            "c2f2": C2f(in_channels=out_channels[0], out_channels=out_channels[1], hiden_layers=1),
            "conv4": Conv(in_channels=out_channels[1], out_channels=out_channels[1], stride=2),
            "c2f3": C2f(in_channels=out_channels[1], out_channels=out_channels[2], hiden_layers=1),
            "sppf": Sppf(
                in_channels=out_channels[2],
                out_channels=out_channels[2]
            )  
        })
    
    def __call__(self, inputs: th.Tensor) -> list[
        th.Tensor, 
        th.Tensor, 
        th.Tensor
    ]:
        
        x = inputs
        outputs = []
        for name, layer in self._model.items():

            x = layer(x)
            if name in ["c2f1", "c2f2", "sppf"]:
                outputs.append(x)
        
        return outputs


class Head(Module):

    def __init__(
            self, 
            in_channels: list[int, int, int],
            out_channels: list[int, int, int]
    ) -> None:

        super().__init__()
        self._up_sample0 = Upsample(scale_factor=2)
        self._up_sample1 = Upsample(scale_factor=2)
        self._c2f0 = C2f(in_channels=(in_channels[-1] + in_channels[1]), out_channels=out_channels[1])
        self._c2f1 = C2f(in_channels=(64 + in_channels[0]), out_channels=out_channels[0])
        self._conv0 = Conv(
            in_channels=32,
            out_channels=32,
            stride=2
        )
        self._c2f2 = C2f(in_channels=96, out_channels=64)
        self._conv1 = Conv(
            in_channels=64,
            out_channels=64,
            stride=2
        )
        self._c2f3 = C2f(in_channels=(in_channels[-1] + 64), out_channels=out_channels[-1])
    
    def __call__(self, inputs: list[
        th.Tensor,
        th.Tensor,
        th.Tensor
    ]) -> list[th.Tensor, th.Tensor, th.Tensor]:

        up0 = self._up_sample0(inputs[-1])
        cat0 = th.cat([inputs[1], up0], dim=1)
        c2f0 = self._c2f0(cat0)

        up1 = self._up_sample1(c2f0)
        cat1 = th.cat([inputs[0], up1], dim=1)
        c2f1 = self._c2f1(cat1)

        conv0 = self._conv0(c2f1)
        cat2 = th.cat([c2f0, conv0], dim=1)
        c2f2 = self._c2f2(cat2)

        conv1 = self._conv1(c2f2)
        cat3 = th.cat([inputs[-1], conv1], dim=1)
        c2f3 = self._c2f3(cat3)

        return [
            c2f1,
            c2f2,
            c2f3
        ]
        

class Inference(Module):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, inputs: list[th.Tensor, th.Tensor, th.Tensor]) -> list[
        list[th.Tensor, th.Tensor],
        list[th.Tensor, th.Tensor],
        list[th.Tensor, th.Tensor]
    ]:
        
        return [
            Detect(in_channels=inputs[0].size()[1])(inputs[0]),
            Detect(in_channels=inputs[1].size()[1])(inputs[1]),
            Detect(in_channels=inputs[2].size()[1])(inputs[2])
        ]
    


class YOLOv8Net(Module):

    def __init__(
            self,
            hiden_channels: list[int, int, int],
            in_channels: int = 3
    ) -> None:
        
        super().__init__()
        self._backbone = BackBone(
            in_channels=in_channels,
            out_channels=[ch for ch in hiden_channels]
        )
        self._head =  Head(
            in_channels=[ch for ch in hiden_channels],
            out_channels=[32, 64, 128]
        )
        self._inf = Inference()
    
    def __call__(self, inputs: th.Tensor) -> list[
        list[th.Tensor, th.Tensor],
        list[th.Tensor, th.Tensor],
        list[th.Tensor, th.Tensor]
    ]:
        
        back_out = self._backbone(inputs)
        head_out = self._head(back_out)
        return self._inf(head_out)


        

if __name__ == "__main__":

    inputs = th.Tensor(np.random.normal(0, 1.120, (32, 3, 128, 128)))
    yolo_net = YOLOv8Net(hiden_channels=[
        32, 
        64, 
        128
    ])
    yolo_out = yolo_net(inputs)
    
   
