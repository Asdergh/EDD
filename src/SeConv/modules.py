import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    Flatten,
    Dense,
    BatchNormalization,
    GlobalAveragePooling2D,
    Multiply
)
from tensorflow.keras import (
    Model,
    Sequential
)
from tensorflow import (
    Module,
    multiply
)


class Conv(Module):

    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 2,
        activation: str = "tanh",
        name: str = None
    ) -> None:
        
        super().__init__()
        self.model_ = Sequential([
            Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="same"
            ),
            BatchNormalization(),
            Activation(activation)
        ], name=name)
    
    def __call__(self, inputs):
        return self.model_(inputs)


class Se(Module):

    def __init__(
            self, 
            in_channels: int,
            em_dim: int = 64,
            name: str = None
    ) -> None:

        super().__init__(name)
        self.model_ = Sequential([
            GlobalAveragePooling2D(),
            Dense(units=em_dim, activation="relu"),
            Dense(units=in_channels, activation="sigmoid")
        ], name=name)

    def __call__(self, inputs):

        se_out = self.model_(inputs)
        return Multiply()([inputs, se_out])
        