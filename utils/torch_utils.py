from functools import partial

import torch.nn as nn


def get_activation_fn(activation: str) -> type[nn.Module]:
    if activation.lower() == 'relu':
        return nn.ReLU
    elif activation.lower() == 'leaky_relu':
        return partial(nn.LeakyReLU, negative_slope=0.1)
    elif activation.lower() == 'tanh':
        return nn.Tanh
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid
    elif activation.lower() == 'elu':
        return nn.ELU
    elif activation.lower() == 'selu':
        return nn.SELU
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


def create_linear_layer(in_features: int, out_features: int, add_bn_and_activation: bool, use_batchnorm: bool,
                         activation_fn: type[nn.Module]):
    layers = [nn.Linear(in_features, out_features)]
    if add_bn_and_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(activation_fn())

    return layers


class SoftAverager:
    def __init__(self, smoothing_factor=0.1):
        self.smoothing_factor = smoothing_factor
        self.smoothed_value = None  # Start with no previous value

    def add_value(self, value: float) -> None:
        # Update the smoothed value using the exponential moving average formula
        if self.smoothed_value is None:
            self.smoothed_value = value  # Initialize on the first value
        else:
            self.smoothed_value = (
                    self.smoothing_factor * value + (1 - self.smoothing_factor) * self.smoothed_value
            )
