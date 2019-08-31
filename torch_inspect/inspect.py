import sys
from dataclasses import dataclass
from functools import partial
from typing import List, IO, Union, Type

import numpy as np
import torch
import torch.nn as nn

from torch.utils.hooks import RemovableHandle


__all__ = ('LayerInfo', 'NetworkInfo', 'inspect', 'summary')


CPU = 'cpu'
CUDA = 'cuda'


@dataclass
class LayerInfo:
    name: str
    input_shape: List[int]
    output_shape: List[int]
    trainable: bool
    nb_params: int


@dataclass
class NetworkInfo:
    total_params: int
    trainable_params: int
    total_input_size: int
    total_output_size: int
    total_params_size: int
    total_size: int


def make_network_info(
    info_list: List[LayerInfo], input_size: List[int], batch_size: int = -1
) -> NetworkInfo:
    trainable_params = 0
    total_params = 0
    total_output = 0
    for layer in info_list:
        total_params += layer.nb_params
        total_output += abs(np.prod(layer.output_shape))
        if layer.trainable:
            trainable_params += layer.nb_params

    # assume 4 bytes/number (float on cuda).
    total_input_size = np.prod(input_size) * abs(batch_size) * 4
    # x2 for gradients
    total_output_size = 2 * total_output * 4
    total_params_size = total_params * 4
    total_size = total_params_size + total_output_size + total_input_size
    ni = NetworkInfo(
        total_params,
        trainable_params,
        total_input_size,
        total_output_size,
        total_params_size,
        total_size,
    )
    return ni


def should_attach_hook(model: nn.Module, module: nn.Module) -> bool:
    v = (
        not isinstance(module, nn.Sequential)
        and not isinstance(module, nn.ModuleList)
        and not (module is model)  # exclude self
    )
    return v


_FTensorType = Type[Union[torch.cuda.FloatTensor, torch.FloatTensor]]


def _infer_dtype(device: str) -> _FTensorType:
    device = device.lower()
    if device not in (CUDA, CPU):
        msg = 'Input device is not valid, please specify "cuda" or "cpu"'
        raise ValueError(msg)

    if device == CUDA and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    return dtype


class _ModuleHook:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.layer_list: List[LayerInfo] = []

    def hook(
        self, module: nn.Module, input: _FTensorType, output: _FTensorType
    ) -> None:
        class_name = str(module.__class__).split('.')[-1].split("'")[0]
        module_idx = len(self.layer_list)

        input_shape = list(input[0].size())
        input_shape[0] = self.batch_size

        if isinstance(output, (list, tuple)):
            output_shape = [[-1] + list(o.size())[1:] for o in output]
        else:
            output_shape = list(output.size())
            output_shape[0] = self.batch_size

        params = 0
        trainable = False
        if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
            params += int(
                torch.prod(torch.LongTensor(list(module.weight.size())))
            )
            trainable = module.weight.requires_grad

        if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))

        name = f'{class_name}-{module_idx + 1}'
        layer = LayerInfo(
            name, input_shape, output_shape, trainable, int(params)
        )
        self.layer_list.append(layer)


def inspect(
    model: nn.Module,
    input_size: List[int],
    batch_size: int = -1,
    device: str = CPU,
) -> List[LayerInfo]:

    hook = _ModuleHook(batch_size)
    handles: List[RemovableHandle] = []

    def register_hook(module: nn.Module) -> None:
        if should_attach_hook(model, module):
            handle: RemovableHandle = module.register_forward_hook(hook.hook)
            handles.append(handle)

    dtype = _infer_dtype(device)
    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    model.apply(register_hook)

    try:
        model(*x)
    finally:
        for h in handles:
            h.remove()

    return hook.layer_list


def summary(
    model: nn.Module,
    input_size: List[int],
    batch_size: int = -1,
    device: str = CPU,
    file: IO[str] = sys.stdout,
    flush: bool = False,
) -> None:
    summary = inspect(model, input_size, batch_size=batch_size, device=device)
    n = make_network_info(summary, input_size, batch_size)
    print_ = partial(print, file=file, flush=flush)
    print_('\n')
    print_('----------------------------------------------------------------')
    line_new = '{:>20}  {:>25} {:>15}'.format(
        'Layer (type)', 'Output Shape', 'Param #'
    )
    print_(line_new)
    print_('================================================================')

    for layer in summary:
        line_new = '{:>20}  {:>25} {:>15}'.format(
            layer.name,
            str(layer.output_shape),
            '{0:,}'.format(layer.nb_params),
        )
        print_(line_new)
    MB = 1024 ** 2
    print_('================================================================')
    print_(f'Total params: {n.total_params:,}')
    print_(f'Trainable params: {n.trainable_params:,}')
    print_(f'Non-trainable params: {n.total_params - n.trainable_params:,}')
    print_('----------------------------------------------------------------')
    print_(f'Input size (MB): {n.total_input_size / MB:0.2f}')
    print_(f'Forward/backward pass size (MB): {n.total_output_size / MB:0.2f}')
    print_(f'Params size (MB): {n.total_params_size / MB :0.2f}')
    print_(f'Estimated Total Size (MB): {n.total_size / MB:0.2f}')
    print_('----------------------------------------------------------------')
