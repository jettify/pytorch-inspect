import sys
from dataclasses import dataclass
from functools import partial
from typing import List, IO, Union, Tuple, Type, Callable

import numpy as np
import torch
import torch.nn as nn

from torch.utils.hooks import RemovableHandle


__all__ = ('LayerInfo', 'NetworkInfo', 'inspect', 'summary')


InputShape = Tuple[int, ...]
OutputShape = List[Union[int, List[int]]]


@dataclass
class LayerInfo:
    name: str
    input_shape: List[int]
    output_shape: OutputShape
    trainable_params: int
    non_trainable_params: int


@dataclass
class NetworkInfo:
    total_params: int
    trainable_params: int
    total_input_size: int
    total_output_size: int
    total_params_size: int
    total_size: int


def make_network_info(
    info_list: List[LayerInfo],
    input_size: Union[InputShape, List[InputShape]],
    batch_size: int = 2,
) -> NetworkInfo:
    trainable_params = 0
    total_params = 0
    total_output = 0
    for layer in info_list:
        total_params += layer.trainable_params + layer.non_trainable_params
        total_output += abs(np.prod(layer.output_shape))
        trainable_params += layer.trainable_params

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # assume 4 bytes/number (float on cuda).
    total_input_size = np.prod(sum(input_size, ())) * abs(batch_size) * 4
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


def _has_running_mean(module: nn.Module) -> bool:
    return (
        hasattr(module, 'running_mean')
        and hasattr(module.running_mean, 'size')
        and hasattr(module, 'track_running_stats')
        and module.track_running_stats
    )


def _has_running_var(module: nn.Module) -> bool:
    return (
        hasattr(module, 'running_var')
        and hasattr(module.running_var, 'size')
        and hasattr(module, 'track_running_stats')
        and module.track_running_stats
    )


class _ModuleHook:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.layer_list: List[LayerInfo] = []

    def hook(
        self,
        module: nn.Module,
        input: torch.FloatTensor,
        output: torch.FloatTensor,
    ) -> None:

        # make layer name
        module_idx = len(self.layer_list)
        class_name = str(module.__class__).split('.')[-1].split("'")[0]
        name = f'{class_name}-{module_idx + 1}'

        # calculate input shape
        input_shape = list(input[0].size())
        output_shape = infer_shape(output)

        # calculate number of params
        trainable_params = 0
        non_trainable_params = 0

        for _, param in module.named_parameters():
            params = np.prod(param.size())
            if param.requires_grad:
                trainable_params += params
            else:
                non_trainable_params += params

        if _has_running_mean(module):
            params = np.prod(module.running_mean.size())
            non_trainable_params += params

        if _has_running_var(module):
            params = np.prod(module.running_var.size())
            non_trainable_params += params

        # recored result
        layer = LayerInfo(
            name,
            input_shape,
            output_shape,
            int(trainable_params),
            int(non_trainable_params),
        )
        self.layer_list.append(layer)


Output = Union[Tuple[torch.Tensor, ...], torch.Tensor]


def infer_shape(output: Output) -> OutputShape:
    shape: OutputShape = []
    if isinstance(output, (list, tuple)):
        shape = [infer_shape(o) for o in output]  # type: ignore
    else:
        shape = [int(s) for s in output.size()]
    return shape


def inspect(
    model: nn.Module,
    input_size: Union[InputShape, List[InputShape]],
    input_dtype: Type[torch.Tensor] = torch.FloatTensor,
    input_initializer: Callable[..., torch.Tensor] = torch.rand,
    batch_size: int = 2,
) -> List[LayerInfo]:
    hook = _ModuleHook(batch_size)
    handles: List[RemovableHandle] = []

    def register_hook(module: nn.Module) -> None:
        if should_attach_hook(model, module):
            h: RemovableHandle = module.register_forward_hook(
                hook.hook
            )
            handles.append(h)

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # make fake input with batch_size of 2 for batchnorm
    x = [
        input_initializer(batch_size, *in_size).type(
            input_dtype  # type: ignore
        )
        for in_size in input_size
    ]
    # attach hooks to each applicable layer
    model.apply(register_hook)

    # forward pass
    try:
        model(*x)
    finally:
        # cleanup all attached hooks, to move model to original state
        for h in handles:
            h.remove()  # type: ignore

    return hook.layer_list


def summary(
    model: nn.Module,
    input_size: Union[InputShape, List[InputShape]],
    input_dtype: Type[torch.Tensor] = torch.FloatTensor,
    input_initializer: Callable[..., torch.Tensor] = torch.rand,
    batch_size: int = 2,
    file: IO[str] = sys.stdout,
    flush: bool = False,
) -> NetworkInfo:
    summary = inspect(
        model,
        input_size,
        input_dtype=input_dtype,
        input_initializer=input_initializer,
        batch_size=batch_size,
    )
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
        total_params = layer.trainable_params + layer.non_trainable_params
        line_new = '{:>20}  {:>25} {:>15}'.format(
            layer.name, str(layer.output_shape), '{0:,}'.format(total_params)
        )
        print_(line_new)
    MB = 1024.0 ** 2
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
    return n
