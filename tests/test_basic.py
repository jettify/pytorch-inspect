import io

import pytest
from torch_inspect import inspect, summary
from torch_inspect.inspect import LayerInfo


def test_inspect_wrong_device(simple_model):
    with pytest.raises(ValueError):
        inspect(simple_model, (1, 32, 32), device='tpu')


def test_inspect(simple_model):
    r = inspect(simple_model, (1, 32, 32), device='cpu')
    L = LayerInfo

    expected = [
        L('Conv2d-1', [-1, 1, 32, 32], [-1, 6, 30, 30], True, 60),
        L('Conv2d-2', [-1, 6, 15, 15], [-1, 16, 13, 13], True, 880),
        L('Linear-3', [-1, 576], [-1, 120], True, 69240),
        L('Linear-4', [-1, 120], [-1, 84], True, 10164),
        L('Linear-5', [-1, 84], [-1, 10], True, 850),
    ]

    assert r == expected

    bsize = 10
    r = inspect(simple_model, (1, 32, 32), bsize, device='cpu')
    expected = [
        L('Conv2d-1', [bsize, 1, 32, 32], [bsize, 6, 30, 30], True, 60),
        L('Conv2d-2', [bsize, 6, 15, 15], [bsize, 16, 13, 13], True, 880),
        L('Linear-3', [bsize, 576], [bsize, 120], True, 69240),
        L('Linear-4', [bsize, 120], [bsize, 84], True, 10164),
        L('Linear-5', [bsize, 84], [bsize, 10], True, 850),
    ]
    assert r == expected


def test_inspect_multi_input(multi_input_net):
    r = inspect(multi_input_net, [(1, 16, 16), (1, 28, 28)], device='cpu')
    L = LayerInfo

    expected = [
        L('Conv2d-1', [-1, 1, 16, 16], [-1, 1, 16, 16], True, 10),
        L('ReLU-2', [-1, 1, 16, 16], [-1, 1, 16, 16], False, 0),
        L('Conv2d-3', [-1, 1, 28, 28], [-1, 1, 28, 28], True, 10),
        L('ReLU-4', [-1, 1, 28, 28], [-1, 1, 28, 28], False, 0),
    ]
    assert r == expected


expected_summary = """

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 30, 30]              60
            Conv2d-2           [-1, 16, 13, 13]             880
            Linear-3                  [-1, 120]          69,240
            Linear-4                   [-1, 84]          10,164
            Linear-5                   [-1, 10]             850
================================================================
Total params: 81,194
Trainable params: 81,194
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 0.31
Estimated Total Size (MB): 0.38
----------------------------------------------------------------
"""


def test_summary(simple_model):
    with io.StringIO() as buf:
        summary(simple_model, (1, 32, 32), device='cpu', file=buf, flush=True)
        r = buf.getvalue()
        assert r == expected_summary
