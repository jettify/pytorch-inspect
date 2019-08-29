from torch_inspect import inspect
from torch_inspect.inspect import LayerInfo


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
