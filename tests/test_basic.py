import io
import torch

from torch_inspect import inspect, summary
from torch_inspect.inspect import LayerInfo as L, NetworkInfo


def test_inspect(simple_model):
    r = inspect(simple_model, (1, 32, 32))

    expected = [
        L('Conv2d-1', [-1, 1, 32, 32], [-1, 6, 30, 30], 60, 0),
        L('Conv2d-2', [-1, 6, 15, 15], [-1, 16, 13, 13], 880, 0),
        L('Linear-3', [-1, 576], [-1, 120], 69240, 0),
        L('Linear-4', [-1, 120], [-1, 84], 10164, 0),
        L('Linear-5', [-1, 84], [-1, 10], 850, 0),
    ]

    assert r == expected

    bsize = 10
    r = inspect(
        simple_model,
        (1, 32, 32),
        input_dtype=torch.FloatTensor,
        input_initializer=torch.zeros,
        batch_size=bsize,
    )
    expected = [
        L('Conv2d-1', [bsize, 1, 32, 32], [bsize, 6, 30, 30], 60, 0),
        L('Conv2d-2', [bsize, 6, 15, 15], [bsize, 16, 13, 13], 880, 0),
        L('Linear-3', [bsize, 576], [bsize, 120], 69240, 0),
        L('Linear-4', [bsize, 120], [bsize, 84], 10164, 0),
        L('Linear-5', [bsize, 84], [bsize, 10], 850, 0),
    ]
    assert r == expected


def test_inspect_multi_input(multi_input_net):
    r = inspect(multi_input_net, [(1, 16, 16), (1, 28, 28)])

    expected = [
        L('Conv2d-1', [-1, 1, 16, 16], [-1, 1, 16, 16], 10, 0),
        L('ReLU-2', [-1, 1, 16, 16], [-1, 1, 16, 16], 0, 0),
        L('Conv2d-3', [-1, 1, 28, 28], [-1, 1, 28, 28], 10, 0),
        L('ReLU-4', [-1, 1, 28, 28], [-1, 1, 28, 28], 0, 0),
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
        summary(
            simple_model,
            [(1, 32, 32)],
            input_dtype=torch.FloatTensor,
            input_initializer=torch.zeros,
            file=buf,
            flush=True,
        )
        r = buf.getvalue()
        assert r == expected_summary


def test_inspect_net_with_batch_norm(netbatchnorm):
    r = inspect(netbatchnorm, (20,))

    expected = [
        L('Linear-1', [-1, 20], [-1, 15], 300, 0),
        L('BatchNorm1d-2', [-1, 15], [-1, 15], 30, 30),
        L('Linear-3', [-1, 15], [-1, 15], 225, 0),
        L('BatchNorm1d-4', [-1, 15], [-1, 15], 30, 30),
        L('Linear-5', [-1, 15], [-1, 1], 16, 0),
    ]
    assert r == expected
    with io.StringIO() as buf:
        network_info = summary(netbatchnorm, (20,), file=buf)
    expected_info = NetworkInfo(661, 601, 80, 488, 2644, 3212)
    assert expected_info == network_info


def test_simpleconv(simpleconv):
    bs = -1
    r = inspect(simpleconv, [(1, 16, 16), (1, 28, 28)], batch_size=bs)
    expected = [
        L('Conv2d-1', [-1, 1, 16, 16], [-1, 1, 16, 16], 10, 0),
        L('ReLU-2', [-1, 1, 16, 16], [-1, 1, 16, 16], 0, 0),
        L('Conv2d-3', [-1, 1, 28, 28], [-1, 1, 28, 28], 10, 0),
        L('ReLU-4', [-1, 1, 28, 28], [-1, 1, 28, 28], 0, 0),
    ]
    assert r == expected


def test_autoencoder(autoencoder):
    bs = 10
    r = inspect(autoencoder, [(3, 32, 32)], batch_size=bs)
    expected = [
        L('Conv2d-1', [bs, 3, 32, 32], [bs, 6, 28, 28], 456, 0),
        L('ReLU-2', [bs, 6, 28, 28], [bs, 6, 28, 28], 0, 0),
        L('Conv2d-3', [bs, 6, 28, 28], [bs, 16, 24, 24], 2416, 0),
        L('ReLU-4', [bs, 16, 24, 24], [bs, 16, 24, 24], 0, 0),
        L('ConvTranspose2d-5', [bs, 16, 24, 24], [bs, 6, 28, 28], 2406, 0),
        L('ReLU-6', [bs, 6, 28, 28], [bs, 6, 28, 28], 0, 0),
        L('ConvTranspose2d-7', [bs, 6, 28, 28], [bs, 3, 32, 32], 453, 0),
        L('ReLU-8', [bs, 3, 32, 32], [bs, 3, 32, 32], 0, 0),
        L('Sigmoid-9', [bs, 3, 32, 32], [bs, 3, 32, 32], 0, 0),
    ]
    assert r == expected


def test_rnn(rnn):
    bs = 10
    r = inspect(rnn, [(6, 3)], batch_size=bs, input_initializer=torch.zeros)
    expected = [
        L('RNN-1', [bs, 6, 3], [[bs, 6, 5], [bs, 2, 5]], 170, 0),
        L('Linear-2', [bs, 5], [bs, 1], 6, 0),
    ]
    summary(rnn, [(6, 3)], batch_size=2, input_initializer=torch.zeros)
    assert r == expected
