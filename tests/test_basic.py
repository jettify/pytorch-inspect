import io
import torch

from torch_inspect import inspect, summary
from torch_inspect.inspect import LayerInfo as L, NetworkInfo


def test_inspect(simple_model):
    bs = 2  # default batch size
    r = inspect(simple_model, (1, 32, 32))

    expected = [
        L('Conv2d-1', [bs, 1, 32, 32], [bs, 6, 30, 30], 60, 0),
        L('Conv2d-2', [bs, 6, 15, 15], [bs, 16, 13, 13], 880, 0),
        L('Linear-3', [bs, 576], [bs, 120], 69240, 0),
        L('Linear-4', [bs, 120], [bs, 84], 10164, 0),
        L('Linear-5', [bs, 84], [bs, 10], 850, 0),
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
    bs = 10
    r = inspect(multi_input_net, [(1, 16, 16), (1, 28, 28)], batch_size=bs)
    expected = [
        L('Conv2d-1', [bs, 1, 16, 16], [bs, 1, 16, 16], 10, 0),
        L('ReLU-2', [bs, 1, 16, 16], [bs, 1, 16, 16], 0, 0),
        L('Conv2d-3', [bs, 1, 28, 28], [bs, 1, 28, 28], 10, 0),
        L('ReLU-4', [bs, 1, 28, 28], [bs, 1, 28, 28], 0, 0),
    ]
    assert r == expected


expected_summary = """

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1             [2, 6, 30, 30]              60
            Conv2d-2            [2, 16, 13, 13]             880
            Linear-3                   [2, 120]          69,240
            Linear-4                    [2, 84]          10,164
            Linear-5                    [2, 10]             850
================================================================
Total params: 81,194
Trainable params: 81,194
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.13
Params size (MB): 0.31
Estimated Total Size (MB): 0.44
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
    bs = 10
    r = inspect(netbatchnorm, (20,), batch_size=bs)

    expected = [
        L('Linear-1', [bs, 20], [bs, 15], 300, 0),
        L('BatchNorm1d-2', [bs, 15], [bs, 15], 30, 30),
        L('Linear-3', [bs, 15], [bs, 15], 225, 0),
        L('BatchNorm1d-4', [bs, 15], [bs, 15], 30, 30),
        L('Linear-5', [bs, 15], [bs, 1], 16, 0),
    ]
    assert r == expected
    with io.StringIO() as buf:
        network_info = summary(netbatchnorm, (20,), file=buf, batch_size=bs)
    expected_info = NetworkInfo(661, 601, 80 * bs, 488 * bs, 2644, 8324)
    assert expected_info == network_info


def test_simpleconv(simpleconv):
    bs = 2
    r = inspect(simpleconv, [(1, 16, 16), (1, 28, 28)], batch_size=bs)
    expected = [
        L('Conv2d-1', [bs, 1, 16, 16], [bs, 1, 16, 16], 10, 0),
        L('ReLU-2', [bs, 1, 16, 16], [bs, 1, 16, 16], 0, 0),
        L('Conv2d-3', [bs, 1, 28, 28], [bs, 1, 28, 28], 10, 0),
        L('ReLU-4', [bs, 1, 28, 28], [bs, 1, 28, 28], 0, 0),
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
    bs = 12
    r = inspect(rnn, [(6, 3)], batch_size=bs, input_initializer=torch.zeros)
    expected = [
        L('RNN-1', [bs, 6, 3], [[bs, 6, 5], [3, bs, 5]], 170, 0),
        L('Linear-2', [6 * bs, 5], [6 * bs, 1], 6, 0),
    ]
    assert r == expected


def test_multi_input_net2(multi_input_net2):
    bs = 10
    r = inspect(multi_input_net2, [(3, 128, 1024), (4,)], batch_size=bs)
    expected = [
        L('Conv2d-1', [10, 3, 128, 1024], [10, 64, 128, 1024], 1792, 0),
        L('MaxPool2d-2', [10, 64, 128, 1024], [10, 64, 64, 512], 0, 0),
        L('Conv2d-3', [10, 64, 64, 512], [10, 96, 64, 512], 55392, 0),
        L('MaxPool2d-4', [10, 96, 64, 512], [10, 96, 32, 256], 0, 0),
        L('Conv2d-5', [10, 96, 32, 256], [10, 128, 32, 256], 110720, 0),
        L('MaxPool2d-6', [10, 128, 32, 256], [10, 128, 16, 128], 0, 0),
        L('Conv2d-7', [10, 128, 16, 128], [10, 192, 16, 128], 221376, 0),
        L('AdaptiveAvgPool2d-8', [10, 192, 16, 128], [10, 192, 1, 1], 0, 0),
        L('Linear-9', [10, 192], [10, 64], 12352, 0),
        L('Linear-10', [10, 64], [10, 4], 260, 0),
    ]
    assert r == expected

    expected_info = NetworkInfo(
        401892, 401892, 62914560, 1289769280, 1607568, 1354291408
    )
    with io.StringIO() as buf:
        net_info = summary(
            multi_input_net2, [(3, 128, 1024), (4,)], batch_size=bs, file=buf
        )
    assert net_info == expected_info


def test_lstm_model(lstm_model):
    bs = 10
    r = inspect(
        lstm_model, [(1, 28)], batch_size=bs, input_initializer=torch.zeros
    )
    out = [[10, 1, 100], [[1, 10, 100], [1, 10, 100]]]
    expected = [
        L('LSTM-1', [10, 1, 28], out, 52000, 0),
        L('Linear-2', [10, 100], [10, 10], 1010, 0),
    ]
    assert r == expected


def test_lstm_tagger_with_embedding(lstm_tagger):
    bs = 10
    r = inspect(
        lstm_tagger, [(1, 1)], batch_size=bs, input_initializer=torch.zeros,
        input_dtype=torch.LongTensor,
    )
    expected = [
        L('Embedding-1', [bs, 1, 1], [bs, 1, 1, 6], 30, 0),
        L('LSTM-2', [bs, 1, 6], [[bs, 1, 6], [[1, 1, 6], [1, 1, 6]]], 336, 0),
        L('Linear-3', [bs, 6], [bs, 3], 21, 0)
    ]
    assert r == expected
