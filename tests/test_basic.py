import torchvision
from torch_inspect import inspect, summary


def test_inspect():
    model = torchvision.models.mobilenet_v2()
    r = inspect(model, (3, 224, 224), device='cpu')
    summary(model, (3, 224, 224), batch_size=100, device='cpu')
    assert r


def test_inspect2():
    model = torchvision.models.resnet18()
    r = inspect(model, (3, 224, 224), device='cpu')
    summary(model, (3, 224, 224), batch_size=1, device='cpu')
    assert r
