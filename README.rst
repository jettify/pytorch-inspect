torch-inspect
=============
.. image:: https://travis-ci.com/jettify/pytorch-inspect.svg?branch=master
    :target: https://travis-ci.com/jettify/pytorch-inspect
.. image:: https://codecov.io/gh/jettify/pytorch-inspect/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jettify/pytorch-inspect
.. image:: https://img.shields.io/pypi/pyversions/torch-inspect.svg
    :target: https://pypi.org/project/torch-inspect
.. image:: https://img.shields.io/pypi/v/torch-inspect.svg
    :target: https://pypi.python.org/pypi/torch-inspect

**torch-inspect** -- collection of utility functions to inspect low level information of neural network for PyTorch_

Features
========
* Provides helper function ``summary`` that prints Keras style model summary.
* Provides helper function ``inspect`` that returns object with network summary information for programmatic access.
* RNN/LSTM support.
* Library has tests and reasonable code coverage.


Simple example
--------------

.. code:: python

    import torch.nn as nn
    import torch.nn.functional as F
    import torch_inspect as ti

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.fc1 = nn.Linear(16 * 6 * 6, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


      net = SimpleNet()
      ti.summary(net, (1, 32, 32))


Will produce following output:

.. code::

   ----------------------------------------------------------------
           Layer (type)               Output Shape         Param #
   ================================================================
               Conv2d-1           [100, 6, 30, 30]              60
               Conv2d-2          [100, 16, 13, 13]             880
               Linear-3                 [100, 120]          69,240
               Linear-4                  [100, 84]          10,164
               Linear-5                  [100, 10]             850
   ================================================================
   Total params: 81,194
   Trainable params: 81,194
   Non-trainable params: 0
   ----------------------------------------------------------------
   Input size (MB): 0.39
   Forward/backward pass size (MB): 6.35
   Params size (MB): 0.31
   Estimated Total Size (MB): 7.05
   ----------------------------------------------------------------

For programmatic access to network information there is ``inspect`` function:

.. code:: python

      info = ti.inspect(net, (1, 32, 32))
      print(info)


.. code::

     [LayerInfo(name='Conv2d-1', input_shape=[100, 1, 32, 32], output_shape=[100, 6, 30, 30], trainable_params=60, non_trainable_params=0),
      LayerInfo(name='Conv2d-2', input_shape=[100, 6, 15, 15], output_shape=[100, 16, 13, 13], trainable_params=880, non_trainable_params=0),
      LayerInfo(name='Linear-3', input_shape=[100, 576], output_shape=[100, 120], trainable_params=69240, non_trainable_params=0),
      LayerInfo(name='Linear-4', input_shape=[100, 120], output_shape=[100, 84], trainable_params=10164, non_trainable_params=0),
      LayerInfo(name='Linear-5', input_shape=[100, 84], output_shape=[100, 10], trainable_params=850, non_trainable_params=0)]


Installation
------------
Installation process is simple, just::

    $ pip install torch-inspect


Requirements
------------

* Python_ 3.6+
* PyTorch_ 1.0+


References and Thanks
---------------------
This package is based on pytorch-summary_ and  PyTorch issue_ . Compared to
pytorch-summary_, *pytorch-inspect* has support of RNN/LSTMs, also provides programmatic
access to the network summary information. With a bit more modular structure and presence of tests
it is easier to extend and support more features.


.. _Python: https://www.python.org
.. _PyTorch: https://github.com/pytorch/pytorch
.. _pytorch-summary:  https://github.com/sksq96/pytorch-summary
.. _issue:  https://github.com/pytorch/pytorch/issues/2001
