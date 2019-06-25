Installation
============

.. contents::
   :depth: 3
   :local:

Requirements
------------

Pytorch
^^^^^^^

Neuralnet-pytorch is built on top of Pytorch, so obviously Pytorch is needed.
Please refer to the official `Pytorch website <https://pytorch.org/>`_ for installation details.


Other dependencies
^^^^^^^^^^^^^^^^^^

In Neuralnet-pytorch, we use several backends to visualize training, so it is necessary to install
some additional packages.

* `Matplotlib <https://matplotlib.org/>`_

* `Visdom <https://github.com/facebookresearch/visdom>`_

* `TensorboardX <https://github.com/lanpa/tensorboardX>`_

To install all dependencies in a go, simply open a Terminal session and execute ::

    pip install matplotlib visdom tensorboardX


Install Neuralnet-pytorch
-------------------------

There are two ways to install Neuralnet-pytorch: via PyPi and Github.
At the moment, the package is not available on Conda yet.

From PyPi
^^^^^^^^^

The easiest and quickest way to get Neuralnet-pytorch is to install the package from Pypi.
In a Terminal session, simply type ::

    pip install neuralnet-pytorch

From Github
^^^^^^^^^^^

To install the bleeding-edge version, which is highly recommended, run ::

    pip install git+git://github.com/justanhduc/neuralnet-pytorch.git@master


We also provide a version with some fancy Cuda/C++ implementations
collected on various sources. To install this version, run ::

    pip install git+git://github.com/justanhduc/neuralnet-pytorch.git@fancy

Uninstall Neuralnet-pytorch
---------------------------

Simply use pip to uninstall the package ::

    pip uninstall neuralnet-pytorch

Why would you want to do that anyway?

Upgrade Neuralnet-pytorch
-------------------------

Use pip with ``-U`` or ``--upgrade`` option ::

    pip install -U neuralnet-pytorch

However, for maximal experience, please considering using the bleeding-edge version on Github.

Reinstall Neuralnet-pytorch
---------------------------

If you want to reinstall Neuralnet-pytorch, please uninstall and then install it again.
When reinstalling the package, we recommend to use ``--no-cache-dir`` option as pip caches
the previously built binaries ::

    pip uninstall neuralnet-pytorch
    pip install neuralnet-pytorch --no-cache-dir


