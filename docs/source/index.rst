.. neuralnet-pytorch documentation master file, created by
   sphinx-quickstart on Sun Apr  7 14:16:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Neuralnet-Pytorch's documentation!
=============================================

Personally, going from Theano to Pytorch is pretty much like time traveling from 90s to the modern day.
However, despite a lot of bells and whistles, I still feel there are some missing elements from Pytorch
which are confirmed to be never added to the library.
Therefore, this library is written to add more features to the current magical Pytorch. All the modules here
directly subclass the corresponding modules from Pytorch, so everything should still be familiar. For example, the
following snippet in Pytorch ::

    from torch import nn
    model = nn.Sequential()
    model.add_module('conv1', nn.Conv2d(1, 20, 5, padding=2))
    model.add_module('relu1', nn.ReLU())
    model.add_module('conv2', nn.Conv2d(20, 64, 5, padding=2))
    model.add_module('relu2', nn.ReLU())

can be rewritten in Neuralnet-pytorch as ::

    import neuralnet_pytorch as nnt
    model = nnt.Sequential((None, 1, None, None))
    model.add_module('conv1', nnt.Conv2d(model.output_shape, 20, 5, padding='half', activation='relu'))
    model.add_module('conv2', nnt.Conv2d(model.output_shape, 64, 5, padding='half', activation='relu'))

which frees you from doing a lot of manual calculations when adding one layer on top of another. Theano folks will also
find some reminiscence as many functions are highly inspired by Theano.


.. toctree::
   :maxdepth: 2
   :caption: Overview:

   installation
   reference-manual

.. toctree::
   :maxdepth: 2
   :caption: Misc Notes:

   license
   help
