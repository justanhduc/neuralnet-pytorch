Welcome to Neuralnet-Pytorch's documentation!
=============================================

Personally, going from Theano to Pytorch is pretty much like
time traveling from 90s to the modern day.
However, we feel like despite having a lot of bells and whistles,
Pytorch is still missing many elements
that are confirmed to never be added to the library.
Therefore, this library is written to supplement more features
to the current magical Pytorch.
All the modules in the package directly subclass
the corresponding modules from Pytorch,
so everything should still be familiar.
For example, the following snippet in Pytorch ::

    from torch import nn
    model = nn.Sequential(
        nn.Conv2d(1, 20, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(20, 64, 5, padding=2),
        nn.ReLU()
    )


can be rewritten in Neuralnet-pytorch as ::

    import neuralnet_pytorch as nnt
    model = nnt.Sequential(
        nnt.Conv2d(1, 20, 5, padding='half', activation='relu'),
        nnt.Conv2d(20, 64, 5, padding='half', activation='relu')
    )

which is the same as the native Pytorch, or ::

    import neuralnet_pytorch as nnt
    model = nnt.Sequential(input_shape=1)
    model.add_module('conv1', nnt.Conv2d(model.output_shape, 20, 5, padding='half', activation='relu'))
    model.add_module('conv2', nnt.Conv2d(model.output_shape, 64, 5, padding='half', activation='relu'))

which frees you from doing a lot of manual calculations when adding one layer on top of another. Theano folks will also
find some reminiscence as many functions are highly inspired by Theano.


.. toctree::
   :maxdepth: 4
   :caption: Overview:

   installation
   manual/index

.. toctree::
   :maxdepth: 3
   :caption: Misc Notes:

   license
   help
