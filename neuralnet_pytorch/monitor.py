from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import atexit
import threading
import queue
import numpy as np
import collections
import pickle as pkl
from imageio import imwrite
import os
import time
from datetime import datetime as dt
import visdom
from shutil import copyfile
import torch as T
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import neuralnet_pytorch as nnt
from neuralnet_pytorch import utils

__all__ = ['Monitor', 'track', 'get_tracked_variables', 'eval_tracked_variables', 'hooks']
_TRACKS = collections.OrderedDict()
hooks = {}


def track(name, x, direction=None):
    """
    An identity function that registers hooks to
    track the value and gradient of the specified tensor.

    Here is an example of how to track an intermediate output ::

        input = ...
        conv1 = nnt.track('op', nnt.Conv2d(shape, 4, 3), 'all')
        conv2 = nnt.Conv2d(conv1.output_shape, 5, 3)
        intermediate = conv1(input)
        output = nnt.track('conv2_output', conv2(intermediate), 'all')
        loss = T.sum(output ** 2)
        loss.backward(retain_graph=True)
        d_inter = T.autograd.grad(loss, intermediate, retain_graph=True)
        d_out = T.autograd.grad(loss, output)
        tracked = nnt.eval_tracked_variables()

        testing.assert_allclose(tracked['conv2_output'], nnt.utils.to_numpy(output))
        testing.assert_allclose(np.stack(tracked['grad_conv2_output']), nnt.utils.to_numpy(d_out[0]))
        testing.assert_allclose(tracked['op'], nnt.utils.to_numpy(intermediate))
        for d_inter_, tracked_d_inter_ in zip(d_inter, tracked['grad_op_output']):
            testing.assert_allclose(tracked_d_inter_, nnt.utils.to_numpy(d_inter_))

    :param name:
        name of the tracked tensor.
    :param x:
        tensor or module to be tracked.
        If module, the output of the module will be tracked.
    :param direction:
        there are 4 options

        ``None``: tracks only value.

        ```forward```: tracks only value.

        ```backward```: tracks only gradient.

        ```all``: tracks both value and gradient.

        Default: ``None``.
    :return: `x`.
    """

    assert isinstance(name, str), 'name must be a string, got %s' % type(name)
    assert isinstance(x, (T.nn.Module, T.Tensor)), 'x must be a Torch Module or Tensor, got %s' % type(x)
    assert direction in (
        'forward', 'backward', 'all', None), 'direction must be None, \'forward\', \'backward\', or \'all\''

    if isinstance(x, T.nn.Module):
        if direction in ('forward', 'all', None):
            def _forward_hook(module, input, output):
                _TRACKS[name] = output.detach()

            hooks[name] = x.register_forward_hook(_forward_hook)

        if direction in ('backward', 'all'):
            def _backward_hook(module, grad_input, grad_output):
                _TRACKS['grad_' + name + '_output'] = tuple([grad_out.detach() for grad_out in grad_output])

            hooks['grad_' + name + '_output'] = x.register_backward_hook(_backward_hook)
    else:
        if direction in ('forward', 'all', None):
            _TRACKS[name] = x.detach()

        if direction in ('backward', 'all'):
            def _hook(grad):
                _TRACKS['grad_' + name] = tuple([grad_.detach() for grad_ in grad])

            hooks['grad_' + name] = x.register_hook(_hook)

    return x


def get_tracked_variables(name=None, return_name=False):
    """
    Gets tracked variable given name.

    :param name:
        name of the tracked variable.
        can be ``str`` or``list``/``tuple`` of ``str``s.
        If ``None``, all the tracked variables will be returned.
    :param return_name:
        whether to return the names of the tracked variables.
    :return:
        the tracked variables.
    """

    assert isinstance(name, (str, list, tuple)) or name is None, 'name must either be None, a tring, or a list/tuple.'
    if name is None:
        tracked = ([n for n in _TRACKS.keys()], [val for val in _TRACKS.values()]) if return_name \
            else [val for val in _TRACKS.values()]
        return tracked
    elif isinstance(name, (list, tuple)):
        tracked = (name, [_TRACKS[n] for n in name]) if return_name else [_TRACKS[n] for n in name]
        return tracked
    else:
        tracked = (name, _TRACKS[name]) if return_name else _TRACKS[name]
        return tracked


def eval_tracked_variables():
    """
    Retrieves the values of tracked variables.

    :return: a dictionary containing the values of tracked variables
        associated with the given names.
    """

    name, vars = get_tracked_variables(return_name=True)
    dict = collections.OrderedDict()
    for n, v in zip(name, vars):
        if isinstance(v, (list, tuple)):
            dict[n] = [val.item() if val.numel() == 1 else utils.to_numpy(val) for val in v]
        else:
            dict[n] = v.item() if v.numel() == 1 else utils.to_numpy(v)
    return dict


def _spawn_defaultdict_ordereddict():
    return collections.OrderedDict()


class Monitor:
    """
    Collects statistics and displays the results using various backends.
    The collected stats are stored in '<root>/<model_name>/run<#id>'
    where #id is automatically assigned each time a new run starts.

    Examples
    --------
    The following snippet shows how to collect statistics and display them
    every 100 iterations.

    .. code-block:: python

        import neuralnet as nnt

        ...
        mon = nnt.Monitor(print_freq=100)
        for epoch in mon.iter_epoch(range(n_epochs)):
            for data in mon.iter_batch(data_loader):
                loss = net(data)
                mon.plot('training loss', loss)
                mon.imwrite('input images', data['images'])
        ...

    Parameters
    ----------
    model_name : str
        name of the model folder.
    root : str
        path to store the collected statistics.
    current_folder : str
        if given, all the stats in here will be overwritten or resumed.
    print_freq : int
        statistics display frequency.
        Unit is iteration.
    num_iters : int
        number of iterations/epoch.
        If specified, training iteration percentage will be displayed along with epoch.
        Otherwise, it will be automatically calculated after the first epoch.
    use_visdom : bool
        whether to use Visdom for real-time monitoring.
    use_tensorboard : bool
        whether to use Tensorboard for real-time monitoring.
    send_slack : bool
        whether to send the statistics to Slack chatroom.
    kwargs
        some miscellaneous options for Visdom and other functions.

    Attributes
    ----------
    path
        contains all the runs of `model_name`.
    current_folder
        path to the current run.
    vis
        an instance of :mod:`Visdom` when `use_visdom` is set to ``True``.
    writer
        an instance of Tensorboard's :class:`SummaryWriter`
        when `use_tensorboard` is set to ``True``.
    plot_folder
        path to the folder containing the collected plots.
    file_folder
        path to the folder containing the collected files.
    image_folder
        path to the folder containing the collected images.
    hist_folder
        path to the folder containing the collected histograms.
    """

    def __init__(self, model_name='my_model', root='results', current_folder=None, print_freq=None, num_iters=None,
                 use_visdom=False, use_tensorboard=False, send_slack=False, **kwargs):
        self._iter = 0
        self._last_epoch = 0
        self._num_since_beginning = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._num_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._hist_since_beginning = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._hist_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._img_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._points_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._options = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._dump_files = collections.OrderedDict()
        self._schedule = {'beginning': collections.defaultdict(_spawn_defaultdict_ordereddict),
                          'end': collections.defaultdict(_spawn_defaultdict_ordereddict)}
        self._timer = time.time()
        self._io_method = {'pickle_save': self._save_pickle, 'txt_save': self._save_txt,
                           'torch_save': self._save_torch, 'pickle_load': self._load_pickle,
                           'txt_load': self._load_txt, 'torch_load': self._load_torch}
        self._prefix = kwargs.pop('prefix', 'run')
        self._num_iters = num_iters

        self.print_freq = print_freq
        self.num_iters = num_iters
        if current_folder:
            self.current_folder = current_folder
            try:
                log = self.read_log('log.pkl')
                try:
                    self.num_stats = log['num']
                except KeyError:
                    print('No record found for \'num\'')

                try:
                    self.hist_stats = log['hist']
                except KeyError:
                    print('No record found for \'hist\'')

                try:
                    self.options = log['options']
                except KeyError:
                    print('No record found for \'options\'')

                if self.num_iters is None:
                    try:
                        self.num_iters = log['num_iters']
                    except KeyError:
                        print('No record found for \'num_iters\'')

                try:
                    self.epoch = log['epoch']
                except KeyError:
                    if self.num_iters:
                        self.epoch = log['iter'] // self.num_iters
                    else:
                        print('No record found for \'epoch\'')

                try:
                    self.iter = self.epoch * self.num_iters if self.num_iters else log['iter']
                except KeyError:
                    print('No record found for \'iter\'')

            except FileNotFoundError:
                print('\'log.pkl\' not found in \'%s\'' % os.path.join(self.current_folder, 'files'))

        else:
            self.path = os.path.join(root, model_name)
            os.makedirs(self.path, exist_ok=True)
            self.current_folder = self._get_new_folder()
            os.mkdir(self.current_folder)

        self.use_visdom = use_visdom
        if use_visdom:
            if kwargs.pop('disable_visdom_logging', True):
                import logging
                logging.disable(logging.CRITICAL)

            server = kwargs.pop('server', 'http://localhost')
            port = kwargs.pop('port', 8097)
            self.vis = visdom.Visdom(server=server, port=port)
            if not self.vis.check_connection():
                from subprocess import Popen, PIPE
                Popen('visdom', stdout=PIPE, stderr=PIPE)

            self.vis.close()
            print('You can navigate to \'%s:%d\' for visualization' % (server, port))

        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            os.makedirs(os.path.join(self.current_folder, 'tensorboard'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.current_folder, 'tensorboard'))

        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

        self.send_slack = send_slack
        if send_slack:
            assert kwargs.get('channel', None) is not None and kwargs.get('token', None) is not None, \
                'channel and token must be provided to send a slack message'

            if kwargs.get('username', None) is None:
                kwargs['username'] = 'me'

        self.kwargs = kwargs

        # make folders to store statistics
        self.plot_folder = os.path.join(self.current_folder, 'plots')
        os.makedirs(self.plot_folder, exist_ok=True)

        self.file_folder = os.path.join(self.current_folder, 'files')
        os.makedirs(self.file_folder, exist_ok=True)

        self.image_folder = os.path.join(self.current_folder, 'images')
        os.makedirs(self.image_folder, exist_ok=True)

        self.hist_folder = os.path.join(self.current_folder, 'histograms')
        os.makedirs(self.hist_folder, exist_ok=True)

        # schedule to flush when the program finishes
        atexit.register(self._atexit)
        print('Result folder: %s' % self.current_folder)

    def _get_new_folder(self):
        runs = [folder for folder in os.listdir(self.path) if folder.startswith(self._prefix)]
        if not runs:
            idx = 1
        else:
            indices = sorted([int(r[len(self._prefix):]) for r in runs])
            idx = indices[-1] + 1

        return os.path.join(self.path, '{}{}'.format(self._prefix, idx))

    def iter_epoch(self, iterator):
        """
        tracks training epoch and returns the item in `iterator`.

        :param iterator:
            the epoch iterator.
            For e.g., ``range(num_epochs)``.
        :return:
            a generator over `iterator`.

        Examples
        --------

        >>> num_epochs = 10
        >>> mon = Monitor(print_freq=1000)
        >>> for epoch in mon.iter_epoch(range(num_epochs))
        ...     # do something here

        See Also
        --------
        :meth:`~iter_batch`
        """

        for item in iterator:
            if self.epoch > 0 and self.num_iters is None:
                self.num_iters = self.iter // self.epoch

            yield item
            self.epoch += 1

    def iter_batch(self, iterator):
        """
        tracks training iteration and returns the item in `iterator`.

        :param iterator:
            the batch iterator.
            For e.g., ``enumerator(loader)``.
        :return:
            a generator over `iterator`.

        Examples
        --------

        >>> dataset = ...
        >>> data_loader = ...
        >>> num_epochs = 10
        >>> mon = Monitor(print_freq=1000)
        >>> for epoch in mon.iter_epoch(range(num_epochs)):
        ...     for data in mon.iter_batch(enumerate(data_loader)):
        ...         # do something here

        See Also
        --------
        :meth:`~iter_epoch`
        """

        for item in iterator:
            yield item
            if self.print_freq:
                if self.iter % self.print_freq == 0:
                    self.flush()

            self.iter += 1

    @utils.deprecated(iter_batch, '1.2.0')
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.print_freq:
            if self.iter % self.print_freq == 0:
                self.flush()
        self.tick()

    @property
    def iter(self):
        """
        returns the current iteration.

        :return:
            :attr:`~_iter`.
        """

        return self._iter

    @iter.setter
    def iter(self, iter):
        """
        sets the iteration counter to a specific value.

        :param iter:
            the iteration number to set.
        :return:
            ``None``.
        """
        assert iter >= 0, 'Iteration must be non-negative'
        self._iter = int(iter)

    @property
    def epoch(self):
        """
        returns the current epoch.

        :return:
            :attr:`~_last_epoch`.
        """

        return self._last_epoch

    @epoch.setter
    def epoch(self, epoch):
        """
        sets the epoch for logging and keeping training status.
        Should start from 0.

        :param epoch:
            epoch number. Should start from 0.
        :return:
            ``None``.
        """

        assert epoch >= 0, 'Epoch must be non-negative'
        self._last_epoch = int(epoch)

    @property
    def num_stats(self):
        """
        returns the collected scalar statistics from beginning.

        :return:
            :attr:`~_num_since_beginning`.
        """

        return dict(self._num_since_beginning)

    @num_stats.setter
    def num_stats(self, stats_dict):
        self._num_since_beginning.update(stats_dict)

    @num_stats.deleter
    def num_stats(self):
        self._num_since_beginning.clear()

    def clear_num_stats(self, key):
        """
        removes the collected statistics for scalar plot of the specified `key`.

        :param key:
            the name of the scalar collection.
        :return: ``None``.
        """

        self._num_since_beginning[key].clear()

    @property
    def hist_stats(self):
        """
        returns the collected tensors from beginning.

        :return:
            :attr:`~_hist_since_beginning`.
        """

        return dict(self._hist_since_beginning)

    @hist_stats.setter
    def hist_stats(self, stats_dict):
        self._hist_since_beginning.update(stats_dict)

    @hist_stats.deleter
    def hist_stats(self):
        self._hist_since_beginning.clear()

    def clear_hist_stats(self, key):
        """
        removes the collected statistics for histogram plot of the specified `key`.

        :param key:
            the name of the histogram collection.
        :return: ``None``.
        """

        self._hist_since_beginning[key].clear()

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options_dict):
        self._options.update(options_dict)

    @options.deleter
    def options(self):
        self._options.clear()

    def set_option(self, name, option, value):
        """
        sets option for histogram plotting.

        :param name:
             name of the histogram plot.
             Must be the same as the one specified when using :meth:`~hist`.
        :param option:
            there are two options which should be passed as a ``str``.

            ```last_only```: plot the histogram of the last recorded tensor only.

            ```n_bins```: number of bins of the histogram.
        :param value:
            value of the chosen option. Should be ``True``/``False`` for ```last_only```
            and an integer for ```n_bins```.
        :return: ``None``.
        """

        self._options[name][option] = value

    def run_training(self, net, optim, train_loader, n_epochs, eval_loader=None, valid_freq=None, start_epoch=None,
                     train_stats_func=None, val_stats_func=None, *args, **kwargs):
        """
        Runs the training loop for the given neural network.

        :param net:
            must be an instance of :class:`~neuralnet_pytorch.layers.Net`
            and :class:`~neuralnet_pytorch.layers.Module`.
        :param train_loader:
            provides training data for neural net.
        :param n_epochs:
            number of training epochs.
        :param eval_loader:
            provides validation data for neural net. Optional.
        :param valid_freq:
            indicates how often validation is run.
            In effect if only `eval_loader` is given.
        :param start_epoch:
            the epoch from which training will continue.
            If ``None``, training counter will be set to 0.
        :param train_stats_func:
            a custom function to handle statistics returned from the training procedure.
            If ``None``, a default handler will be used.
            For a list of supported statistics, see :class:`~neuralnet_pytorch.layers.Net`.
        :param val_stats_func:
            a custom function to handle statistics returned from the validation procedure.
            If ``None``, a default handler will be used.
            For a list of suported statistics, see :class:`~neuralnet_pytorch.layers.Net`.
        :param args:
            additional arguments that will be passed to neural net.
        :param kwargs:
            additional keyword arguments that will be passed to neural net.
        :return: ``None``.

        Examples
        --------

        .. code-block:: python

            import neuralnet_pytorch as nnt

            class MyNet(nnt.Net, nnt.Module):
                ...

            # define the network, and training and testing loaders
            net = MyNet(...)
            train_loader = ...
            eval_loader = ...

            # instantiate a Monitor object
            mon = Monitor(model_name='my_net', print_freq=100)

            # save string representations of the model, optimization and lr scheduler
            mon.dump_model(net)
            mon.dump_rep('optimizer', optim['optimizer'])
            if optim['scheduler']:
                mon.dump_rep('scheduler', optim['scheduler'])

            # collect the parameters of the network
            states = {
                'model_state_dict': net.state_dict(),
                'opt_state_dict': optim['optimizer'].state_dict()
            }
            if optim['scheduler']:
                states['scheduler_state_dict'] = optim['scheduler'].state_dict()

            # save a checkpoint after each epoch and keep only the 5 latest checkpoints
            mon.schedule(mon.dump, optim, beginning=False, name='training.pt', obj=states, type='torch', keep=5)

            print('Training...')

            # run the training loop
            mon.run_training(net, train_loader, n_epochs, eval_loader, valid_freq=val_freq)
            print('Training finished!')
        """

        assert isinstance(net, nnt.Net), 'net must be an instance of Net'
        assert isinstance(net, (nnt.Module, nn.Module, nnt.Sequential, nn.Sequential)), \
            'net must be an instance of Module or Sequential'

        collect = {
            'scalars': self.plot,
            'images': self.imwrite,
            'histograms': self.hist,
            'pointclouds': self.scatter
        }

        start_epoch = self.epoch if start_epoch is None else start_epoch
        for epoch in self.iter_epoch(range(start_epoch, n_epochs)):
            if optim['scheduler'] is not None:
                optim['scheduler'].step(epoch)

            for func_dict in self._schedule['beginning'].values():
                func_dict['func'](*func_dict['args'], **func_dict['kwargs'])

            for it, batch in self.iter_batch(enumerate(train_loader)):
                net.train(True)
                if nnt.cuda_available:
                    batch = utils.batch_to_cuda(batch)

                net.learn(optim, *batch, *args, **kwargs)

                if train_stats_func is None:
                    for t, d in net.stats['train'].items():
                        for k, v in d.items():
                            if t == 'scalars':
                                if np.isnan(v) or np.isinf(v):
                                    raise ValueError('{} is NaN/inf. Training failed!'.format(k))

                            collect[t](k, v)
                else:
                    train_stats_func(net.stats['train'], it)

                if valid_freq:
                    if self.iter % valid_freq == 0:
                        net.eval()

                        with T.set_grad_enabled(False):
                            eval_dict = {
                                'scalars': collections.defaultdict(lambda: []),
                                'histograms': collections.defaultdict(lambda: [])
                            }

                            for itt, batch in enumerate(eval_loader):
                                if nnt.cuda_available:
                                    batch = utils.batch_to_cuda(batch)

                                try:
                                    net.eval_procedure(*batch, *args, **kwargs)
                                except NotImplementedError:
                                    print('An evaluation procedure must be specified')
                                    raise

                                if val_stats_func is None:
                                    for t, d in net.stats['eval'].items():
                                        if t in ('scalars', 'histograms'):
                                            for k, v in d.items():
                                                eval_dict[t][k].append(v)
                                        else:
                                            for k, v in d.items():
                                                collect[t](k + '_%d' % itt, v)
                                else:
                                    val_stats_func(net.stats['eval'], itt)

                            for t in ('scalars', 'histograms'):
                                for k, v in eval_dict[t].items():
                                    v = np.mean(v) if t == 'scalars' else np.concatenate(v)
                                    collect[t](k, v)

            for func_dict in self._schedule['end'].values():
                func_dict['func'](*func_dict['args'], **func_dict['kwargs'])

    def _atexit(self):
        self.flush()
        plt.close()
        if self.use_tensorboard:
            self.writer.close()

        self._q.join()

    def dump_rep(self, name, obj):
        """
        saves a string representation of the given object.

        :param name:
            name of the txt file containing the string representation.
        :param obj:
            object to saved as string representation.
        :return: ``None``.
        """

        with open(os.path.join(self.current_folder, name + '.txt'), 'w') as outfile:
            outfile.write(str(obj))
            outfile.close()

    def dump_model(self, network, *args, **kwargs):
        """
        saves a string representation of the given neural net.

        :param network:
            neural net to be saved as string representation.
        :param args:
            additional arguments to Tensorboard's :meth:`SummaryWriter`
            when :attr:`~use_tensorboard` is ``True``.
        :param kwargs:
            additional keyword arguments to Tensorboard's :meth:`SummaryWriter`
            when :attr:`~use_tensorboard` is ``True``.
        :return: ``None``.
        """

        assert isinstance(network, (
            nn.Module, nn.Sequential)), 'network must be an instance of Module or Sequential, got {}'.format(
            type(network))
        self.dump_rep('network.txt', network)

    def backup(self, files):
        """
        saves a copy of the given files to :attr:`~current_folder`.
        Accepts a str or list/tuple of file names.
        You can backup your codes and/or config files for later use.

        :param files:
            file to be saved.
        :return: ``None``.
        """
        assert isinstance(files, (str, list, tuple)), \
            'unknown type of \'files\'. Expect list, tuple or string, got {}'.format(type(files))

        files = (files,) if isinstance(files, str) else files
        for file in files:
            try:
                copyfile(file, '%s/%s' % (self.file_folder, os.path.split(file)[1]))
            except FileNotFoundError:
                print('No such file or directory: %s' % file)

    @utils.deprecated(backup, '1.1.0')
    def copy_files(self, files):
        self.backup(files)

    def tick(self):
        """
        increases the iteration counter by 1.
        Do not call this if using :class:`Monitor`'s context manager mode.

        :return: ``None``.
        """
        self.iter += 1

    def plot(self, name: str, value):
        """
        schedules a plot of scalar value.
        A :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            scalar value to be plotted.
        :return: ``None``.
        """

        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._num_since_last_flush[name][self.iter] = value
        if self.use_tensorboard:
            self.writer.add_scalar('data/' + name.replace(' ', '-'), value, self.iter)

    def scatter(self, name: str, value, latest=False):
        """
        schedules a scattor plot of (a batch of) points.
        A 3D :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            2D or 3D tensor to be plotted. The last dim should be 3.
        :param latest:
            whether to save only the latest statistics or keep everything from beginning.
        :return: ``None``.
        """

        if self.iter == 0:
            self._options[name]['last_only'] = latest

        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._points_since_last_flush[name][self.iter] = value

    def imwrite(self, name: str, value, latest=False):
        """
        schedules to save images.
        The images will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            3D or 4D tensor to be plotted.
            The expected shape is (C, H, W) for 3D tensors and
            (N, C, H, W) for 4D tensors.
            If the number of channels is other than 3 or 1, each channel is saved as
            a gray image.
        :param latest:
            whether to save only the latest statistics or keep everything from beginning.
        :return: ``None``.
        """

        if self.iter == 0:
            self._options[name]['last_only'] = latest

        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._img_since_last_flush[name][self.iter] = value
        if self.use_tensorboard:
            for idx, img in enumerate(value):
                self.writer.add_image('image/' + name.replace(' ', '-') + '-%d' % idx, img, self.iter)

    def hist(self, name, value, n_bins=20, latest=False):
        """
        schedules a histogram plot of (a batch of) points.
        A :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            any-dim tensor to be histogrammed. The last dim should be 3.
        :param n_bins:
            number of bins of the histogram.
        :param latest:
            whether to save only the latest statistics or keep everything from beginning.
        :return: ``None``.
        """

        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        if self.iter == 0:
            self._options[name]['last_only'] = latest
            self._options[name]['n_bins'] = n_bins

        self._hist_since_last_flush[name][self.iter] = value
        if self.use_tensorboard:
            self.writer.add_histogram('hist/' + name.replace(' ', '-'), value, self.iter)

    def schedule(self, func, beginning=True, *args, **kwargs):
        """
        uses to schedule a routine during every epoch in :meth:`~run_training`.

        :param func:
            a routine to be executed in :meth:`~run_training`.
        :param beginning:
            whether to run at the beginning of every epoch. Default: ``True``.
        :param args:
            additional arguments to `func`.
        :param kwargs:
            additional keyword arguments to `func`.
        :return: ``None``
        """

        assert callable(func), 'func must be callable'
        name = func.__name__
        when = 'beginning' if beginning else 'end'
        self._schedule[when][name]['func'] = func
        self._schedule[when][name]['args'] = args
        self._schedule[when][name]['kwargs'] = kwargs

    def _worker(self, it, epoch, _num_since_last_flush, _img_since_last_flush, _hist_since_last_flush,
                _pointcloud_since_last_flush):
        prints = []

        # plot statistics
        fig = plt.figure()
        plt.xlabel('iteration')
        for name, val in list(_num_since_last_flush.items()):
            self._num_since_beginning[name].update(val)

            x_vals = list(self._num_since_beginning[name].keys())
            plt.ylabel(name)
            y_vals = [self._num_since_beginning[name][x] for x in x_vals]
            if isinstance(y_vals[0], dict):
                keys = list(y_vals[0].keys())
                y_vals = [tuple([y_val[k] for k in keys]) for y_val in y_vals]
                plot = plt.plot(x_vals, y_vals)
                plt.legend(plot, keys)
                prints.append(
                    "{}\t{:.5f}".format(name, np.mean(np.array([[val[k] for k in keys] for val in val.values()]), 0)))
            else:
                max_, min_, med_, mean_ = np.max(y_vals), np.min(y_vals), np.median(y_vals), np.mean(y_vals)
                argmax_, argmin_ = np.argmax(y_vals), np.argmin(y_vals)
                plt.title('max: {:.8f} at iter {} \nmin: {:.8f} at iter {} \nmedian: {:.8f} '
                          '\nmean: {:.8f}'.format(max_, x_vals[argmax_], min_, x_vals[argmin_], med_, mean_))

                plt.plot(x_vals, y_vals)
                prints.append("{}\t{:.6f}".format(name, np.mean(np.array(list(val.values())), 0)))

            fig.savefig(os.path.join(self.plot_folder, name.replace(' ', '_') + '.jpg'))
            if self.use_visdom:
                self.vis.matplot(fig, win=name)
            fig.clear()

        # save recorded images
        for name, val in list(_img_since_last_flush.items()):
            latest = self._options[name].get('last_only')

            for itt, val in val.items():
                if val.dtype != 'uint8':
                    val = (255.99 * val).astype('uint8')

                if len(val.shape) == 4:
                    if self.use_visdom:
                        self.vis.images(val, win=name)

                    for num in range(val.shape[0]):
                        img = val[num]
                        if img.shape[0] == 3:
                            img = np.transpose(img, (1, 2, 0))

                            if latest:
                                imwrite(os.path.join(self.image_folder,
                                                     name.replace(' ', '_') + '_%d.jpg' % num), img)
                            else:
                                imwrite(os.path.join(self.image_folder,
                                                     name.replace(' ', '_') + '_%d_%d.jpg' % (itt, num)), img)

                        else:
                            for ch in range(img.shape[0]):
                                img_normed = (img[ch] - np.min(img[ch])) / (np.max(img[ch]) - np.min(img[ch]))
                                # in case all image values are the same
                                img_normed[np.isnan(img_normed)] = 0
                                img_normed[np.isinf(img_normed)] = 0

                                if latest:
                                    imwrite(os.path.join(self.image_folder,
                                                         name.replace(' ', '_') + '_%d_%d.jpg' % (num, ch)), img_normed)
                                else:
                                    imwrite(os.path.join(self.image_folder,
                                                         name.replace(' ', '_') + '_%d_%d_%d.jpg' % (itt, num, ch)),
                                            img_normed)

                elif len(val.shape) == 3 or len(val.shape) == 2:
                    if self.use_visdom:
                        self.vis.image(val if len(val.shape) == 2 else np.transpose(val, (2, 0, 1)), win=name)

                    if latest:
                        imwrite(os.path.join(self.image_folder, name.replace(' ', '_') + '.jpg'), val)
                    else:
                        imwrite(os.path.join(self.image_folder, name.replace(' ', '_') + '_%d.jpg' % itt), val)

                else:
                    raise NotImplementedError

        # make histograms of recorded data
        for name, val in list(_hist_since_last_flush.items()):
            if self.use_tensorboard:
                k = max(list(_hist_since_last_flush[name].keys()))
                self.writer.add_histogram(name, np.array(val[k]).flatten(), global_step=k)

            n_bins = self._options[name].get('n_bins', 20)
            latest = self._options[name].get('last_only', False)

            if latest:
                k = max(list(_hist_since_last_flush[name].keys()))
                plt.hist(np.array(val[k]).flatten(), bins='auto')
            else:
                self._hist_since_beginning[name].update(val)

                z_vals = np.array(list(self._hist_since_beginning[name].keys()))
                vals = [np.array(self._hist_since_beginning[name][i]).flatten() for i in z_vals]
                hists = [np.histogram(v, bins=n_bins) for v in vals]
                y_vals = np.array([hists[i][0] for i in range(len(hists))])
                x_vals = np.array([hists[i][1] for i in range(len(hists))])
                x_vals = (x_vals[:, :-1] + x_vals[:, 1:]) / 2.
                z_vals = np.tile(z_vals[:, None], (1, n_bins))

                ax = fig.gca(projection='3d')
                surf = ax.plot_surface(x_vals, z_vals, y_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.view_init(45, -90)
                fig.colorbar(surf, shrink=0.5, aspect=5)
            fig.savefig(os.path.join(self.hist_folder, name.replace(' ', '_') + '_hist.jpg'))
            fig.clear()

        # scatter pointcloud(s)
        for name, vals in list(_pointcloud_since_last_flush.items()):
            latest = self._options[name].get('last_only')
            for itt, val in vals.items():
                if len(val.shape) == 2:
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(*[val[:, i] for i in range(val.shape[-1])])

                    if latest:
                        plt.savefig(os.path.join(self.plot_folder, name.replace(' ', '_') + '.jpg'))
                    else:
                        plt.savefig(os.path.join(self.plot_folder, name.replace(' ', '_') + '_%d.jpg' % itt))

                elif len(val.shape) == 3:
                    for ii in range(val.shape[0]):
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(*[val[ii, :, i] for i in range(val.shape[-1])])
                        if latest:
                            plt.savefig(
                                os.path.join(self.plot_folder, name.replace(' ', '_') + '_%d.jpg' % (ii + 1)))
                        else:
                            plt.savefig(os.path.join(self.plot_folder,
                                                     name.replace(' ', '_') + '_%d_%d.jpg' % (itt, ii + 1)))

                        fig.clear()
                else:
                    raise NotImplementedError(
                        'Point cloud tensor must have 2 or 3 dimensions, got %d.' % len(val.shape))

                fig.clear()
            fig.clear()
        plt.close('all')

        with open(os.path.join(self.file_folder, 'log.pkl'), 'wb') as f:
            pkl.dump({'iter': it, 'epoch': epoch, 'num_iters': self.num_iters,
                      'num': dict(self._num_since_beginning),
                      'hist': dict(self._hist_since_beginning),
                      'options': dict(self._options)}, f, pkl.HIGHEST_PROTOCOL)
            f.close()

        iter_show = 'Iteration {}/{} ({:.2f}%) Epoch {}'.format(it % self.num_iters, self.num_iters,
                                                                (it % self.num_iters) / self.num_iters * 100.,
                                                                epoch + 1) if self.num_iters \
            else 'Iteration {} Epoch {}'.format(it, epoch)

        elapsed_time = time.time() - self._timer
        time_unit = 'mins' if elapsed_time < 3600. else 'hrs'
        elapsed_time = '{:.2f}'.format(elapsed_time / 60. if elapsed_time < 3600.
                                       else elapsed_time / 3600.) + time_unit
        now = dt.now().strftime("%d/%m/%Y %H:%M:%S")
        log = '{} Elapsed time {}\t{}\t{}'.format(now, elapsed_time, iter_show, '\t'.join(prints))
        print(log)

        if self.send_slack:
            message = 'From %s ' % self.current_folder
            message += log
            utils.slack_message(message=message, **self.kwargs)

    def _work(self):
        while True:
            items = self._q.get()
            work = items[0]
            work(*items[1:])
            self._q.task_done()

    def flush(self):
        """
        executes all the scheduled plots.
        Do not call this if using :class:`Monitor`'s context manager mode.

        :return: ``None``.
        """

        self._q.put((self._worker, self.iter, self.epoch, dict(self._num_since_last_flush),
                     dict(self._img_since_last_flush), dict(self._hist_since_last_flush),
                     dict(self._points_since_last_flush)))
        self._num_since_last_flush.clear()
        self._img_since_last_flush.clear()
        self._hist_since_last_flush.clear()
        self._points_since_last_flush.clear()

    def _version(self, file, keep):
        name, ext = os.path.splitext(file)
        versioned_filename = os.path.normpath(name + '-%d' % self.iter + ext)

        if file not in self._dump_files.keys():
            self._dump_files[file] = []

        if versioned_filename not in self._dump_files[file]:
            self._dump_files[file].append(versioned_filename)

        if len(self._dump_files[file]) > keep:
            oldest_file = self._dump_files[file][0]
            full_file = os.path.join(self.current_folder, oldest_file)
            if os.path.exists(full_file):
                os.remove(full_file)
            else:
                print("The oldest saved file does not exist")
            self._dump_files[file].remove(oldest_file)

        with open(os.path.join(self.current_folder, '_version.pkl'), 'wb') as f:
            pkl.dump(self._dump_files, f, pkl.HIGHEST_PROTOCOL)
        return versioned_filename

    def dump(self, name, obj, method='pickle', keep=-1, **kwargs):
        """
        saves the given object.

        :param name:
            name of the file to be saved.
        :param obj:
            object to be saved.
        :param method:
            ``str`` or ``callable``.
            If ``callable``, it should be a custom method to dump object.
            There are 3 types of ``str``.

            ```pickle```: use :func:`pickle.dump` to store object.

            ```torch```: use :func:`torch.save` to store object.

            ```txt```: use :func:`numpy.savetxt` to store object.

            Default: ```pickle```.
        :param keep:
            the number of versions of the saved file to keep.
            Default: -1 (keeps only the latest version).
        :param kwargs:
            additional keyword arguments to the underlying save function.
        :return: ``None``.
        """
        assert callable(method) or isinstance(method, str), 'method must be a string or callable'
        if isinstance(method, str):
            assert method in ('pickle', 'torch', 'txt'), 'method must be one of \'pickle\', \'torch\', or \'txt\''

        method = method if callable(method) else self._io_method[method + '_save']
        self._dump(name.replace(' ', '_'), obj, keep, method, **kwargs)

    def load(self, file, method='pickle', version=-1, **kwargs):
        """
        loads from the given file.

        :param file:
            name of the saved file without version.
        :param method:
            ``str`` or ``callable``.
            If ``callable``, it should be a custom method to load object.
            There are 3 types of ``str``.

            ```pickle```: use :func:`pickle.dump` to store object.

            ```torch```: use :func:`torch.save` to store object.

            ```txt```: use :func:`numpy.savetxt` to store object.

            Default: ```pickle```.
        :param version:
            the version of the saved file to load.
            Default: -1 (loads the latest version of the saved file).
        :param kwargs:
            additional keyword arguments to the underlying load function.
        :return: ``None``.
        """
        assert callable(method) or isinstance(method, str), 'method must be a string or callable'
        if isinstance(method, str):
            assert method in ('pickle', 'torch', 'txt'), 'method must be one of \'pickle\', \'torch\', or \'txt\''

        method = method if callable(method) else self._io_method[method + '_load']
        return self._load(file, method, version, **kwargs)

    def _dump(self, name, obj, keep, method, **kwargs):
        assert isinstance(keep, int), 'keep must be an int, got %s' % type(keep)

        if keep < 2:
            name = os.path.join(self.current_folder, name)
            method(name, obj, **kwargs)
            print('Object dumped to %s' % name)
        else:
            normed_name = self._version(name, keep)
            normed_name = os.path.join(self.current_folder, normed_name)
            method(normed_name, obj, **kwargs)
            print('Object dumped to %s' % normed_name)

    def _load(self, file, method, version=-1, **kwargs):
        assert isinstance(version, int), 'keep must be an int, got %s' % type(version)

        full_file = os.path.join(self.current_folder, file)
        try:
            with open(os.path.join(self.current_folder, '_version.pkl'), 'rb') as f:
                self._dump_files = pkl.load(f)

            versions = self._dump_files.get(file, [])
            if len(versions) == 0:
                try:
                    obj = method(full_file, **kwargs)
                except FileNotFoundError:
                    print('No file named %s found' % file)
                    return None
            else:
                if version <= 0:
                    if len(versions) > 0:
                        latest = versions[-1]
                        obj = method(os.path.join(self.current_folder, latest), **kwargs)
                    else:
                        return method(full_file, **kwargs)
                else:
                    name, ext = os.path.splitext(file)
                    file_name = os.path.normpath(name + '-%d' % version + ext)
                    if file_name in versions:
                        obj = method(os.path.join(self.current_folder, file_name), **kwargs)
                    else:
                        print('Version %d of %s is not found in %s' % (version, file, self.current_folder))
                        return None
        except FileNotFoundError:
            try:
                obj = method(full_file, **kwargs)
            except FileNotFoundError:
                print('No file named %s found' % file)
                return None

        text = str(version) if version > 0 else 'latest'
        print('Version \'%s\' loaded' % text)
        return obj

    def _save_pickle(self, name, obj):
        with open(name, 'wb') as f:
            pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
            f.close()

    def _load_pickle(self, name):
        with open(name, 'rb') as f:
            obj = pkl.load(f)
            f.close()
        return obj

    def _save_txt(self, name, obj, **kwargs):
        np.savetxt(name, obj, **kwargs)

    def _load_txt(self, name, **kwargs):
        return np.loadtxt(name, **kwargs)

    def _save_torch(self, name, obj, **kwargs):
        T.save(obj, name, **kwargs)

    def _load_torch(self, name, **kwargs):
        return T.load(name, **kwargs)

    def reset(self):
        """
        factory-resets the monitor object.
        This includes clearing all the collected data,
        set the iteration and epoch counters to 0,
        and reset the timer.

        :return: ``None``.
        """

        del self.num_stats
        del self.hist_stats
        del self.options
        self._num_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._img_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._hist_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._points_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._dump_files = collections.OrderedDict()
        self._iter = 0
        self._last_epoch = 0
        self.num_iters = self._num_iters
        self._timer = time.time()

    def read_log(self, log):
        """
        reads a saved log file.

        :param log:
            name of the log file.
        :return:
            contents of the log file.
        """

        with open(os.path.join(self.current_folder, 'files', log), 'rb') as f:
            f.seek(0)
            try:
                contents = pkl.load(f)
            except EOFError:
                contents = {}

            f.close()
        return contents
