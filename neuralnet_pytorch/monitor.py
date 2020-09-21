from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
import matplotlib.pyplot as plt
import threading
import queue
import numpy as np
import collections
import pickle as pkl
import os
import time
import torch as T
import torch.nn as nn
import atexit
import logging
from matplotlib import cm
from imageio import imwrite
from shutil import copyfile, copytree, ignore_patterns

try:
    import visdom
except ImportError:
    visdom = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # for Pytorch earlier than 1.1.0
    from tensorboardX import SummaryWriter

from . import layers
from . import utils
from .utils import root_logger, log_formatter

matplotlib.use('Agg')
__all__ = ['Monitor', 'monitor', 'logger', 'track', 'get_tracked_variables', 'eval_tracked_variables', 'hooks']
_TRACKS = collections.OrderedDict()
hooks = {}
lock = utils.ReadWriteLock()

# setup logger
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(log_formatter)
root_logger.addHandler(consoleHandler)
logger = root_logger


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

        ``'forward'``: tracks only value.

        ``'backward'``: tracks only gradient.

        ``'all'``: tracks both value and gradient.

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


def check_path_init(f):
    def set_default_path(self, *args, **kwargs):
        if not self._initialized:
            logger.info('Working folder not initialized! Initialize working folder to default.')
            self.set_path()
            self._initialized = True

        return f(self, *args, **kwargs)

    return set_default_path


def standardize_name(f):
    def func(self, name: str, *args, **kwargs):
        name = name.replace(' ', '-')
        f(self, name, *args, **kwargs)

    return func


class Monitor:
    """
    Collects statistics and displays the results using various backends.
    The collected stats are stored in '<root>/<model_name>/<prefix><#id>'
    where #id is automatically assigned each time a new run starts.

    Examples
    --------
    The following snippet shows how to plot smoothed training losses and
    save images from the current iteration, and then display them every 100 iterations.

    .. code-block:: python

        from neuralnet_pytorch import monitor as mon

        mon.model_name = 'foo-model'
        mon.set_path()
        mon.print_freq = 100

        ...
        for epoch in mon.iter_epoch(range(n_epochs)):
            for data in mon.iter_batch(data_loader):
                loss = net(data)
                mon.plot('training loss', loss, smooth=.99, filter_outliers=True)
                mon.imwrite('input images', data['images'], latest_only=True)
        ...

    Parameters
    ----------
    model_name : str
        name of the model folder.
        Default: ``None``.
    root : str
        path to store the collected statistics.
        Default: ``None``.
    current_folder : str
        if given, all the stats will be loaded from the given folder.
        Default: ``None``.
    print_freq : int
        statistics display frequency.
        Unit is iteration.
        Default: ``None``.
    num_iters : int
        number of iterations per epoch.
        If specified, training iteration percentage will be displayed along with epoch.
        Otherwise, it will be automatically calculated in the first epoch.
        Default: 100.
    prefix : str
        predix for folder name of of each run.
        Default: ``'run'``.
    use_visdom : bool
        whether to use Visdom for real-time monitoring.
        Default: ``False``.
    use_tensorboard : bool
        whether to use Tensorboard for real-time monitoring.
        Default: ``False``.
    send_slack : bool
        whether to send the statistics to Slack chatroom.
        Default: ``False``.
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
    _initialized = False
    _begin_epoch_ = 'begin_epoch'
    _end_epoch_ = 'end_epoch'
    _begin_iter_ = 'begin_iter'
    _end_iter_ = 'end_iter'

    def __init__(self, model_name=None, root=None, current_folder=None, print_freq=100, num_iters=None,
                 prefix='run', use_visdom=False, use_tensorboard=False, send_slack=False, **kwargs):
        self._iter = 0
        self._last_epoch = 0
        self._num_since_beginning = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._num_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._hist_since_beginning = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._hist_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._mat_since_beginning = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._mat_since_last_flush = {}
        self._img_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._points_since_last_flush = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._options = collections.defaultdict(_spawn_defaultdict_ordereddict)
        self._dump_files = collections.OrderedDict()
        self._schedule = {
            self._begin_epoch_: collections.defaultdict(_spawn_defaultdict_ordereddict),
            self._end_epoch_: collections.defaultdict(_spawn_defaultdict_ordereddict),
            self._begin_iter_: collections.defaultdict(_spawn_defaultdict_ordereddict),
            self._end_iter_: collections.defaultdict(_spawn_defaultdict_ordereddict)
        }
        self._timer = time.time()
        self._io_method = {'pickle_save': self._save_pickle, 'txt_save': self._save_txt,
                           'torch_save': self._save_torch, 'pickle_load': self._load_pickle,
                           'txt_load': self._load_txt, 'torch_load': self._load_torch}

        self.model_name = model_name
        self.root = root
        self._prefix = prefix
        self._num_iters = num_iters
        self.print_freq = print_freq
        self.num_iters = num_iters
        self.use_tensorboard = use_tensorboard
        self.use_visdom = use_visdom
        self.current_folder = current_folder
        self.kwargs = kwargs
        self.plot_folder = None
        self.file_folder = None
        self.image_folder = None
        self.hist_folder = None
        self.current_run = None
        self.writer = None
        if current_folder is not None or model_name is not None:
            self.set_path(current_folder)

        self.vis = None
        if use_visdom and visdom is not None:
            self.init_visdom()

        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._flush, daemon=True)
        self._thread.start()

        self.send_slack = send_slack
        if send_slack:
            self.init_slack()

        # schedule to flush when the program finishes
        atexit.register(self._atexit)

    def __setattr__(self, attr, val):
        if self._initialized and attr in ('model_name', 'root', 'current_folder', 'plot_folder', 'file_folder',
                                          'image_folder', 'hist_folder', 'current_run'):
            raise ValueError('{} attribute must not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super().__setattr__(attr, val)

    def set_path(self, path=None):
        if path is None:
            root = 'results' if self.root is None else self.root
            model_name = 'my-model' if self.model_name is None else self.model_name
            path = os.path.join(root, model_name)
            os.makedirs(path, exist_ok=True)
            path = self._get_new_folder(path)

        self.current_folder = os.path.normpath(path)
        if os.path.exists(self.current_folder):
            lock.acquire_read()
            self.load_state()
            lock.release_read()
        else:
            os.makedirs(self.current_folder, exist_ok=True)

        # make folders to store statistics
        self.plot_folder = os.path.join(self.current_folder, 'plots')
        os.makedirs(self.plot_folder, exist_ok=True)

        self.file_folder = os.path.join(self.current_folder, 'files')
        os.makedirs(self.file_folder, exist_ok=True)

        self.image_folder = os.path.join(self.current_folder, 'images')
        os.makedirs(self.image_folder, exist_ok=True)

        self.hist_folder = os.path.join(self.current_folder, 'histograms')
        os.makedirs(self.hist_folder, exist_ok=True)

        file_handler = logging.FileHandler('{0}/{1}.log'.format(self.file_folder, 'history'))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info('Result folder: %s' % self.current_folder)
        self._initialized = True

        if self.use_tensorboard:
            self.init_tensorboard()

    def load_state(self):
        self.current_run = os.path.basename(self.current_folder)

        try:
            log = self.read_log('log.pkl')
            try:
                self.num_stats = log['num']
            except KeyError:
                root_logger.warning('No record found for `num`', exc_info=True)

            try:
                self.num_stats = log['mat']
            except KeyError:
                root_logger.warning('No record found for `mat`', exc_info=True)

            try:
                self.hist_stats = log['hist']
            except KeyError:
                root_logger.warning('No record found for `hist`', exc_info=True)

            if self.num_iters is None:
                try:
                    self.num_iters = log['num_iters']
                except KeyError:
                    root_logger.warning('No record found for `num_iters`', exc_info=True)

            try:
                self.iter = log['iter']
            except KeyError:
                root_logger.warning('No record found for `iter`', exc_info=True)

            try:
                self.epoch = log['epoch']
            except KeyError:
                if self.num_iters:
                    self.epoch = self.iter // self.num_iters
                else:
                    root_logger.warning('No record found for `epoch`', exc_info=True)

        except FileNotFoundError:
            root_logger.warning('`log.pkl` not found in `%s`' % os.path.join(self.current_folder, 'files'),
                                exc_info=True)

    def _get_new_folder(self, path):
        runs = [folder for folder in os.listdir(path) if folder.startswith(self._prefix)]
        if not runs:
            idx = 1
        else:
            indices = sorted([int(r[len(self._prefix):]) for r in runs])
            idx = indices[-1] + 1

        self.current_run = '{}{}'.format(self._prefix, idx)
        return os.path.join(path, self.current_run)

    def init_tensorboard(self):
        assert self._initialized, 'Working folder must be set by set_path first.'
        os.makedirs(os.path.join(self.current_folder, 'tensorboard'), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.current_folder, 'tensorboard'))
        self.use_tensorboard = True

    def init_visdom(self):
        server = self.kwargs.pop('server', 'http://localhost')
        port = self.kwargs.pop('port', 8097)
        self.vis = visdom.Visdom(server=server, port=port)
        if not self.vis.check_connection():
            from subprocess import Popen, PIPE
            Popen('visdom', stdout=PIPE, stderr=PIPE)

        self.vis.close()
        print('You can navigate to \'%s:%d\' for visualization' % (server, port))
        self.use_visdom = True

    def init_slack(self):
        assert self.kwargs.get('channel', None) is not None and self.kwargs.get('token', None) is not None, \
            'channel and token must be provided to send a slack message'

        if self.kwargs.get('username', None) is None:
            self.kwargs['username'] = 'me'

        self.send_slack = True

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

        >>> from neuralnet_pytorch import monitor as mon
        >>> mon.print_freq = 1000
        >>> num_epochs = 10
        >>> for epoch in mon.iter_epoch(range(mon.epoch, num_epochs))
        ...     # do something here

        See Also
        --------
        :meth:`~iter_batch`
        """

        if self.num_iters:
            self.iter = self.epoch * self.num_iters

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

        >>> from neuralnet_pytorch import monitor as mon
        >>> mon.print_freq = 1000
        >>> data_loader = ...
        >>> num_epochs = 10
        >>> for epoch in mon.iter_epoch(range(num_epochs)):
        ...     for idx, data in mon.iter_batch(enumerate(data_loader)):
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
        self.iter += 1
        if self.num_iters:
            self.epoch = self.iter // self.num_iters

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
        if self.num_iters:
            self.iter = self.epoch * self.num_iters

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
    def mat_stats(self):
        """
        returns the collected scalar statistics from beginning.

        :return:
            :attr:`~_num_since_beginning`.
        """

        return dict(self._mat_since_beginning)

    @mat_stats.setter
    def mat_stats(self, stats_dict):
        self._mat_since_beginning.update(stats_dict)

    @mat_stats.deleter
    def mat_stats(self):
        self._mat_since_beginning.clear()

    def clear_mat_stats(self, key):
        """
        removes the collected statistics for matrix plot of the specified `key`.

        :param key:
            the name of the matrix collection.
        :return: ``None``.
        """

        self._mat_since_beginning[key].clear()

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

    @standardize_name
    def set_option(self, name, option, value):
        """
        sets option for histogram plotting.

        :param name:
             name of the histogram plot.
             Must be the same as the one specified when using :meth:`~hist`.
        :param option:
            there are two options which should be passed as a ``str``.

            ``'latest_only'``: plot the histogram of the last recorded tensor only.

            ``'n_bins'``: number of bins of the histogram.
        :param value:
            value of the chosen option. Should be ``True``/``False`` for ``'latest_only'``
            and an integer for ``'n_bins'``.
        :return: ``None``.
        """

        self._options[name][option] = value

    def run_training(self, net, solver: T.optim.Optimizer, train_loader, n_epochs: int, closure=None, eval_loader=None,
                     valid_freq=None, start_epoch=None, scheduler=None, scheduler_iter=False, device=None, *args,
                     **kwargs):
        """
        Runs the training loop for the given neural network.

        :param net:
            must be an instance of :class:`~neuralnet_pytorch.layers.Net`
            and :class:`~neuralnet_pytorch.layers.Module`.
        :param solver:
            a solver for optimization.
        :param train_loader:
            provides training data for neural net.
        :param n_epochs:
            number of training epochs.
        :param closure:
            a method to calculate loss in each optimization step. Optional.
        :param eval_loader:
            provides validation data for neural net. Optional.
        :param valid_freq:
            indicates how often validation is run.
            In effect if only `eval_loader` is given.
        :param start_epoch:
            the epoch from which training will continue.
            If ``None``, training counter will be set to 0.
        :param scheduler:
            a learning rate scheduler.
            Default: ``None``.
        :param scheduler_iter:
            if ``True``, `scheduler` will run every iteration.
            Otherwise, it will step every epoch.
            Default: ``False``.
        :param device:
            device to perform calculation.
            Default: ``None``.
        :param args:
            additional arguments that will be passed to neural net.
        :param kwargs:
            additional keyword arguments that will be passed to neural net.
        :return: ``None``.

        Examples
        --------

        .. code-block:: python

            import neuralnet_pytorch as nnt
            from neuralnet_pytorch import monitor as mon

            class MyNet(nnt.Net, nnt.Module):
                ...


                def train_procedure(batch, *args, **kwargs):
                    loss = ...
                    mon.plot('train loss', loss)
                    return loss

                def eval_procedure(batch, *args, **kwargs):
                    pred = ...
                    loss = ...
                    acc = ...
                    mon.plot('eval loss', loss)
                    mon.plot('eval accuracy', acc)

            # define the network, and training and testing loaders
            net = MyNet(...)
            train_loader = ...
            eval_loader = ...
            solver = ...
            scheduler = ...

            # instantiate a Monitor object
            mon.model_name = 'my_net'
            mon.print_freq = 100
            mon.set_path()

            # collect the parameters of the network
            def save_checkpoint():
                states = {
                    'states': mon.epoch,
                    'model_state_dict': net.state_dict(),
                    'opt_state_dict': solver.state_dict()
                }
                if scheduler is not None:
                    states['scheduler_state_dict'] = scheduler.state_dict()

                mon.dump(name='training.pt', obj=states, type='torch', keep=5)

            # save a checkpoint after each epoch and keep only the 5 latest checkpoints
            mon.schedule(save_checkpoint)
            print('Training...')

            # run the training loop
            mon.run_training(net, solver, train_loader, n_epochs, eval_loader=eval_loader, scheduler=scheduler,
                             valid_freq=val_freq)
            print('Training finished!')

        Parameters
        ----------
        solver
        scheduler
        scheduler
        """

        assert isinstance(net, (layers.Module, nn.Module, layers.Sequential, nn.Sequential)), \
            '`net` must be an instance of `Module` or `Sequential`'
        assert hasattr(net, 'train_procedure'), '`train_procedure` method must be defined for `net`'

        net = net.to(device)
        start_epoch = self.epoch if start_epoch is None else start_epoch
        for _ in self.iter_epoch(range(start_epoch, n_epochs)):
            for func_dict in self._schedule[self._begin_epoch_].values():
                func_dict['func'](*func_dict['args'], **func_dict['kwargs'])

            for it, batch in self.iter_batch(enumerate(train_loader)):
                for func_dict in self._schedule[self._begin_iter_].values():
                    func_dict['func'](*func_dict['args'], **func_dict['kwargs'])

                net.train(True)
                batch = utils.batch_to_device(batch, device=device)
                solver.zero_grad()
                loss = net.train_procedure(batch, *args, **kwargs)
                if not (T.isnan(loss) or T.isinf(loss)):
                    loss.backward()
                else:
                    raise ValueError('NaN or Inf encountered. Training failed!')

                solver.step(closure)
                if scheduler is not None and scheduler_iter:
                    scheduler.step()

                for func_dict in self._schedule[self._end_iter_].values():
                    func_dict['func'](*func_dict['args'], **func_dict['kwargs'])

                if valid_freq and hasattr(net, 'evaluate'):
                    if self.iter % valid_freq == 0:
                        net.eval()

                        with T.set_grad_enabled(False):
                            for itt, batch in enumerate(eval_loader):
                                batch = utils.batch_to_device(batch, device=device)
                                try:
                                    net.eval_procedure(batch, *args, **kwargs)
                                except NotImplementedError:
                                    root_logger.exception('An evaluation procedure must be specified')
                                    raise

            if scheduler is not None and not scheduler_iter:
                scheduler.step()

            for func_dict in self._schedule[self._end_epoch_].values():
                func_dict['func'](*func_dict['args'], **func_dict['kwargs'])

    def _atexit(self):
        if self._initialized:
            self.flush()
            plt.close()
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

            self._q.join()

    @check_path_init
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

    @check_path_init
    def dump_model(self, network, use_tensorboard=False, *args, **kwargs):
        """
        saves a string representation of the given neural net.

        :param network:
            neural net to be saved as string representation.
        :param use_tensorboard:
            use tensorboard to save `network`'s graph.
        :param args:
            additional arguments to Tensorboard's :meth:`SummaryWriter`
            when `use_tensorboard` is ``True``.
        :param kwargs:
            additional keyword arguments to Tensorboard's :meth:`SummaryWriter`
            when `~se_tensorboard` is ``True``.
        :return: ``None``.
        """

        assert isinstance(network, (
            nn.Module, nn.Sequential)), 'network must be an instance of Module or Sequential, got {}'.format(
            type(network))
        self.dump_rep('network.txt', network)
        if use_tensorboard:
            self.writer.add_graph(network, *args, **kwargs)

    @check_path_init
    def backup(self, files_or_folders, ignore=None):
        """
        saves a copy of the given files to :attr:`~current_folder`.
        Accepts a str or list/tuple of file or folder names.
        You can backup your codes and/or config files for later use.

        :param files_or_folders:
            file to be saved.
        :param ignore:
            files or patterns to ignore.
            Default: ``None``.
        :return: ``None``.
        """
        assert isinstance(files_or_folders, (str, list, tuple)), \
            'unknown type of \'files_or_folders\'. Expect list, tuple or string, got {}'.format(type(files_or_folders))

        files_or_folders = (files_or_folders,) if isinstance(files_or_folders, str) else files_or_folders
        if ignore is None:
            ignore = ()

        # filter ignored files
        import fnmatch
        to_backup = []
        for f in files_or_folders:
            if not any(fnmatch.fnmatch(f, p) for p in ignore):
                to_backup.append(f)

        for f in to_backup:
            try:
                if os.path.isfile(f):
                    copyfile(f, '%s/%s' % (self.file_folder, os.path.split(f)[-1]))
                elif os.path.isdir(f):
                    copytree(f, '%s/%s' % (self.file_folder, os.path.split(f)[-1]))
            except FileNotFoundError:
                root_logger.warning('No such file or directory: %s' % f)

    @utils.deprecated(backup, '1.1.0')
    def copy_files(self, files):
        self.backup(files)

    @standardize_name
    def plot(self, name: str, value, smooth=0, filter_outliers=True, **kwargs):
        """
        schedules a plot of scalar value.
        A :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            scalar value to be plotted.
        :param smooth:
            a value between ``0`` and ``1`` to define the smoothing window size.
            See :func:`~neuralnet_pytorch.utils.numpy_utils.smooth`.
            Default: ``0``.
        :param filter_outliers:
            whether to filter out outliers in plot.
            This affects only the plot and not the raw statistics.
            Default: True.
        :param kwargs:
            additional options to tensorboard.
        :return: ``None``.
        """

        self._options[name]['smooth'] = smooth
        self._options[name]['filter_outliers'] = filter_outliers
        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._num_since_last_flush[name][self.iter] = value
        if self.writer is not None:
            prefix = kwargs.pop('prefix', 'scalar/')
            self.writer.add_scalar(prefix + name.replace(' ', '-'), value, global_step=self.iter, **kwargs)

    @standardize_name
    def plot_matrix(self, name: str, value, labels=None, show_values=False):
        """
        plots the given matrix with colorbar and labels if provided.
        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            matrix value to be plotted.
        :param labels:
            labels of each axis.
            Can be a list/tuple of strings or a nested list/tuple.
            Defaults: None.
        :return: ``None``.
        """

        self._options[name]['labels'] = labels
        self._options[name]['show_values'] = show_values
        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._mat_since_last_flush[name] = value
        self._mat_since_beginning[name][self.iter] = value

    @standardize_name
    def scatter(self, name: str, value, latest_only=False, **kwargs):
        """
        schedules a scattor plot of (a batch of) points.
        A 3D :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            2D or 3D tensor to be plotted. The last dim should be 3.
        :param latest_only:
            whether to save only the latest statistics or keep everything from beginning.
        :param kwargs:
            additional options to tensorboard.
        :return: ``None``.
        """

        self._options[name]['latest_only'] = latest_only
        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        if len(value.shape) == 2:
            value = value[None]

        self._points_since_last_flush[name][self.iter] = value
        if self.writer is not None:
            self.writer.add_mesh(name, value, global_step=self.iter, **kwargs)

    @standardize_name
    def imwrite(self, name: str, value, latest_only=False, **kwargs):
        """
        schedules to save images.
        The images will be rendered and saved every :attr:`~print_freq` iterations.
        There are some assumptions about input data:

        - If the input is ``'uint8'`` it is an 8-bit image.
        - If the input is ``'float32'``, its values lie between ``0`` and ``1``.
        - If the input has 3 dims, the shape is ``[h, w, 3]`` or ``[h, w, 1]``.
        - If the channel dim is different from 3 or 1, it will be considered as multiple gray images.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            2D, 3D or 4D tensor to be plotted.
            The expected shape is ``(H, W)`` for 2D tensor, ``(H, W, C)`` for 3D tensor and
            ``(N, C, H, W)`` for 4D tensor.
            If the number of channels is other than 3 or 1, each channel is saved as
            a gray image.
        :param latest_only:
            whether to save only the latest statistics or keep everything from beginning.
        :param kwargs:
            additional options to tensorboard.
        :return: ``None``.
        """

        self._options[name]['latest_only'] = latest_only
        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        if value.dtype != 'uint8':
            value = (255.99 * value).astype('uint8')

        if len(value.shape) == 3:
            value = np.transpose(value, (2, 0, 1))[None]
        elif len(value.shape) == 2:
            value = value[None, None]

        self._img_since_last_flush[name][self.iter] = value
        if self.writer is not None:
            prefix = kwargs.pop('prefix', 'image/')
            self.writer.add_images(prefix + name.replace(' ', '-'), value,
                                   global_step=self.iter, dataformats='NCHW')

    @standardize_name
    def hist(self, name, value, n_bins=20, latest_only=False, **kwargs):
        """
        schedules a histogram plot of (a batch of) points.
        A :mod:`matplotlib` figure will be rendered and saved every :attr:`~print_freq` iterations.

        :param name:
            name of the figure to be saved. Must be unique among plots.
        :param value:
            any-dim tensor to be histogrammed.
        :param n_bins:
            number of bins of the histogram.
        :param latest_only:
            whether to save only the latest statistics or keep everything from beginning.
        :param kwargs:
            additional options to tensorboard
        :return: ``None``.
        """

        self._options[name]['latest_only'] = latest_only
        self._options[name]['n_bins'] = n_bins
        if isinstance(value, T.Tensor):
            value = utils.to_numpy(value)

        self._hist_since_last_flush[name][self.iter] = value
        if self.writer is not None:
            prefix = kwargs.pop('prefix', 'hist/')
            self.writer.add_histogram(prefix + name.replace(' ', '-'), value, global_step=self.iter, **kwargs)

    def schedule(self, func, when=None, *args, **kwargs):
        """
        uses to schedule a routine during every epoch in :meth:`~run_training`.

        :param func:
            a routine to be executed in :meth:`~run_training`.
        :param when:
            the moment when the ``func`` is executed.
            For the moment, choices are:
            ``'begin_epoch'``, ``'end_epoch'``, ``'begin_iter'``, and ``'end_iter'``.
            Default: ``'begin_epoch'``.
        :param args:
            additional arguments to `func`.
        :param kwargs:
            additional keyword arguments to `func`.
        :return: ``None``
        """

        assert callable(func), 'func must be callable'
        name = func.__name__
        if when is None:
            when = self._begin_epoch_

        self._schedule[when][name]['func'] = func
        self._schedule[when][name]['args'] = args
        self._schedule[when][name]['kwargs'] = kwargs

    def _plot(self, nums, prints):
        fig = plt.figure()
        plt.xlabel('iteration')
        for name, val in list(nums.items()):
            smooth = self._options[name].get('smooth')
            filter_outliers = self._options[name].get('filter_outliers')

            self._num_since_beginning[name].update(val)
            plt.ylabel(name)
            x_vals = sorted(self._num_since_beginning[name].keys())
            y_vals = [self._num_since_beginning[name][x] for x in x_vals]
            max_, min_, med_, mean_ = np.max(y_vals), np.min(y_vals), np.median(y_vals), np.mean(y_vals)
            argmax_, argmin_ = np.argmax(y_vals), np.argmin(y_vals)
            plt.title('max: {:.8f} at iter {} min: {:.8f} at iter {} \nmedian: {:.8f} mean: {:.8f}'
                      .format(max_, x_vals[argmax_], min_, x_vals[argmin_], med_, mean_))

            x_vals, y_vals = np.array(x_vals), np.array(y_vals)
            y_vals_smoothed = utils.smooth(y_vals, smooth)[:x_vals.shape[0]] if smooth else y_vals
            plt.plot(x_vals, y_vals_smoothed)
            if filter_outliers:
                inlier_indices = ~utils.is_outlier(y_vals)
                y_vals_filtered = y_vals[inlier_indices]
                min_, max_ = np.min(y_vals_filtered), np.max(y_vals_filtered)
                interval = (.9 ** np.sign(min_) * min_, 1.1 ** np.sign(max_) * max_)
                if not (np.any(np.isnan(interval)) or np.any(np.isinf(interval))):
                    plt.ylim(interval)

            prints.append("{}\t{:.6f}".format(name, np.mean(np.array(list(val.values())), 0)))
            fig.savefig(os.path.join(self.plot_folder, name.replace(' ', '_') + '.jpg'))
            if self.vis is not None:
                self.vis.matplot(fig, win=name)

            fig.clear()
        plt.close()

    def _plot_matrix(self, mats):
        fig = plt.figure()
        for name, val in list(mats.items()):
            ax = fig.add_subplot(111)
            im = ax.imshow(val)
            fig.colorbar(im)

            labels = self._options[name].get('labels')
            ax.set_xticks(np.arange(len(val)))
            ax.set_yticks(np.arange(len(val)))
            if labels is not None:
                if isinstance(labels[0], (list, tuple)):
                    ax.set_xticklabels(labels[0])
                    ax.set_yticklabels(labels[1])
                else:
                    ax.set_xticklabels(labels)
                    ax.set_yticklabels(labels)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            ax.set_ylim([-.5, len(val) - .5])

            show_values = self._options[name].get('show_values')
            if show_values:
                # Loop over data dimensions and create text annotations.
                for (i, j), z in np.ndenumerate(val):
                    ax.text(j, i, z, ha='center', va='center', color='w')

            ax.set_title(name)
            fig.savefig(os.path.join(self.plot_folder, name + '-matrix.jpg'), transparent=None)

            canvas = FigureCanvas(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            self.writer.add_image('matrix-' + name.replace(' ', '-'), img,
                                  global_step=self.iter, dataformats='HWC')
            fig.clear()
        plt.close()

    def _imwrite(self, imgs):
        for name, val in list(imgs.items()):
            latest_only = self._options[name].get('latest_only')

            for itt, val in val.items():
                if len(val.shape) == 4:
                    if self.vis is not None:
                        self.vis.images(val, win=name)

                    for num in range(val.shape[0]):
                        img = val[num]
                        if img.shape[0] in (1, 3):
                            img = np.transpose(img, (1, 2, 0))
                            if latest_only:
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
                                if latest_only:
                                    imwrite(os.path.join(
                                        self.image_folder,
                                        name.replace(' ', '_') + '_%d_%d.jpg' % (num, ch)), img_normed)
                                else:
                                    imwrite(os.path.join(
                                        self.image_folder,
                                        name.replace(' ', '_') + '_%d_%d_%d.jpg' % (itt, num, ch)), img_normed)
                else:
                    raise NotImplementedError

    def _hist(self, nums):
        fig = plt.figure()
        for name, val in list(nums.items()):
            n_bins = self._options[name].get('n_bins')
            latest_only = self._options[name].get('latest_only')
            if latest_only:
                k = max(list(nums[name].keys()))
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
        plt.close()

    def _scatter(self, points):
        fig = plt.figure()
        for name, vals in list(points.items()):
            latest_only = self._options[name].get('latest_only')
            for itt, val in vals.items():
                for ii, v in enumerate(val):
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(*[v[:, i] for i in range(v.shape[-1])])
                    if latest_only:
                        plt.savefig(
                            os.path.join(self.plot_folder, name.replace(' ', '_') + '_%d.jpg' % (ii + 1)))
                    else:
                        plt.savefig(os.path.join(self.plot_folder,
                                                 name.replace(' ', '_') + '_%d_%d.jpg' % (itt, ii + 1)))

                    fig.clear()
                fig.clear()
            fig.clear()
        plt.close()

    def _flush(self):
        while True:
            items = self._q.get()
            it, epoch, nums, mats, imgs, hists, points = items
            prints = []

            # plot statistics
            self._plot(nums, prints)

            # plot confusion matrix
            self._plot_matrix(mats)

            # save recorded images
            self._imwrite(imgs)

            # make histograms of recorded data
            self._hist(hists)

            # scatter point set(s)
            self._scatter(points)

            lock.acquire_write()
            with open(os.path.join(self.file_folder, 'log.pkl'), 'wb') as f:
                dump_dict = {'iter': it,
                             'epoch': epoch,
                             'num_iters': self.num_iters,
                             'num': dict(self._num_since_beginning),
                             'mat': dict(self._mat_since_beginning),
                             'hist': dict(self._hist_since_beginning)}

                pkl.dump(dump_dict, f, pkl.HIGHEST_PROTOCOL)
                f.close()
            lock.release_write()

            iter_show = 'Epoch {} Iteration {}/{} ({:.2f}%)'.format(
                epoch + 1, it % self.num_iters, self.num_iters,
                (it % self.num_iters) / self.num_iters * 100.) if self.num_iters \
                else 'Epoch {} Iteration {}'.format(epoch + 1, it)

            elapsed_time = time.time() - self._timer
            time_unit = 'mins' if elapsed_time < 3600. else 'hrs'
            elapsed_time = '{:.2f}'.format(elapsed_time / 60. if elapsed_time < 3600.
                                           else elapsed_time / 3600.) + time_unit
            log = 'Elapsed time {} {}\t{}\t{}'.format(elapsed_time, self.current_run, iter_show, '\t'.join(prints))
            root_logger.info(log)

            if self.send_slack:
                message = 'From %s ' % self.current_folder
                message += log
                utils.slack_message(message=message, **self.kwargs)

            self._q.task_done()

    @check_path_init
    def flush(self):
        """
        executes all the scheduled plots.
        Do not call this if using :class:`Monitor`'s context manager mode.

        :return: ``None``.
        """

        self._q.put((self.iter, self.epoch, dict(self._num_since_last_flush), dict(self._mat_since_last_flush),
                     dict(self._img_since_last_flush), dict(self._hist_since_last_flush),
                     dict(self._points_since_last_flush)))
        self._num_since_last_flush.clear()
        self._mat_since_last_flush.clear()
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
                root_logger.warning('The oldest saved file does not exist')
            self._dump_files[file].remove(oldest_file)

        with open(os.path.join(self.current_folder, '_version.pkl'), 'wb') as f:
            pkl.dump(self._dump_files, f, pkl.HIGHEST_PROTOCOL)
        return versioned_filename

    @check_path_init
    @standardize_name
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

            ``'pickle'``: use :func:`pickle.dump` to store object.

            ``'torch'``: use :func:`torch.save` to store object.

            ``'txt'``: use :func:`numpy.savetxt` to store object.

            Default: ``'pickle'``.
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

            ``'pickle'``: use :func:`pickle.dump` to store object.

            ``'torch'``: use :func:`torch.save` to store object.

            ``'txt'``: use :func:`numpy.savetxt` to store object.

            Default: ``'pickle'``.
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
            root_logger.info('Object dumped to %s' % name)
        else:
            normed_name = self._version(name, keep)
            normed_name = os.path.join(self.current_folder, normed_name)
            method(normed_name, obj, **kwargs)
            root_logger.info('Object dumped to %s' % normed_name)

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
                    root_logger.warning('No file named %s found' % file)
                    return None
            else:
                if isinstance(version, int) and version <= 0:
                    if len(versions) > 0:
                        version = versions[version]
                        obj = method(os.path.join(self.current_folder, version), **kwargs)
                    else:
                        return method(full_file, **kwargs)
                else:
                    name, ext = os.path.splitext(file)
                    file_name = os.path.normpath(name + '-%d' % version + ext)
                    if file_name in versions:
                        obj = method(os.path.join(self.current_folder, file_name), **kwargs)
                    else:
                        root_logger.warning(
                            'Version %d of %s is not found in %s' % (version, file, self.current_folder))
                        return None
        except FileNotFoundError:
            try:
                obj = method(full_file, **kwargs)
            except FileNotFoundError:
                root_logger.warning('No file named %s found' % file)
                return None

        root_logger.info('Version \'%s\' loaded' % str(version))
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
        self._mat_since_last_flush = {}
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


monitor = Monitor(use_tensorboard=True)
