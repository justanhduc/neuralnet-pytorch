import numpy as np
import torch as T
import threading
from queue import Queue
from scipy.stats import truncnorm

from . import root_logger

__all__ = ['DataLoader', 'DataPrefetcher', 'truncated_normal', 'batch_set_value', 'bulk_to_cuda', 'bulk_to_cuda_sparse',
           'bulk_to_numpy', 'to_cuda', 'to_cuda_sparse', 'to_numpy', 'batch_to_device', 'batch_to_cuda',
           'batch_set_tensor', 'ReadWriteLock']


class ReadWriteLock:
    """
    A lock object that allows many simultaneous `read locks`, but
    only one `write lock.`
    From https://www.oreilly.com/library/view/python-cookbook/0596001673/ch06s04.html.
    """

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        """
        Acquire a read lock. Blocks only if a thread has
        acquired the write lock.
        """

        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """
        Release a read lock.
        """

        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """
        Acquire a write lock. Blocks until there are no
        acquired read or write locks.
        """

        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """
        Release a write lock.
        """

        self._read_ready.release()


class ThreadsafeIter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    Parameters
    ----------
    it
        an iterable object.

    Attributes
    ----------
    lock
        a thread lock.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(generator):
    """
    A decorator that takes a generator function and makes it thread-safe.

    :param generator:
        a generator object.
    :return:
        a thread-safe generator.
    """

    def safe_generator(*agrs, **kwargs):
        return ThreadsafeIter(generator(*agrs, **kwargs))

    return safe_generator


class DataLoader:
    """
    A lightweight data loader. Works comparably to
    Pytorch's :class:`torch.utils.data.Dataloader`
    when the workload is light, but it initializes much faster.
    It is totally compatible with :class:`torch.utils.data.Dataset`
    and can be an alternative for :class:`torch.utils.data.Dataloader`.

    Parameters
    ----------
    dataset
        an instance of :class:`torch.utils.data.Dataset`.
    batch_size
        how many samples per batch to load.
        Default: 1.
    shuffle
        whether to shuffle in each iteration.
        Default: ``False``.
    sampler
        defines the strategy to draw samples from the dataset.
        If specified, :attr:`shuffle` must be ``False``.
    batch_sampler
        like :attr:`sampler`, but returns a batch of indices at a time.
        Mutually exclusive with :attr:`batch_size`,
        :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
    num_workers
        number of threads to be used.
    collate_fn
        function to specify how a batch is loaded.
    pin_memory
        if ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.  If your data elements
        are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type.
        Default: False.
    num_cached
        number of batches to be cached.
    drop_last
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller.
        Default: ``False``.
    kwargs
        arguments what will not be used.
        For compatibility with :class:`torch.utils.data.Dataloader` only.

    Attributes
    ----------
    batches
        contains batches of data when iterating over the dataset.
    """
    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, pin_memory=False, drop_last=False, num_cached=10, **kwargs):
        if len(kwargs.keys()) > 0:
            root_logger.warning(str(kwargs.keys()) +
                                ' are present for compatibility with '
                                '`torch.utils.data.DataLoader` interface'
                                ' and will be ignored.')

        if batch_sampler is not None and sampler is not None:
            root_logger.warning('sampler will not be used as batch_sampler is provided.')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            raise NotImplementedError('batch_size=None is currently not supported')

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_cached = num_cached
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.batches = None

        from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler
        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        if collate_fn is None:
            from torch.utils.data._utils.collate import default_collate, default_convert
            if self._auto_collation:
                collate_fn = default_collate
            else:
                collate_fn = default_convert

        self.collate_fn = collate_fn
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super().__setattr__(attr, val)

    def __iter__(self):
        self.batches = self._get_batches()
        return self

    def __next__(self):
        if self.batches is None:
            self.batches = self._get_batches()
        return self.batches.__next__()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self):
        return len(self._index_sampler)

    def _get_batches(self):
        batches = self._generator()
        batches = self._generate_in_background(batches)
        for it, batch in enumerate(batches):
            yield batch

    def _generate_in_background(self, generator):
        generator = ThreadsafeIter(generator)
        queue = Queue(maxsize=self.num_cached)

        # define producer (putting items into queue)
        def producer():
            for item in generator:
                queue.put(item)
            queue.join()
            queue.put(None)

        # start producer (in a background thread)
        for i in range(self.num_workers):
            thread = threading.Thread(target=producer, daemon=True)
            thread.start()

        # run as consumer (read items from queue, in current thread)
        while True:
            item = queue.get()
            if item is None:
                break
            yield item
            queue.task_done()

    def _getdata(self, index):
        batch = [self.dataset[i] for i in index]
        batch = self.collate_fn(batch)
        if self.pin_memory:
            batch = T.utils.data._utils.pin_memory.pin_memory(batch)

        return batch

    def _generator(self):
        assert isinstance(self.dataset[0],
                          (list, tuple, np.ndarray)), 'Dataset should consist of lists/tuples or Numpy ndarray'

        batch = None
        index = -1
        sampler_iter = iter(self._index_sampler)
        for index in sampler_iter:
            if not isinstance(index, (tuple, list)):
                index = [index]

            if len(index) == self.batch_size:
                batch = self._getdata(index)
                yield batch
            batch = None

        if batch is not None and not self.drop_last:
            batch = self._getdata(index)
            yield batch


class DataPrefetcher:
    """
    A prefetcher for :class:`torch.utils.data.Dataloader`.
    Adapted from https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py.

    Parameters
    ----------
    loader
        an instance of :class:`torch.utils.data.Dataloader`.
    transform
        a function to preprocess each batch in GPU.
        Default: ``None``.

    Examples
    --------
    This prefetcher can be added seamlessly to the current code pipeline.

    .. code-block:: python

        from torch.utils.data import DataLoader
        import neuralnet_pytorch as nnt

        ...
        loader = DataLoader(dataset, ...)
        loader = nnt.DataPrefetcher(loader)
        ...

    Attributes
    ----------
    next_data
        the prefetched data.
    stream
        an instance of :class:`torch.cuda.Stream`.
    """

    def __init__(self, loader, transform=None, device=T.device('cuda')):
        self._loader = loader
        self.loader = iter(loader)
        self.transform = transform
        self.device = device
        self.stream = T.cuda.Stream(device=device)
        self.next_data = None
        self.preload()

    def __iter__(self):
        return self

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.loader = iter(self._loader)
            self.next_data = None
            return

        with T.cuda.stream(self.stream):
            self.next_data = batch_to_device(self.next_data, device=self.device, non_blocking=True)
            if self.transform is not None:
                self.next_data = self.transform(self.next_data)

    @staticmethod
    def _record_stream(data, device):
        if isinstance(data, T.Tensor):
            data.record_stream(T.cuda.current_stream(device=device))
        else:
            try:
                for e in data:
                    DataPrefetcher._record_stream(e, device)
            except (TypeError, AttributeError):
                root_logger.error('Unknown data type', exc_info=True)
                raise

    def __next__(self):
        if self.next_data is None:
            self.preload()
            raise StopIteration

        T.cuda.current_stream(device=self.device).wait_stream(self.stream)
        data = self.next_data
        DataPrefetcher._record_stream(data, self.device)

        self.preload()
        return data


def truncated_normal(x: T.Tensor, a=-1, b=1, mean=0., std=1.):
    """
    Initializes a tensor from a truncated normal distribution.

    :param x:
        a :class:`torch.Tensor`.
    :param a:
        lower bound of the truncated normal.
    :param b:
        higher bound of the truncated normal.
    :param mean:
        mean of the truncated normal.
    :param std:
        standard deviation of the truncated normal.
    :return:
        ``None``.
    """

    values = truncnorm.rvs(a, b, loc=mean, scale=std, size=list(x.shape))
    with T.no_grad():
        x.data.copy_(T.tensor(values))


def batch_set_value(params, values):
    """
    Sets values of a tensor to another.

    :param params:
        a :class:`torch.Tensor`.
    :param values:
        a :class:`numpy.ndarray` of the same shape as `params`.
    :return:
        ``None``.
    """

    for p, v in zip(params, values):
        p.data.copy_(T.from_numpy(v).data)


def batch_set_tensor(params, values):
    """
    Sets values of a tensor to another.

    :param params:
        a :class:`torch.Tensor`.
    :param values:
        a :class:`torch.Tensor` of the same shape as `params`.
    :return:
        ``None``.
    """

    for p, v in zip(params, values):
        p.data.copy_(v.data)


def to_numpy(x: T.Tensor):
    """
    Moves a tensor to :mod:`numpy`.

    :param x:
        a :class:`torch.Tensor`.
    :return:
        a :class:`numpy.ndarray`.
    """

    return x.cpu().detach().data.numpy()


def to_cuda(x: np.ndarray, non_blocking=False):
    """
    Moves a :mod:`numpy` to tensor.

    :param x:
        a :class:`numpy.ndarray`.
    :return:
        a :class:`torch.Tensor`.
    """

    return T.from_numpy(x).cuda(non_blocking=non_blocking)


def to_cuda_sparse(coo, non_blocking=False):
    """
    Moves a sparse matrix to cuda tensor.

    :param x:
        a :class:`scipy.sparse.coo.coo_matrix`.
    :return:
        a :class:`torch.Tensor`.
    """

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = T.LongTensor(list(indices))
    v = T.FloatTensor(values)
    shape = coo.shape
    return T.sparse.FloatTensor(i, v, T.Size(shape)).cuda(non_blocking=non_blocking)


def bulk_to_numpy(xs):
    """
    Moves a list of tensors to :class:`numpy.ndarray`.

    :param xs:
        a list/tuple of :class:`torch.Tensor` s.
    :return:
        a tuple of :class:`numpy.ndarray` s.
    """

    return tuple([to_numpy(x) for x in xs])


def bulk_to_cuda(xs, non_blocking=False):
    """
    Moves a list of :class:`numpy.ndarray` to tensors.

    :param xs:
        a tuple of :class:`numpy.ndarray` s.
    :return:
        a tuple of :class:`torch.Tensor` s.
    """

    return tuple([to_cuda(x, non_blocking=non_blocking) for x in xs])


def bulk_to_cuda_sparse(xs, non_blocking=False):
    """
    Moves a list sparse matrices to cuda tensor.

    :param x:
        a list/tuple of :class:`scipy.sparse.coo.coo_matrix`.
    :return:
        a :class:`torch.Tensor`.
    """

    return tuple([to_cuda_sparse(x, non_blocking=non_blocking) for x in xs])


def batch_to_device(batch, *args, **kwargs):
    """
    Moves a batch to the specified device.

    :param batch:
        a :class:`torch.Tensor` or an iterable of :class:`torch.Tensor`.
    :return:
        a copy of the original batch that on the specified device.
    """

    if isinstance(batch, T.Tensor) or hasattr(batch, 'to'):
        batch_device = batch.to(*args, **kwargs)
    else:
        try:
            batch_device = [batch_to_device(b, *args, **kwargs) for b in batch]
        except TypeError:
            root_logger.error('batch must be a Tensor or iterable', exc_info=True)
            raise

    return batch_device


def batch_to_cuda(batch, *args, **kwargs):
    """
    Moves a batch to the default CUDA device.

    :param batch:
        a :class:`torch.Tensor` or an iterable of :class:`torch.Tensor`.
    :return:
        a copy of the original batch that on the default CUDA device.
    """

    return batch_to_device(batch, T.device('cuda'), *args, **kwargs)
