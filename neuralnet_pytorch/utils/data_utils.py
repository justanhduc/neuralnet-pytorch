import numpy as np
import torch as T
import threading
import warnings
from torch.utils.data.dataloader import default_collate
from queue import Queue
from scipy.stats import truncnorm

from . import root_logger

__all__ = ['DataLoader', 'DataPrefetcher', 'truncated_normal', 'batch_set_value', 'batch_to_cuda', 'bulk_to_cuda',
           'bulk_to_cuda_sparse', 'bulk_to_numpy', 'to_cuda', 'to_cuda_sparse', 'to_numpy']


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
        batch size
    shuffle
        whether to shuffle in each iteration.
    num_workers
        number of threads to be used.
    collate_fn
        function to specify how a batch is loaded.
    num_cached
        number of batches to be cached.
    kwargs
        arguments what will not be used.
        For compatibility with :class:`torch.utils.data.Dataloader` only.

    Attributes
    ----------
    batches
        contains batches of data when iterating over the dataset.
    _indices
        contains the indices of samples of the dataset.
    num_batches
        the number of batches in the dataset.
    """

    def __init__(self, dataset, batch_size, shuffle=False, num_workers=10, collate_fn=None, num_cached=10, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_cached = num_cached
        self.num_workers = num_workers
        self.collate_fn = default_collate if collate_fn is None else collate_fn
        self.batches = None
        self._indices = None
        self.num_batches = len(self.dataset) // self.batch_size

        if len(kwargs.keys()) > 0:
            warnings.warn(str(kwargs.keys()) +
                          " are present for compatibility with "
                          "`torch.utils.data.DataLoader` interface"
                          " and will be ignored.",
                          stacklevel=2)

    def __iter__(self):
        self.batches = self._get_batches()
        return self

    def __next__(self):
        if self.batches is None:
            self.batches = self._get_batches()
        return self.batches.__next__()

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

    def _generator(self):
        assert isinstance(self.dataset[0],
                          (list, tuple, np.ndarray)), 'Dataset should consist of lists/tuples or Numpy ndarray'
        self._indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self._indices)

        for i in range(self.num_batches):
            slice_ = self._indices[i * self.batch_size:(i + 1) * self.batch_size]
            batch = [self.dataset[s] for s in slice_]
            batch = self.collate_fn(batch)
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

    Attributes
    ----------
    next_data
        the prefetched data.
    stream
        an instance of :class:`torch.cuda.Stream`.
    """

    def __init__(self, loader, transform=None):
        self._loader = loader
        self.loader = iter(loader)
        self.transform = transform
        self.stream = T.cuda.Stream()
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
            self.next_data = batch_to_cuda(self.next_data, non_blocking=True)
            if self.transform is not None:
                self.next_data = self.transform(self.next_data)

    @staticmethod
    def _record_stream(data):
        if isinstance(data, T.Tensor):
            data.record_stream(T.cuda.current_stream())
        else:
            try:
                for e in data:
                    DataPrefetcher._record_stream(e)
            except (TypeError, AttributeError):
                root_logger.error('Unknown data type', exc_info=True)
                raise

    def __next__(self):
        if self.next_data is None:
            self.preload()
            raise StopIteration

        T.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        DataPrefetcher._record_stream(data)

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
        a :class:`torch.Tensor` of the same shape as `params`.
    :return:
        ``None``.
    """

    for p, v in zip(params, values):
        p.data.copy_(T.from_numpy(v))


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
        a list/tuple of :class:`torch.Tensor` s.
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


def batch_to_cuda(batch, non_blocking=False):
    assert isinstance(batch, (T.Tensor, list, tuple)), 'Unknownn type of batch'

    batch_cuda = batch.cuda(non_blocking=non_blocking) if isinstance(batch, T.Tensor) \
        else [b.cuda(non_blocking=non_blocking) if not isinstance(b, (list, tuple))
              else [bb.cuda(non_blocking=non_blocking) for bb in b] for b in batch]
    return batch_cuda
