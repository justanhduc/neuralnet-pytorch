import torch as T

from . import root_logger

__all__ = ['time_cuda_module', 'slack_message']


def time_cuda_module(f, *args, **kwargs):
    """
    Measures the time taken by a Pytorch module.

    :param f:
        a Pytorch module.
    :param args:
        arguments to be passed to `f`.
    :param kwargs:
        keyword arguments to be passed to `f`.
    :return:
        the time (in second) that `f` takes.
    """

    start = T.cuda.Event(enable_timing=True)
    end = T.cuda.Event(enable_timing=True)

    start.record()
    f(*args, **kwargs)
    end.record()

    # Waits for everything to finish running
    T.cuda.synchronize()

    total = start.elapsed_time(end)
    root_logger.info('Took %fms' % total)
    return total


def slack_message(username: str, message: str, channel: str, token: str, **kwargs):
    """
    Sends a slack message to the specified chatroom.

    :param username:
        Slack username.
    :param message:
        message to be sent.
    :param channel:
        Slack channel.
    :param token:
        Slack chatroom token.
    :param kwargs:
        additional keyword arguments to slack's :meth:`api_call`.
    :return:
        ``None``.
    """

    try:
        from slackclient import SlackClient
    except (ModuleNotFoundError, ImportError):
        from slack import RTMClient as SlackClient

    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel, text=message, username=username, **kwargs)
