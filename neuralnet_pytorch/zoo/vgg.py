import torch.nn as nn
from torchvision.models.vgg import model_urls
from torchvision.models.utils import load_state_dict_from_url
from ..layers import *
from ..utils import batch_set_tensor

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(Sequential):

    def __init__(self, features, num_classes=1000, default_init=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = wrapper(output_size=(7, 7))(nn.AdaptiveAvgPool2d)()
        self.classifier = Sequential(
            FC(512 * 7 * 7, 4096, flatten=True, activation='relu'),
            wrapper()(nn.Dropout)(),
            FC(4096, 4096, activation='relu'),
            wrapper()(nn.Dropout)(),
            FC(4096, num_classes),
        )
        if default_init:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = ConvNormAct if batch_norm else Conv2d
            layers += [conv(in_channels, v, kernel_size=3, padding=1, activation='relu')]
            in_channels = v
    return Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        try:
            batch_set_tensor(model.state_dict().values(), state_dict.values())
        except (RuntimeError, ValueError):
            state_dict_iter = iter(state_dict.items())
            for k, v in model.state_dict().items():
                if 'num_batches_tracked' not in k:
                    k_t, v_t = next(state_dict_iter)
                    param_name = k.split('.')[-1]
                    value_name = k_t.split('.')
                    if param_name != value_name[-1]:
                        value_name[-1] = param_name
                        value_name = '.'.join(value_name)
                        v.data.copy_(state_dict[value_name].data)
                    else:
                        v.data.copy_(v_t.data)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    """
    VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    """
    VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """
    VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    """
    VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    """
    VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    """
    VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    """
    VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
