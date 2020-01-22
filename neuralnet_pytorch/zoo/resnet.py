import torch.nn as nn
from torchvision.models.resnet import model_urls
from torchvision.models.utils import load_state_dict_from_url
from ..layers import Sequential, Conv2d, BatchNorm2d, FC, MaxPool2d, GlobalAvgPool2D, GroupNorm, \
    ResNetBottleneckBlock, ResNetBasicBlock
from ..utils import batch_set_tensor

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(Sequential):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, default_init=True):
        super(ResNet, self).__init__(input_shape=3)
        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.stem = Sequential(*(
            Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes, activation='relu'),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = GlobalAvgPool2D()
        self.fc = FC(512 * block.expansion, num_classes)

        if default_init:
            for m in self.modules():
                if isinstance(m, Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (BatchNorm2d, GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBottleneckBlock):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNetBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return Sequential(*layers)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
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


def resnet18(pretrained=False, progress=True, **kwargs):
    """
    ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _resnet('resnet18', ResNetBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """
    ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _resnet('resnet34', ResNetBasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', ResNetBottleneckBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """
    ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _resnet('resnet101', ResNetBottleneckBlock, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """
    ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    return _resnet('resnet152', ResNetBottleneckBlock, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """
    ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', ResNetBottleneckBlock, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """
    ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', ResNetBottleneckBlock, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """
    Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', ResNetBottleneckBlock, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    """
    Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    :param pretrained:
        If True, returns a model pre-trained on ImageNet.
    :param progress:
        If True, displays a progress bar of the download to stderr.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', ResNetBottleneckBlock, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
