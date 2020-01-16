import numpy as np
import torch as T
import torchvision
import neuralnet_pytorch as nnt
from torchvision import transforms
import neuralnet_pytorch.gin_nnt as gin

config_file = 'configs/cifar.gin'
backup_files = (__file__, config_file)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


@gin.configurable('Classifier')
def train_test(name, model, dataset, optimizer, scheduler, lr=1e-1, weight_decay=5e-4, momentum=0.9, bs=128,
               n_epochs=300, gamma=.1, milestones=(150, 250), start_epoch=0, print_freq=1000, val_freq=10000,
               checkpoint_folder=None, version=-1, use_jit=True, use_amp=False, opt_level='O1'):
    assert dataset in ('cifar10', 'cifar100')

    net = model(num_classes=10 if dataset == 'cifar10' else 100)
    optimizer = optimizer(net.trainable, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if scheduler is not None:
        scheduler = scheduler(optimizer, milestones=milestones, gamma=gamma)

    dataset = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
    train_data = dataset(root='./data', train=True, download=True, transform=transform_train)
    train_loader = T.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    train_loader = nnt.DataPrefetcher(train_loader)

    test_data = dataset(root='./data', train=False, download=True, transform=transform_test)
    test_loader = T.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)
    test_loader = nnt.DataPrefetcher(test_loader)

    if use_jit:
        img = T.rand(1, 3, 32, 32).cuda()
        net.train(True)
        net_train = T.jit.trace(net, img)
        net.eval()
        net_eval = T.jit.trace(net, img)

    if use_amp:
        import apex
        if use_jit:
            net_train, optimizer = apex.amp.initialize(net_train, optimizer, opt_level=opt_level)
            net_eval = apex.amp.initialize(net_eval, opt_level=opt_level)
        else:
            net, optimizer = apex.amp.initialize(net, optimizer, opt_level=opt_level)

    if checkpoint_folder is None:
        mon = nnt.Monitor(name, print_freq=print_freq, num_iters=len(train_data) // bs, use_tensorboard=True)
        mon.backup(backup_files)

        mon.dump_rep('network', net)
        mon.dump_rep('optimizer', optimizer)

        states = {
            'model_state_dict': net.state_dict(),
            'opt_state_dict': optimizer.state_dict()
        }

        if scheduler is not None:
            mon.dump_rep('scheduler', scheduler)
            states['scheduler_state_dict'] = scheduler.state_dict()

        if use_amp:
            states['amp'] = apex.amp.state_dict()

        print('Training...')
    else:
        mon = nnt.Monitor(current_folder=checkpoint_folder, print_freq=print_freq, num_iters=len(train_data) // bs,
                          use_tensorboard=True)
        states = mon.load('training.pt', method='torch', version=version)
        net.load_state_dict(states['model_state_dict'])
        optimizer.load_state_dict(states['opt_state_dict'])
        if scheduler:
            scheduler.load_state_dict(states['scheduler_state_dict'])

        if use_amp:
            apex.amp.load_state_dict(states['amp'])

        if start_epoch:
            start_epoch = start_epoch - 1
            mon.epoch = start_epoch

        print('Resume from epoch %d...' % mon.epoch)

    def get_loss(images, labels, training=False, reduction='mean'):
        if use_jit:
            pred = net_train(images) if training else net_eval(images)
        else:
            net.train(training)
            pred = net(images)

        _, predicted = pred.max(1)
        accuracy = predicted.eq(labels).mean()

        loss = T.nn.functional.cross_entropy(input, labels, reduction=reduction)
        return loss, accuracy

    def learn(images, labels, reduction='mean'):
        optimizer.zero_grad()
        loss, accuracy = get_loss(images, labels, training=True, reduction=reduction)
        if not (T.isnan(loss) or T.isinf(loss)):
            if use_amp:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
        else:
            raise ValueError('NaN encountered!')

        mon.plot('train-loss', nnt.utils.to_numpy(loss))
        del loss

    start_epoch = mon.epoch if start_epoch is None else start_epoch
    with T.jit.optimized_execution(use_jit):
        for epoch in mon.iter_epoch(range(start_epoch, n_epochs)):
            if scheduler is not None:
                scheduler.step(epoch)

            opt = scheduler.optimizer if scheduler is not None else optimizer
            for idx, group in enumerate(opt.param_groups):
                mon.plot('lr-%d' % idx, group['lr'])

            for batch in mon.iter_batch(train_loader):
                if nnt.cuda_available:
                    batch = nnt.utils.batch_to_cuda(batch)

                learn(*batch)
                if val_freq and mon.iter % val_freq == 0:
                    with T.set_grad_enabled(False):
                        losses, accuracies = [], []
                        for itt, batch in enumerate(test_loader):
                            if nnt.cuda_available:
                                batch = nnt.utils.batch_to_cuda(batch)

                            loss, acc = get_loss(*batch)
                            losses.append(nnt.utils.to_numpy(loss))
                            accuracies.append(nnt.utils.to_numpy(acc))

                        mon.plot('test-loss', np.mean(losses))
                        mon.plot('test-accuracy', np.mean(accuracies))
            mon.dump('training.pt', states, method='torch', keep=10)
    print('Training finished!')


if __name__ == '__main__':
    gin.parse_config_file(config_file)
    train_test()
