import argparse
import numpy as np
import torch as T
import torchvision
from torchvision import transforms
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
import neuralnet_pytorch as nnt
import neuralnet_pytorch.gin_nnt as gin
from inspect import signature

gin.enter_interactive_mode()
parser = argparse.ArgumentParser(description='Neuralnet-pytorch CIFAR10 Training')
parser.add_argument('config', type=str, help='Gin config file to dictate training')
parser.add_argument('--device', '-d', type=str, default=0, nargs='+', help='device(s) to run training')
parser.add_argument('--no-wait-eval', action='store_true', help='whether to use multiprocessing for evaluation')
parser.add_argument('--eval-device', type=int, default=0, help='device to run evaluation')
args = parser.parse_args()

device = [T.device(int(dev) if dev.isdigit() else dev) for dev in args.device]
eval_device = T.device(args.eval_device)
no_wait_eval = args.no_wait_eval
if no_wait_eval:
    assert eval_device not in device, 'device for evaluation must be different from the one for training'

if nnt.cuda_available:
    T.backends.cudnn.benchmark = True

config_file = args.config
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


def eval_queue(q, ckpt):
    mon = nnt.Monitor(current_folder=ckpt)
    while True:
        item = q.get()
        if item == 'DONE':
            break
        elif item is not None:
            it = item
            mon.iter = it
            states = mon.load('tmp.pt', method='torch')
            gin.parse_config_file(config_file)
            loss, accuracy = evaluate(states=states)
            mon.plot('test-loss', loss)
            mon.plot('test-accuracy', accuracy)
            mon.flush()


def get_loss(net, images, labels, reduction='mean'):
    pred = net(images)
    _, predicted = pred.max(1)
    accuracy = predicted.eq(labels).float().mean()

    loss = T.nn.functional.cross_entropy(pred, labels, reduction=reduction)
    return loss, accuracy


@gin.configurable('Classifier')
def train_eval(name, model, dataset, optimizer, scheduler, lr=1e-1, weight_decay=5e-4, bs=128, n_epochs=300,
               start_epoch=0, print_freq=1000, val_freq=10000, checkpoint_folder=None, version=-1,
               use_jit=True, use_amp=False, opt_level='O1', **kwargs):
    assert dataset in ('cifar10', 'cifar100')
    if use_amp:
        import apex

    stem = nnt.ConvNormAct(3, 64, kernel_size=3, stride=1, padding=1, bias=False, activation='relu')
    net = model(num_classes=10 if dataset == 'cifar10' else 100, stem=stem, default_init=False)
    net = net.to(device[0])

    opt_sig = signature(optimizer)
    opt_kwargs = dict([(k, kwargs[k]) for k in kwargs.keys() if k in opt_sig.parameters.keys()])
    optimizer = optimizer(net.trainable, lr=lr, weight_decay=weight_decay, **opt_kwargs)
    if scheduler is not None:
        sch_sig = signature(scheduler)
        sch_kwargs = dict([(k, kwargs[k]) for k in kwargs.keys() if k in sch_sig.parameters.keys()])
        scheduler = scheduler(optimizer, **sch_kwargs)

    dataset = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
    train_data = dataset(root='./data', train=True, download=True, transform=transform_train)
    train_loader = T.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=5)

    if checkpoint_folder is None:
        mon = nnt.Monitor(name, print_freq=print_freq, num_iters=int(np.ceil(len(train_data) / bs)),
                          use_tensorboard=True)
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

    else:
        mon = nnt.Monitor(current_folder=checkpoint_folder, print_freq=print_freq, num_iters=len(train_data) // bs,
                          use_tensorboard=True)
        states = mon.load('training.pt', method='torch', version=version)
        net.load_state_dict(states['model_state_dict'])
        optimizer.load_state_dict(states['opt_state_dict'])
        if scheduler:
            scheduler.load_state_dict(states['scheduler_state_dict'])

        if use_amp and 'amp' in states.keys():
            apex.amp.load_state_dict(states['amp'])

        if start_epoch:
            start_epoch = start_epoch - 1
            mon.epoch = start_epoch

        print('Resume from epoch %d...' % mon.epoch)

    if not no_wait_eval:
        eval_data = dataset(root='./data', train=False, download=True, transform=transform_test)
        eval_loader = T.utils.data.DataLoader(eval_data, batch_size=bs, shuffle=False, num_workers=2)

    if nnt.cuda_available:
        train_loader = nnt.DataPrefetcher(train_loader, device=device[0])
        if not no_wait_eval:
            eval_loader = nnt.DataPrefetcher(eval_loader, device=device[0])

    if use_jit:
        img = T.rand(1, 3, 32, 32).to(device[0])
        net.train(True)
        net_train = T.jit.trace(net, img)
        net.eval()
        net_eval = T.jit.trace(net, img)

    if use_amp:
        if use_jit:
            net_train, optimizer = apex.amp.initialize(net_train, optimizer, opt_level=opt_level)
            net_eval = apex.amp.initialize(net_eval, opt_level=opt_level)
        else:
            net, optimizer = apex.amp.initialize(net, optimizer, opt_level=opt_level)

        if 'amp' not in states.keys():
            states['amp'] = apex.amp.state_dict()

    if use_jit:
        net_train = T.nn.DataParallel(net_train, device_ids=device)
        net_eval = T.nn.DataParallel(net_eval, device_ids=device)
    else:
        net = T.nn.DataParallel(net, device_ids=device)

    def learn(images, labels, reduction='mean'):
        net.train(True)
        optimizer.zero_grad()
        loss, accuracy = get_loss(net_train if use_jit else net, images, labels, reduction=reduction)
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
        mon.plot('train-accuracy', nnt.utils.to_numpy(accuracy))
        del loss, accuracy

    if no_wait_eval:
        q = Queue()
        eval_proc = Process(target=eval_queue, args=(q, mon.current_folder))
        eval_proc.daemon = True
        eval_proc.start()

    start_epoch = mon.epoch if start_epoch is None else start_epoch
    print('Training...')
    with T.jit.optimized_execution(use_jit):
        for epoch in mon.iter_epoch(range(start_epoch, n_epochs)):
            if scheduler is not None:
                scheduler.step(epoch)

            opt = scheduler.optimizer if scheduler is not None else optimizer
            for idx, group in enumerate(opt.param_groups):
                mon.plot('lr-%d' % idx, group['lr'])

            for batch in mon.iter_batch(train_loader):
                batch = nnt.utils.batch_to_device(batch, device[0])

                learn(*batch)
                if val_freq and mon.iter % val_freq == 0:
                    if no_wait_eval:
                        mon.dump('tmp.pt', states, method='torch')
                        q.put(mon.iter)
                        q.put(None)
                    else:
                        net.eval()
                        with T.set_grad_enabled(False):
                            losses, accuracies = [], []
                            for itt, batch in enumerate(eval_loader):
                                batch = nnt.utils.batch_to_device(batch, device[0])

                                loss, acc = get_loss(net_eval if use_jit else net, *batch)
                                losses.append(nnt.utils.to_numpy(loss))
                                accuracies.append(nnt.utils.to_numpy(acc))

                            mon.plot('test-loss', np.mean(losses))
                            mon.plot('test-accuracy', np.mean(accuracies))
            mon.dump('training.pt', states, method='torch', keep=10)

    if no_wait_eval:
        q.put('DONE')
        eval_proc.join()

    print('Training finished!')


@gin.configurable('Classifier')
def evaluate(model, dataset, states, bs=128, use_jit=True, use_amp=False, opt_level='O1', **kwargs):
    assert dataset in ('cifar10', 'cifar100')

    stem = nnt.ConvNormAct(3, 64, kernel_size=3, stride=1, padding=1, bias=False, activation='relu')
    net = model(num_classes=10 if dataset == 'cifar10' else 100, stem=stem, default_init=False)
    net = net.to(eval_device)
    net.load_state_dict(states['model_state_dict'])
    net.eval()

    dataset = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
    eval_data = dataset(root='./data', train=False, download=True, transform=transform_test)
    eval_loader = T.utils.data.DataLoader(eval_data, batch_size=bs, shuffle=False)

    if nnt.cuda_available:
        eval_loader = nnt.DataPrefetcher(eval_loader, device=eval_device)

    if use_jit:
        img = T.rand(1, 3, 32, 32).to(eval_device)
        net = T.jit.trace(net, img)

    if use_amp:
        import apex
        net = apex.amp.initialize(net, opt_level=opt_level)
        apex.amp.load_state_dict(states['amp'])

    net = T.nn.DataParallel(net, device_ids=[eval_device])
    with T.set_grad_enabled(False):
        losses, accuracies = [], []
        for itt, batch in enumerate(eval_loader):
            batch = nnt.utils.batch_to_device(batch, eval_device)

            loss, acc = get_loss(net, *batch)
            losses.append(nnt.utils.to_numpy(loss))
            accuracies.append(nnt.utils.to_numpy(acc))

    return np.mean(losses), np.mean(accuracies)


@gin.configurable('Classifier')
def test(name, model, dataset, bs=128, print_freq=1000, checkpoint_folder=None, version=-1,
         use_jit=True, use_amp=False, opt_level='O1', **kwargs):
    assert dataset in ('cifar10', 'cifar100')

    stem = nnt.ConvNormAct(3, 64, kernel_size=3, stride=1, padding=1, bias=False, activation='relu')
    net = model(num_classes=10 if dataset == 'cifar10' else 100, stem=stem, default_init=False)
    net = net.to(device[0])
    net.eval()

    dataset = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100
    test_data = dataset(root='./data', train=False, download=True, transform=transform_test)
    test_loader = T.utils.data.DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=5)

    if nnt.cuda_available:
        test_loader = nnt.DataPrefetcher(test_loader, device=device[0])

    mon = nnt.Monitor(current_folder=checkpoint_folder, print_freq=print_freq, num_iters=len(test_data) // bs,
                      use_tensorboard=True)
    states = mon.load('training.pt', method='torch', version=version)
    net.load_state_dict(states['model_state_dict'])
    if use_jit:
        img = T.rand(1, 3, 32, 32).to(device[0])
        net = T.jit.trace(net, img)

    if use_amp:
        import apex
        net = apex.amp.initialize(net, opt_level=opt_level)
        apex.amp.load_state_dict(states['amp'])

    net = T.nn.DataParallel(net, device_ids=device)
    print('Resume from epoch %d...' % mon.epoch)
    print('Testing...')
    with T.set_grad_enabled(False):
        for itt, batch in mon.iter_batch(enumerate(test_loader)):
            batch = nnt.utils.batch_to_device(batch, device[0])

            loss, acc = get_loss(net,*batch)
            mon.plot('test-loss', nnt.utils.to_numpy(loss))
            mon.plot('test-accuracy', nnt.utils.to_numpy(acc))
    print('Testing finished!')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    gin.parse_config_file(config_file)
    train_eval()
