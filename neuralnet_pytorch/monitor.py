'''
Original version from https://github.com/igul222/improved_wgan_training
Collected and modified by Nguyen Anh Duc
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import pickle as pickle
from scipy.misc import imsave
import os
from shutil import copyfile

from neuralnet_pytorch import utils, model


class Monitor(utils.ConfigParser):
    def __init__(self, config_file):
        super(Monitor, self).__init__(config_file)
        self._num_since_beginning = collections.defaultdict(lambda: {})
        self._num_since_last_flush = collections.defaultdict(lambda: {})
        self._img_since_last_flush = collections.defaultdict(lambda: {})
        self._iter = [0]
        self.name = self.config['model']['name']
        self.root = 'results' if self.config['result']['root'] is None else self.config['result']['root']
        self.path = self.root + '/' + self.name
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        subfolders = os.listdir(self.path)
        self.current_folder = self.path + '/run%d' % (len(subfolders) + 1)
        idx = 1
        while os.path.exists(self.current_folder):
            self.current_folder = self.path + '/run%d' % (len(subfolders) + 1 + idx)
            idx += 1
        os.mkdir(self.current_folder)
        copyfile(config_file, '%s/network_config.config' % self.current_folder)
        print('Result folder: %s' % self.current_folder)

    def dump_model(self, network):
        assert isinstance(network, model.Model), 'network must be an instance of Model, got {}'.format(type(network))
        with open('%s/network.txt' % self.current_folder, 'w') as outfile:
            outfile.write("\n".join(str(x) for x in network))

    def tick(self):
        self._iter[0] += 1

    def reset(self):
        self._num_since_beginning = collections.defaultdict(lambda: {})
        self._num_since_last_flush = collections.defaultdict(lambda: {})
        self._img_since_last_flush = collections.defaultdict(lambda: {})
        self._iter = [0]

    def plot(self, name, value):
        self._num_since_last_flush[name][self._iter[0]] = value

    def save_image(self, name, tensor_img, callback=lambda x: x):
        '''

        :param tensor_img: (int, int, int, int)
        :return:
        '''
        self._img_since_last_flush[name][self._iter[0]] = callback(tensor_img)

    def flush(self):
        prints = []

        for name, vals in list(self._num_since_last_flush.items()):
            prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
            self._num_since_beginning[name].update(vals)

            x_vals = np.sort(list(self._num_since_beginning[name].keys()))
            y_vals = [self._num_since_beginning[name][x] for x in x_vals]

            plt.clf()
            plt.plot(x_vals, y_vals)
            plt.xlabel('iteration')
            plt.ylabel(name)
            plt.savefig(self.current_folder + '/' + name.replace(' ', '_')+'.jpg')
        self._num_since_last_flush.clear()

        for name, vals in list(self._img_since_last_flush.items()):
            for val in vals.values():
                if val.dtype == 'float32':
                    val = (255.99 * val).astype('uint8')
                if len(val.shape) == 4:
                    for num in range(val.shape[0]):
                        img = val[num]
                        img = np.squeeze(np.transpose(img, (1, 2, 0)))
                        imsave(self.current_folder + '/' + name + '_%d.jpg' % num, img)
                elif len(val.shape) == 3:
                    imsave(name + '.jpg', val)
                else:
                    raise NotImplementedError
        self._img_since_last_flush.clear()

        with open(self.current_folder + '/log.pkl', 'wb') as f:
            pickle.dump(dict(self._num_since_beginning), f, pickle.HIGHEST_PROTOCOL)

        print("Iteration {}\t{}".format(self._iter[0], "\t".join(prints)))

    def read_log(self):
        raise NotImplemented
