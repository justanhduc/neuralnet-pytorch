from neuralnet_pytorch import utils


class Training(utils.ConfigParser):
    def __init__(self, config_file, **kwargs):
        super(Training, self).__init__(config_file, **kwargs)

        self.n_epochs = self.config['training']['n_epochs']
        self.continue_training = self.config['training']['continue']
        self.batch_size = self.config['training']['batch_size']
        self.validation_frequency = self.config['training']['validation_frequency']
        self.validation_batch_size = self.config['training']['validation_batch_size']
        self.extract_params = self.config['training']['extract_params']
        self.param_file = self.config['training']['param_file']
        self.testing_batch_size = self.config['testing']['batch_size']
        self.path = self.config['data']['path']
        self.batch_size = self.config['training']['batch_size']
        self.test_batch_size = self.config['training']['validation_batch_size']
        self.shuffle = self.config['data']['shuffle']
        self.no_target = self.config['data']['no_target']
        self.augmentation = self.config['data']['augmentation']
        self.num_cached = self.config['data']['num_cached']
        self.training_set = None
        self.testing_set = None
        self.num_train_data = None
        self.num_test_data = None

    def load_data(self):
        raise NotImplementedError
