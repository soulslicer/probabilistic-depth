import torch
import time
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
import torch.distributed as dist
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint, AdamW

class BaseTrainer(object):
    """
    Base class for all trainers
    """

    def __init__(self, id, model, loss_func, _log, save_root, config, shared):
        self.id = id
        self._log = _log
        self.shared = shared

        self.cfg = config
        self.save_root = save_root
        if self.id == 0:
            self.summary_writer = SummaryWriter(str(save_root))
        else:
            self.summary_writer = None

        self.best_error = np.inf
        self.i_epoch = 0
        self.i_iter = 0

        self.device, self.device_ids = self._prepare_device()
        self.model = self._init_model(model)
        self.optimizer = self._create_optimizer()
        self.loss_func = loss_func

    @abstractmethod
    def _run_one_epoch(self):
        pass

    @abstractmethod
    def _validate_with_gt(self):
        pass

    def train(self):
        for epoch in range(self.cfg.train.epoch_num):
            self._run_one_epoch()

            if self.i_epoch % self.cfg.train.val_epoch_size == 0:
                errors, error_names = self._validate_with_gt()
                valid_res = ' '.join(
                    '{}: {:.2f}'.format(*t) for t in zip(error_names, errors))
                self._log.info(self.id, ' * Epoch {} '.format(self.i_epoch) + valid_res)

            if self.i_epoch in self.cfg.train.halflr:
                self._log.info(self.id, 'Halfving LR')
                for g in self.optimizer.param_groups:
                    g['lr'] /= 2

            if self.i_epoch == self.cfg.train.epoch_num:
                break

    def eval(self):
        errors, error_names = self._validate_with_gt()

    def _init_model(self, model):
        # Load Model to Device
        n_gpu_use = self.cfg.train.n_gpu
        if self.cfg.mp.enabled:
            id = self.id
            if n_gpu_use > 0:
                torch.cuda.set_device(self.device)
                model = model.to(self.device)
        else:
            model = model.to(self.device)

        # Load Weights
        if self.cfg.train.pretrained_model:
            self._log.info(self.id, "=> using pre-trained weights {}.".format(
                self.cfg.train.pretrained_model))
            epoch, weights = load_checkpoint(self.cfg.train.pretrained_model)
            self.i_epoch = epoch

            from collections import OrderedDict
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights)
        else:
            self._log.info(self.id, "=> Train from scratch.")
            model.init_weights()

        # Load Custom
        if hasattr(self.cfg.train, 'init_model'):
            if self.cfg.train.init_model:
                self._log.info(self.id, "=> using init weights {}.".format(
                    self.cfg.train.init_model))
                epoch, weights = load_checkpoint(self.cfg.train.init_model)
                from collections import OrderedDict
                new_weights = OrderedDict()
                model_keys = list(model.state_dict().keys())
                weight_keys = list(weights.keys())
                for a, b in zip(model_keys, weight_keys):
                    new_weights[a] = weights[b]
                weights = new_weights
                model.load_state_dict(weights, strict=False)

        # Model Type
        if self.cfg.mp.enabled:
            if self.cfg.var.bn_avg:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, 0)
            if n_gpu_use > 0:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=self.device_ids, find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        else:
            if n_gpu_use > 0:
                model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            else:
                model = torch.nn.DataParallel(model).to(self.device)

        return model

    def _create_optimizer(self):
        self._log.info(self.id, '=> setting Adam solver')
        param_groups = [
            {'params': bias_parameters(self.model.module),
             'weight_decay': self.cfg.train.bias_decay},
            {'params': weight_parameters(self.model.module),
             'weight_decay': self.cfg.train.weight_decay}]

        if self.cfg.train.optim == 'adamw':
            optimizer = AdamW(param_groups, self.cfg.train.lr,
                              betas=(self.cfg.train.momentum, self.cfg.train.beta))
        elif self.cfg.train.optim == 'adam':
            # optimizer = torch.optim.Adam(param_groups, self.cfg.train.lr,
            #                              betas=(self.cfg.train.momentum, self.cfg.train.beta),
            #                              eps=1e-7)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr,
                                         betas=(self.cfg.train.momentum, self.cfg.train.beta))
        else:
            raise NotImplementedError(self.cfg.train.optim)

        for item in self.cfg.train.halflr:
            if self.i_epoch >= item:
                self._log.info(self.id, 'Halfving LR')
                for g in optimizer.param_groups:
                    g['lr'] /= 2

        return optimizer

    def _prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        # GPU Checks
        id = self.id
        n_gpu_use = self.cfg.train.n_gpu
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning(self.id, "Warning: There\'s no GPU available on this machine,"
                                       "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(self.id,
                              "Warning: The number of GPU\'s configured to use is {}, "
                              "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if self.cfg.mp.enabled:
            device = torch.device('cuda:' + str(id) if n_gpu_use > 0 else 'cpu')
            list_ids = [id]
            return device, list_ids
        else:
            device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
            list_ids = list(range(n_gpu_use))
            return device, list_ids

    def save_model(self, error, name):
        if self.id > 0:
            return

        is_best = error < self.best_error

        if is_best:
            self.best_error = error

        models = {'epoch': self.i_epoch,
                  'state_dict': self.model.module.state_dict()}

        self._log.info(self.id, "=> Saving Model..")

        save_checkpoint(self.save_root, models, name, is_best)
