import torch

import numpy as np

import os
import csv

from .Checkpoint import SaveCheckpoint


class EarlyStopping(object):
    def __init__(self, min_delta, patience, model_save_path, verbose=1, log_path=None):
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.model_save_path = model_save_path
        self.best_epoch = 0
        self.verbose = verbose
        self.log_path = log_path

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, current, net, training_info):
        if np.greater(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            SaveCheckpoint(net,
                           self.model_save_path,
                           'best',
                           verbose=self.verbose)
            if hasattr(net, 'cap_theta') and hasattr(net, 'cap_theta_sq') and hasattr(net, 'cap_D'):
                torch.save(net.cap_theta, os.path.join(self.model_save_path, 'cap_theta.pt'))
                torch.save(net.cap_theta_sq, os.path.join(self.model_save_path, 'cap_theta_sq.pt'))
                torch.save(net.cap_D, os.path.join(self.model_save_path, 'cap_D.pt'))
            if hasattr(net, 'cap_theta') and hasattr(net, 'cap_b') and hasattr(net, 'cap_D'):
                torch.save(net.cap_theta, os.path.join(self.model_save_path, 'cap_theta.pt'))
                torch.save(net.cap_b, os.path.join(self.model_save_path, 'cap_b.pt'))
                torch.save(net.cap_D, os.path.join(self.model_save_path, 'cap_D.pt'))
            self.best_epoch = epoch
            with open(os.path.join(self.log_path, 'update_info.csv'), 'a', encoding='utf-8', newline='') as fp:
                writer = csv.writer(fp)
                if fp.tell() == 0:
                    title = list(training_info.keys())
                    title.append('Epoch')
                    writer.writerow(title)
                training_info = np.around(np.array(list(training_info.values())), 5).tolist()
                training_info.append(epoch)
                writer.writerow(training_info)
            if self.verbose:
                print('Updata: {}(Epoch: {}) +++'.format(np.round(self.best, 5), epoch))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

    def on_train_end(self):
        if self.stopped_epoch > 0:
            if self.verbose:
                print('Early stopping(Epoch: {}) +++' % (self.stopped_epoch))
