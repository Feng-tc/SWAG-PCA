import os, time, random, torch

import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from config import config
from Networks import BaseNetwork
from Controllers import BaseController
from Utils import EarlyStopping, SaveCheckpoint, LoadCheckpoint
from DataLoader import (Caseset, CentralCropTensor, CollateGPU, Dataset, RandomAffineTransform, RandomMirrorTensor2D)
from DataLoader.Utils import ContourPointExtractor


class Solver:

    def __init__(self, config=config):
        
        self.config, seed = config, 1234

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['GPUNo']

        # Pa. 1 Random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministric = True
        torch.backends.cudnn.benchmark = False
        
        _time = time.strftime('%y%m%d%H%M')

        self.getController()

        # if mode is Train, create a new directory to save model
        model_name = self.config['network']
        name = self.config['name']
        mode = self.config['mode']
        if mode in ['Train']:
            self.model_save_name = '{}_{}_{}_{}_{}'.format(_time,
                                                           model_name,
                                                           self.net.name,
                                                           self.config['dataset']['name'],
                                                           name)
            self.model_save_path = os.path.join('res', model_name, 'model', self.model_save_name)
        elif mode in ['CTrain_SWAG']:
            self.model_save_name = '{}_{}_c{}k{}se{}k_{}'.format(_time,
                                                                 model_name,
                                                                 self.config[mode]['c'],
                                                                 self.config[mode]['k'],
                                                                 int(self.config[mode]['swag_se'] // 1e3),
                                                                 self.config[mode]['model_save_path'])
            self.model_save_path = os.path.join('res', model_name, 'model', self.model_save_name)
        elif mode in ['CTrain']:
            self.model_save_name = '{}_{}_se{}k_{}'.format(_time,
                                                                 model_name,
                                                                 int(self.config[mode]['se'] // 1e3),
                                                                 self.config[mode]['model_save_path'])
            self.model_save_path = os.path.join('res', model_name, 'model', self.model_save_name)
        elif mode in ['Post_hoc']:
            _name = self.config[mode]['model_save_path'].split('_')[:3]
            _name = '_'.join(_name)
            self.model_save_name = '{}_{}_{}'.format(_time,
                                                     model_name,
                                                     _name)
            self.model_save_path = os.path.join('res', model_name, 'model', self.model_save_name)
        elif mode == 'Hyperopt':
            self.model_save_name = '{}_{}_{}'.format(_time, name, 'Hyper')
            self.model_save_path = os.path.join(self.config['Train']['model_save_dir'], model_name, self.model_save_name)
        else:
            self.model_save_path = os.path.join('res', model_name, 'model', self.config[mode]['model_save_path'])
            self.model_save_name = os.path.basename(self.model_save_path)
        
        self.getLogger(_time, mode)
        
        if not os.path.exists(self.model_save_path) and mode not in ['Test', 'Test_SWAG', 'Post_hoc']:
            os.makedirs(self.model_save_path)
        if not os.path.exists('res/TestSave') and mode not in ['Test', 'Test_SWAG', 'Post_hoc']:
            os.makedirs('res/TestSave')

    def getController(self):
        print('loading controller...')
        name = self.config['network']
        self.net: BaseNetwork = self.config[name]['network'](**self.config[name]['params'])
        self.controller: BaseController = self.config[name]['controller'](self.net)
        self.controller.cuda()

    def getLogger(self, _time, mode=None):
        if mode in ['Train']:
            self.logger = SummaryWriter(os.path.join('logs', '{}_log_{}_{}_{}_{}'.format(_time,
                                                                                         self.config['network'],
                                                                                         self.net.name,
                                                                                         self.config['dataset']['name'],
                                                                                         self.config['name'])))
        elif mode in ['CTrain_SWAG']:
            _name = self.model_save_name.split('_')
            _name.insert(1, 'log')
            self.logger = SummaryWriter(os.path.join('logs', '_'.join(_name)))
        elif mode in ['CTrain']:
            _name = self.model_save_name.split('_')
            _name.insert(1, 'log')
            self.logger = SummaryWriter(os.path.join('logs', '_'.join(_name)))
        else:
            self.logger = None

    def getTrainDataloader(self):
        training_list_path = self.config['dataset']['training_list_path']
        pair_dir = self.config['dataset']['pair_dir']

        num_NuPs = self.config[self.config['network']]['params'].get('nucpoint_num', 64)  # 默认值 64
        gap = int((200 - 96) / 2 + 16) if self.config[self.config['network']]['params']['i_size'][0] == 96 else int((200 - 128) / 2 + 32) # 128 * 128图中心区域距离边界32个像素
        # gap = int((200 - 96) / 2 + 16) if self.config[self.config['network']]['params']['vol_size'][0] == 96 else int((200 - 128) / 2 + 32) # 128 * 128图中心区域距离边界32个像素
        CPE = ContourPointExtractor([200, 200], gap, num_NuPs)

        dataset = Dataset(training_list_path, pair_dir, num_NuPs, CPE)
        self.train_dataloader = torch.utils.data.DataLoader(dataset,
                                                            batch_size=self.config[self.config['mode']]['batch_size'],
                                                            shuffle=True,
                                                            collate_fn=CollateGPU(transforms=transforms.Compose([
                                                                                  RandomAffineTransform(img_size=[200, 200]),
                                                                                  CentralCropTensor(img_size=[96, 96]),
                                                                                  RandomMirrorTensor2D()])), 
                                                            pin_memory=False,
                                                            num_workers=0)
    
    def getTrainDataloaderInTest(self):

        training_list_path = self.config['dataset']['training_list_path']
        pair_dir = self.config['dataset']['pair_dir']

        num_NuPs = None if self.config[self.config['network']]['params'].get('nucpoint_num') == None else self.config[self.config['network']]['params'].get('nucpoint_num')
        gap = int((200 - 96) / 2 + 16) if self.config[self.config['network']]['params']['i_size'][0] == 96 else int((200 - 128) / 2 + 32) # 128 * 128图中心区域距离边界32个像素
        # gap = int((200 - 96) / 2 + 16) if self.config[self.config['network']]['params']['vol_size'][0] == 96 else int((200 - 128) / 2 + 32) # 128 * 128图中心区域距离边界32个像素
        CPE = ContourPointExtractor([200, 200], gap, num_NuPs)

        dataset = Dataset(training_list_path, pair_dir, num_NuPs, CPE)
        self.trainInTest_dataloader = torch.utils.data.DataLoader(dataset,
                                                                  batch_size=self.config[self.config['mode']]['batch_size'],
                                                                  shuffle=False,
                                                                  collate_fn=CollateGPU(transforms=transforms.Compose([
                                                                                        CentralCropTensor(img_size=[96, 96])])), 
                                                                  pin_memory=False,
                                                                  num_workers=0)

    def getValidationDataloader(self):
        validation_list_path = self.config['dataset']['validation_list_path']
        pair_dir = self.config['dataset']['pair_dir']

        num_NuPs = self.config[self.config['network']]['params'].get('nucpoint_num', 64)  # 默认值 64
        gap = 16 if self.config[self.config['network']]['params']['i_size'][0] == 96 else 32 # 128 * 128图中心区域距离边界32个像素
        # gap = 16 if self.config[self.config['network']]['params']['vol_size'][0] == 96 else 32 # 128 * 128图中心区域距离边界32个像素
        CPE = ContourPointExtractor([200, 200], gap, num_NuPs)

        dataset = Caseset(validation_list_path, 
                          pair_dir, 
                          self.config['dataset']['resolution_path'], 
                          num_NuPs, 
                          CPE,
                          (96, 96))

        self.validation_dataloader = torch.utils.data.DataLoader(dataset,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 pin_memory=False,
                                                                 num_workers=0)

    def getTestDataloader(self, data_list_path: str):

        num_NuPs = self.config[self.config['network']]['params'].get('nucpoint_num', 64)
        gap = 16 if self.config[self.config['network']]['params']['i_size'][0] == 96 else 32  # 128 * 128图中心区域距离边界32个像素
        # gap = 16 if self.config[self.config['network']]['params']['vol_size'][0] == 96 else 32  # 128 * 128图中心区域距离边界32个像素
        CPE = ContourPointExtractor([200, 200], gap, num_NuPs)
        
        dataset = Caseset(data_list_path, 
                          self.config['dataset']['pair_dir'], 
                          self.config['dataset']['resolution_path'], 
                          num_NuPs, 
                          CPE,
                          (96, 96))
        
        self.test_dataloader = torch.utils.data.DataLoader(dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)

    def loadCheckpoint(self, net, epoch=None):
        LoadCheckpoint(net, self.model_save_path, epoch)
    
    def saveCheckpoint(self, net, epoch, optimizer=None):
        if epoch % self.config[config['mode']]['save_checkpoint_step']:
            pass
        else:
            SaveCheckpoint(net, self.model_save_path, epoch, optimizer=optimizer)

    def train(self):
        # data loading
        print('loading data...')
        self.getTrainDataloader()
        self.getValidationDataloader()

        # early stop
        earlystop = EarlyStopping(min_delta=self.config['Train']['earlystop']['min_delta'], 
                                  patience=self.config['Train']['earlystop']['patience'],
                                  model_save_path=self.model_save_path, log_path=self.logger.log_dir)

        # train
        print('training...')
        return self.controller.train(network=self.config['network'],
                                     train_dataloader=self.train_dataloader,
                                     validation_dataloader=self.validation_dataloader,
                                     save_checkpoint=self.saveCheckpoint,
                                     earlystop=earlystop,
                                     logger=self.logger,
                                     start_epoch=0,
                                     max_epoch=self.config['Train']['max_epoch'],
                                     lr=self.config['Train']['lr'])
    
    def test(self):
        self.getTestDataloader(self.config['dataset']['testing_list_path'])
        self.loadCheckpoint(self.controller.net, self.config['Test']['epoch'])

        res = self.controller.test(dataloader=self.test_dataloader, 
                                   name=self.model_save_name,
                                   network=self.config['network'],
                                   excel_save_path=self.config['Test']['excel_save_path'],
                                   logger=self.logger,
                                   verbose=self.config['Test']['verbose'])
        return res

    def SWAG_continue_train(self):
        # data loading
        print('loading data...')
        self.getTrainDataloader()
        self.getValidationDataloader()

        # load_path = os.path.join('res', 'NuNet', 'model', self.config['CTrain_SWAG']['model_save_path'])
        load_path = os.path.join('res', self.config['network'], 'model', self.config['CTrain_SWAG']['model_save_path'])
        LoadCheckpoint(self.controller.net, load_path, self.config['CTrain_SWAG']['swag_se'])
        
        optimizer = torch.optim.Adam(self.controller.net.parameters(), lr=self.config['CTrain_SWAG']['lr'])
        optimizer_path = os.path.join(load_path, 'optimizer_{}.pt'.format(self.config['CTrain_SWAG']['swag_se']))
        optimizer.load_state_dict(torch.load(optimizer_path))
        # early stop
        earlystop = EarlyStopping(min_delta=self.config['CTrain_SWAG']['earlystop']['min_delta'], 
                                  patience=self.config['CTrain_SWAG']['earlystop']['patience'],
                                  model_save_path=self.model_save_path, log_path=self.logger.log_dir)

        # train
        print('training...')
        return self.controller.SWAG_continue_train(network=self.config['network'],
                                                   optimizer=optimizer,
                                                   train_dataloader=self.train_dataloader,
                                                   validation_dataloader=self.validation_dataloader,
                                                   save_checkpoint=self.saveCheckpoint,
                                                   earlystop=earlystop,
                                                   logger=self.logger,
                                                   start_epoch=self.config['CTrain_SWAG']['swag_se'],
                                                   max_epoch=self.config['CTrain_SWAG']['max_epoch'],
                                                   c=self.config['CTrain_SWAG']['c'],
                                                   k=self.config['CTrain_SWAG']['k'],
                                                   pca_enable=self.config['CTrain_SWAG']['pca_enable'],
                                                   pca_save_prop=self.config['CTrain_SWAG']['pca_save_prop'])

    def continue_train(self):
        # data loading
        print('loading data...')
        self.getTrainDataloader()
        self.getValidationDataloader()

        load_path = os.path.join('res', self.config['network'], 'model', self.config['CTrain']['model_save_path'])
        LoadCheckpoint(self.controller.net, load_path, self.config['CTrain']['se'])

        optimizer = torch.optim.Adam(self.controller.net.parameters(), lr=self.config['CTrain']['lr'])
        optimizer_path = os.path.join(load_path, 'optimizer_{}.pt'.format(self.config['CTrain']['se']))
        optimizer.load_state_dict(torch.load(optimizer_path))
        # early stop
        earlystop = EarlyStopping(min_delta=self.config['CTrain']['earlystop']['min_delta'],
                                  patience=self.config['CTrain']['earlystop']['patience'],
                                  model_save_path=self.model_save_path, log_path=self.logger.log_dir)

        # train
        print('training...')
        return self.controller.continue_train(network=self.config['network'],
                                                   optimizer=optimizer,
                                                   train_dataloader=self.train_dataloader,
                                                   validation_dataloader=self.validation_dataloader,
                                                   save_checkpoint=self.saveCheckpoint,
                                                   earlystop=earlystop,
                                                   logger=self.logger,
                                                   start_epoch=self.config['CTrain']['se'],
                                                   max_epoch=self.config['CTrain']['max_epoch'],
                                                   pca_enable=self.config['CTrain']['pca_enable'],
                                                   pca_save_prop=self.config['CTrain']['pca_save_prop'])

    def SWAG_test(self):
        self.getTestDataloader(self.config['dataset']['testing_list_path'])
        self.loadCheckpoint(self.controller.net, self.config['Test_SWAG']['epoch'])

        cap_theta = torch.load(os.path.join(self.model_save_path, 'cap_theta.pt'))
        cap_theta_sq = torch.load(os.path.join(self.model_save_path, 'cap_theta_sq.pt'))
        cap_D = torch.load(os.path.join(self.model_save_path, 'cap_D.pt'))
        
        setattr(self.controller.net, 'cap_theta', cap_theta)
        setattr(self.controller.net, 'cap_theta_sq', cap_theta_sq)
        setattr(self.controller.net, 'cap_D', cap_D)

        res = self.controller.SWAG_test(test_dataloader=self.test_dataloader, 
                                        name=self.model_save_name,
                                        network=self.config['network'],
                                        excel_save_path=self.config['Test_SWAG']['excel_save_path'],
                                        logger=self.logger,
                                        verbose=self.config['Test_SWAG']['verbose'],
                                        k=self.config['CTrain_SWAG']['k'],
                                        s=self.config['CTrain_SWAG']['s'])
        return res
    
    def Post_hoc(self):
        # data loading
        print('loading data...')
        self.getTrainDataloaderInTest()
        self.getTestDataloader(self.config['dataset']['testing_list_path'])

        print('loading SWAG...')
        load_path = os.path.join('res', 'NuNetSWAG', 'model', self.config['Post_hoc']['model_save_path'])
        LoadCheckpoint(self.controller.net, load_path, 'best')

        cap_theta = torch.load(os.path.join(load_path, 'cap_theta.pt'))
        cap_theta_sq = torch.load(os.path.join(load_path, 'cap_theta_sq.pt'))
        prior_diag_cov = cap_theta_sq - cap_theta ** 2
        prior_diag_cov = torch.where(prior_diag_cov > 0, prior_diag_cov, 1e-8)
        setattr(self.controller.net, 'prior_diag_cov', prior_diag_cov)

        print('Generating BNN...')
        return self.controller.post_hoc(train_dataloader=self.trainInTest_dataloader, 
                                        test_dataloader=self.test_dataloader,
                                        name=self.model_save_name,
                                        network=self.config['network'],
                                        excel_save_path=self.config['Post_hoc']['excel_save_path'],
                                        logger=None,
                                        verbose=self.config['Post_hoc']['verbose'])

    def hyperopt(self):
        SaveCheckpoint(self.controller.net, self.model_save_path, 0)
        hyperparams = self.config[self.config['network']]['hyperparams']
        n_trials = self.config['Hyperopt']['n_trials']
        self.getTrainDataloader()
        self.getValidationDataloader()
        self.getTestDataloader(self.config['dataset']['validation_list_path'])
        # early stop
        earlystop = EarlyStopping(
            min_delta=self.config['Hyperopt']['earlystop']['min_delta'],
            patience=self.config['Hyperopt']['earlystop']['patience'],
            model_save_path=self.model_save_path,
            verbose=0)
        max_epoch = self.config['Hyperopt']['max_epoch']
        lr = self.config['Hyperopt']['lr']

        self.controller.hyperOpt(hyperparams, self.loadCheckpoint, n_trials,
                                 self.train_dataloader,
                                 self.validation_dataloader,
                                 self.test_dataloader, earlystop, self.logger,
                                 max_epoch, lr)

    def speedTest(self):
        self.getTestDataloader(self.config['dataset']['testing_list_path'])
        self.loadCheckpoint(self.controller.net,
                            self.config['SpeedTest']['epoch'])
        self.controller.speedTest(self.test_dataloader,
                                  self.config['SpeedTest']['device'])

    def run(self):
        if self.config['mode'] == 'Train':
            self.train()
            self.test()
        elif self.config['mode'] == 'CTrain_SWAG':
            start_time = time.perf_counter()
            self.SWAG_continue_train()
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"SWAG_continue_train() execution_time: {execution_time:.4f} s")
            self.SWAG_test()
            self.test()
        elif self.config['mode'] == 'CTrain':
            start_time = time.perf_counter()
            self.continue_train()
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"continue_train() execution_time: {execution_time:.4f} s")
            self.test()
        elif self.config['mode'] == 'Test_SWAG':
            self.SWAG_test()
        elif self.config['mode'] == 'Post_hoc':
            self.Post_hoc()
        elif self.config['mode'] == 'Test':
            self.test()
        elif self.config['mode'] == 'Hyperopt':
            self.hyperopt()
        elif self.config['mode'] == 'SpeedTest':
            self.speedTest()


if __name__ == "__main__":
    solver = Solver(config)
    solver.run()
