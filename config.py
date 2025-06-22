from Controllers import BaseController
from Networks import (DalcaDiffNet, KrebsDiffNet, RBFGIGenerativeNetwork, CycleMorph,
                      VoxelMorph, RBFDCNUGenerativeNetwork, RBFDCNUWGenerativeNetwork, 
                      RBFDCNUGenerativeNetwork_ZiYu,
                      NuNetM, NuNetMCR, 
                      NuNetLapRedux)

from Network import NuNet, NuNetSWAG, TransMatch, LL_Net
from Networks import CycleMorph, DalcaDiff, hypernet, RBFGI, VoxelMorph


config = {
    'GPUNo': '0',  # GPU Id
    'mode': 'Test_SWAG',  # Train or Test or CTrain_SWAG or Test_SWAG
    'network': 'NuNetSWAG',  # NuNet NuNetSWAG
    'name': 'small york batch=8 lastConvBlocks',
    'dataset': {
        'name': 'smallYork',
        'training_list_path': '../shsdata/dataset2D_shs/york/training_pair.txt',
        'validation_list_path': '../shsdata/dataset2D_shs/york/validation_pair.txt',
        'testing_list_path': '../shsdata/dataset2D_shs/york/testing_pair.txt',
        'pair_dir': '../shsdata/dataset2D_shs/data/',
        'resolution_path': '../shsdata/dataset2D_shs/resolution.txt'
    },
    # 'dataset': {
    #     'name': 'AYM',
    #     'training_list_path': '../shsdata/dataset2D_shs/training_pair.txt',
    #     'validation_list_path': '../shsdata/dataset2D_shs/validation_pair.txt',
    #     'testing_list_path': '../shsdata/dataset2D_shs/testing_pair.txt',
    #     'pair_dir': '../shsdata/dataset2D_shs/data/',
    #     'resolution_path': '../shsdata/dataset2D_shs/resolution.txt'
    # },
    # 'dataset': {
    #     'name': 'MMs',
    #     'training_list_path': '../shsdata/dataset2D_MnMs/training_pair.txt',
    #     'validation_list_path': '../shsdata/dataset2D_MnMs/validation_pair.txt',
    #     'testing_list_path': '../shsdata/dataset2D_MnMs/testing_pair.txt',
    #     'pair_dir': '../shsdata/dataset2D_MnMs/data/',
    #     'resolution_path': '../shsdata/dataset2D_MnMs/resolution.txt'
    # },
    'CTrain': {
        'batch_size': 8,
        'model_save_dir': '',
        'lr': 1e-4,
        'max_epoch': 4000,
        'save_checkpoint_step': 1000,
        'earlystop': {
            'min_delta': 1e-5,
            'patience': 6000
        },
        'se': 6000,  # the start time (epoch) of swag
        'pca_enable': False,
        'pca_save_prop': 0.3,
        'model_save_path': '2505191910_CycleMorph_3-0.5-1.0_smallYork_small york batch=8 lastConvBlocks'
    },
    'CTrain_SWAG': {
        'batch_size': 8,
        'model_save_dir': '',
        'lr': 1e-4,
        'max_epoch': 3000,
        'save_checkpoint_step': 1000,
        'earlystop': {
            'min_delta': 1e-5,
            'patience': 6000
        },
        'c': 5,  # moment update frequency
        'k': 20,  # maximum number of columns in deviation matrix
        's': 30,
        'swag_se': 4000,  # the start time (epoch) of swag
        'pca_enable': True,
        'pca_save_prop': 0.6,
        'model_save_path': '2505172319_NuNet_lam150kjac10_smallYork_small york batch=8 lastConvBlocks'
        # 2506081112_RBFGI_LCC--[9, 9]-147000_smallYork_small york batch=8 lastConvBlocks
        # 'model_save_path': '2505172319_NuNet_lam150kjac10_smallYork_small york batch=8 lastConvBlocks'
    },
    'Test_SWAG': {
        'epoch': 'best',
        'model_save_path': '2506212014_NuNetSWAG_c5k20se4k_2505172319_NuNet_lam150kjac10_smallYork_small york batch=8 lastConvBlocks',
        'excel_save_path': 'res/TestSave',
        'verbose': 1,
    },
    'Train': {
        'batch_size': 8,
        'model_save_dir': '',
        'lr': 1e-4,
        'max_epoch': 3000,
        'save_checkpoint_step': 1000,
        'earlystop': {
            'min_delta': 1e-5,
            'patience': 6000
        },
    },
    'Test': {
        'epoch': 'best',
        'model_save_path': '2505172319_NuNet_lam150kjac10_smallYork_small york batch=8 lastConvBlocks',
        'excel_save_path': 'res/TestSave',
        'verbose': 2,
    },
    'NuNetSWAG': 
    {
        'controller': BaseController,
        'network': NuNetSWAG,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [96, 96],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'hyperparam':{'similarity_factor': 150000,
                          'jac': 10},
            'cropSize': 64,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'TransMatch':
    {
        'controller': BaseController,
        'network': TransMatch,
        'params': {
            'i_size': [96, 96],
            'loss_weights': [1.0, 4.0],
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'LL_Net':
    {
        'controller': BaseController,
        'network': LL_Net,
        'params': {
            'encoder_param': {
                'start_channels': 8,
                'large_kernel': 5,
                'small_kernel': 3,
                'in_channels': 2,
                'out_channels': 2
            },
            'i_size': [96, 96],
            'loss_weights': [1.0, 5.0],
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'NuNet': 
    {
        'controller': BaseController,
        'network': NuNet,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [96, 96],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'hyperparam':{'similarity_factor': 150000,
                          'jac': 10},
            'cropSize': 64,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'Train_SWAG': {
        'batch_size': 8,
        'model_save_dir': '',
        'lr': 1e-4,
        'max_epoch': 3000,
        'save_checkpoint_step': 1000,
        'earlystop': {
            'min_delta': 1e-5,
            'patience': 1000
        },
        'c': 10, # moment update frequency
        'k': 20, # maximum number of columns in deviation matrix
        'swag_se': 4000, # the start time (epoch) of swag
    },
    'LapRedux_train': {
        'batch_size': 32,
        'model_save_dir': '',
        'lr': 1e-4,
        'max_epoch': 3000,
        'save_checkpoint_step': 1000,
        'earlystop': {
            'min_delta': 1e-5,
            'patience': 1000
        },
        'weight_decay': 0.1,
    },
    'LapRedux_test': {
        'batch_size': 32,
        'epoch': 'best',
        'model_save_path': '2504111026_NuNetLapRx_lam200kjac100_AYM_train york',
        'excel_save_path': 'res/TestSave',
        'verbose': 2,
    },
    'LapRedux_PH': {
        'epoch': 'best',
        'batch_size': 32,
        'model_save_dir': '',
        'lr': 1e-4,
        'max_epoch': 2,
        'save_checkpoint_step': 1000,
        'earlystop': {
            'min_delta': 1e-5,
            'patience': 1000
        },
        'EKFAC': False, # True or False
    },
    'NuNetLapRx': 
    {
        'controller': BaseController,
        'network': NuNetLapRedux,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [96, 96],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'hyperparam':{'similarity_factor': 200000,
                          'jac': 100},
            'cropSize': 64,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'NuNetM': 
    {
        'controller': BaseController,
        'network': NuNetM,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [96, 96],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'hyperparam':{'mode': 'iwae',
                          'similarity_factor': 15,
                          'beta': 10000,
                          'jac': 10},
            'cropSize': 64,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'NuNetMCR': 
    {
        'controller': BaseController,
        'network': NuNetMCR,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [96, 96],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'hyperparam':{'mode': 'cat',
                          'similarity_factor': 15,
                          'beta': 10000,
                          'jac': 10},
            'cropSize': 64,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'RBFGI': {
        'controller': BaseController,
        'network': RBFGIGenerativeNetwork,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 32, 32],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 32, 32],
                'local_num_layers': [2, 2, 2, 2, 2]
            },
            'c': 2,
            'i_size': [96, 96],
            'similarity_factor': 147000,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            },
        }
    },
    'CycleMorph': {
        'controller': BaseController,
        'network': CycleMorph,
        'params': {
            'vol_size': [96, 96],
            'enc_nf': [16, 32, 32, 32, 32],
            'dec_nf': [32, 32, 32, 8, 8, 2],
            'lambda_R': 3,
            'lambda_A': 0.5,
            'lambda_B': 1.0,
            # LCC
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
               'win': [9, 9]
            }
        }
    },
    'DalcaDiff': {
        'controller': BaseController,
        'network': DalcaDiffNet,
        'params': {
            'i_size': [96, 96],
            'encoder_layers': [16, 32, 32, 32],
            'decoder_layers': [32, 32, 32],
            'lam': 100000,
            'similarity_loss': 'MSE',
            'theta_sq': 1 / (0.035**2),
            'K': 1,
            'steps': 7,
            'vel_resize': 1 / 2,
            'bidir': False
        }
    },
    'VoxelMorph': {
        'controller': BaseController,
        'network': VoxelMorph,
        'params': {
            'vol_size': [96, 96],
            'enc_nf': [16, 32, 32, 32],
            'dec_nf': [32, 32, 32, 32, 16, 16],
            'reg_param': 1,
            'full_size': True,
            'similarity_loss': 'MSE',
            'similarity_loss_param': {
                'weight': 1
            }
        }
    },
    'KrebsDiff': {
        'controller': BaseController,
        'network': KrebsDiffNet,
        'params': {
            'i_size': [96, 96],
            'z_dim': 16,
            'encoder_param': [16, 32, 4],
            'decoder_param': [32, 32, 32, 16, 2],
            'down_sampling': [2, 4],
            'similarity_factor': 10000,
            'smooth_kernel': 15,
            'smooth_sigma': 3,
            'N': 4,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            },
        }
    },
    'RBFDCNU': 
    {
        'controller': BaseController,
        'network': RBFDCNUGenerativeNetwork,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [128, 128],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'similarity_factor': 120000,
            'loss_mode': 0,
            'cropSize': 64,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'RBFDCNU_ZiYu': 
    {
        'controller': BaseController,
        'network': RBFDCNUGenerativeNetwork_ZiYu,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [128, 128],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'similarity_factor': 240000,
            'loss_mode': 0,
            'cropSize': 64,
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
    },
    'RBFDCNUW': 
    {
        'controller': BaseController,
        'network': RBFDCNUWGenerativeNetwork,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 64, 64],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [16, 32, 32, 64],
                'local_num_layers': [2, 2, 2, 2]
            },
            'i_size': [128, 128],
            'c_factor': 2,
            'cpoint_num': 128,
            'nucpoint_num': 64,
            'ucpoint_num': 64,
            'similarity_factor': 240000,
            'loss_mode': 0,
            'cropSize': 64,
            'similarity_loss': 'ULCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
        'hyperparams': {
            'factor_list': [
                {
                    'type': 'suggest_int',
                    'params': {
                        'low': 5000,
                        'high': 500000,
                        'step': 5000
                    }
                }, 0, 0
                #  {
                #     'type': 'suggest_int',
                #     'params': {
                #         'low': 0,
                #         'high': 1000,
                #         'step': 50
                #     }
                # }, {
                #     'type': 'suggest_float',
                #     'params': {
                #         'low': 0,
                #         'high': 10,
                #         'step': 0.5
                #     }
                # }
            ],
            #LCC
            'similarity_loss_param': {
                'win': [9, 9]
            }
        }
    },
    'SpeedTest': {
        'epoch': 'best',
        'model_save_path': 'res/RBFDCNU/model/',
        'device': 'cpu'
    },
    'Hyperopt': {
        'n_trials': 30,
        'earlystop': {
            'min_delta': 0.00001,
            'patience': 500
        },
        'max_epoch': 800,
        'lr': 1e-4
    },
}
