import os, time, optuna, csv

import numpy as np

import torch

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Metrics import MetricTest
from Modules.Loss import DiceCoefficientAll
from Networks import BaseRegistraionNetwork
from Utils import EarlyStopping, ParamsAll

from collections import defaultdict
import random


class BaseController:
    def __init__(self, net: BaseRegistraionNetwork):
        self.net = net
        self.layer_param_blocks = {}
        self.pca_save_count = 0
        self.pca_save_param = {}
        self.idx_map = {}

    def get_all_param_names(self):
        """获取所有参数名列表（按内存地址排序）"""
        param_objects = []
        for name, param in self.net.named_parameters():
            param_objects.append((name, param))

        sorted_params = sorted(param_objects, key=lambda x: x[1].data_ptr())
        return [name for name, _ in sorted_params]

    def build_layer_param_mapping(self):
        """重构参数映射逻辑，支持复杂层结构"""
        # 获取所有目标参数
        params = list(self.net.iter_params_with_lastConvBlockInEncoder())
        vec_params = parameters_to_vector(params)

        # 获取参数名列表，确保与params顺序一致
        param_names = []
        for name, param in self.net.named_parameters():
            # 检查这个参数是否在我们的目标参数列表中
            for target_param in params:
                if param is target_param:
                    param_names.append(name)
                    break

        print(f"Total params: {len(params)}, Total names: {len(param_names)}")

        current_idx = 0
        self.layer_param_blocks = {}
        self.idx_map = {}

        # 创建参数名到基础名的映射（去掉.weight和.bias后缀）
        param_base_names = {}
        for name in param_names:
            if name.endswith('.weight'):
                base_name = name[:-7]  # 去掉'.weight'
            elif name.endswith('.bias'):
                base_name = name[:-5]  # 去掉'.bias'
            else:
                base_name = name
            param_base_names[name] = base_name

        # 按基础名分组参数
        base_groups = defaultdict(list)
        for i, name in enumerate(param_names):
            base_name = param_base_names[name]
            base_groups[base_name].append((name, params[i]))

        # 为每个基础名创建连续块
        for base_name, param_list in base_groups.items():
            start_idx = float('inf')
            end_idx = 0

            for name, param in param_list:
                # 记录单个参数范围
                param_size = param.numel()
                self.idx_map[name] = (current_idx, current_idx + param_size)

                # 更新整个参数块范围
                start_idx = min(start_idx, current_idx)
                end_idx = max(end_idx, current_idx + param_size)
                current_idx += param_size

            # 保存整个参数块范围（包含weight和bias）
            self.layer_param_blocks[base_name] = (start_idx, end_idx)

        print("All param names:", [name for name in self.layer_param_blocks])
        # print("Params from iter:", [name for name, _ in self.net.encoder.named_parameters()])

    def _is_target_param(self, param_name):
        """检查参数名是否属于目标层"""
        for name, param in self.net.named_parameters():
            if param_name == name:
                return True
        return False

    def compute_gradient_based_importance(self, train_dataloader, network):
        """基于梯度的参数重要性计算"""
        self.net.train()

        # 累积梯度平方
        gradient_importance = defaultdict(float)

        # 计算多个batch的梯度
        num_batches = min(5, len(train_dataloader))

        for i, data in enumerate(train_dataloader):
            if i >= num_batches:
                break

            src = data['src']['img']
            tgt = data['tgt']['img']
            nucp_loc = data['tgt']['NuPs']

            # 前向传播
            if network in ['NuNet', 'NuNetSWAG']:
                loss_dict = self.net.objective(src, tgt, nucp_loc)
            else:
                loss_dict = self.net.objective(src, tgt)

            loss = loss_dict['loss'].mean()

            # 反向传播
            self.net.zero_grad()
            loss.backward()

            # 收集梯度信息
            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    # 找到对应的参数块
                    block_name = None
                    if name in self.layer_param_blocks:
                        block_name = name
                    elif name.endswith('.weight'):
                        base_name = name[:-7]
                        if base_name in self.layer_param_blocks:
                            block_name = base_name
                    elif name.endswith('.bias'):
                        base_name = name[:-5]
                        if base_name in self.layer_param_blocks:
                            block_name = base_name

                    if block_name:
                        # 使用梯度的L2范数作为重要性指标
                        grad_norm = param.grad.abs().mean().item()  # 使用绝对值均值
                        gradient_importance[block_name] += grad_norm

        # 转换为普通字典并确保所有值为正
        gradient_importance = {k: max(0.0, v) for k, v in gradient_importance.items()}

        # 归一化
        total_importance = sum(gradient_importance.values())
        if total_importance > 0:
            for param_name in gradient_importance:
                gradient_importance[param_name] = gradient_importance[param_name] / total_importance * 100
        else:
            # 如果所有梯度都是0，给每个参数相同的重要性
            num_params = len(gradient_importance)
            if num_params > 0:
                equal_importance = 100.0 / num_params
                for param_name in gradient_importance:
                    gradient_importance[param_name] = equal_importance

        # 更新参数选择
        self.update_pca_save_param(gradient_importance)
        self.freeze_noncritical_params()

    def compute_pca(self):
        """使用SWAG统计信息进行PCA计算"""
        # 确保SWAG统计信息已初始化
        if not hasattr(self.net, 'cap_theta') or not hasattr(self.net, 'cap_D'):
            print("SWAG statistics not initialized. Skipping PCA.")
            return

        # 获取当前参数
        current_params = parameters_to_vector(
            list(self.net.iter_params_with_lastConvBlockInEncoder())
        ).clone().detach()

        # 计算参数块的重要性
        param_stats = {}

        for param_name, (start, end) in self.layer_param_blocks.items():
            # 提取该参数块的SWAG统计信息
            cap_theta_block = self.net.cap_theta[start:end]
            cap_theta_sq_block = self.net.cap_theta_sq[start:end]
            cap_D_block = self.net.cap_D[:, start:end]

            # 参数块的方差（从偏差矩阵计算）
            # variance = torch.mean(cap_theta_sq_block - cap_theta_block ** 2).item()
            variance = torch.norm(cap_theta_sq_block - cap_theta_block ** 2, p=2).item()

            # 综合重要性分数
            importance = variance

            param_stats[param_name] = max(0.0, importance)

        # 更新保留参数
        self.update_pca_save_param(param_stats)

        # 冻结非关键参数
        self.freeze_noncritical_params()

    def update_pca_save_param(self, param_stats):
        # # 当pca_save_prop=1时直接保留全部参数
        # if self.pca_save_count == len(self.layer_param_blocks):
        #     self.pca_save_param = {param: 1.0 for param in self.layer_param_blocks.keys()}
        #     print("保留全部参数块 (pca_save_prop=1)")
        #     return

        # 原有的选择逻辑（用于pca_save_prop<1的情况）
        non_zero_stats = {k: v for k, v in param_stats.items() if v > 0}

        if not non_zero_stats:
            print("Warning: All scores zero, saving ALL parameters")
            self.pca_save_param = {param: 1.0 for param in self.layer_param_blocks.keys()}
            return

        sorted_params = sorted(non_zero_stats.items(), key=lambda x: x[1], reverse=True)
        total_importance = sum(score for _, score in sorted_params)

        # 更新选择逻辑
        selected_params = []
        cumulative_importance = 0

        for param_name, score in sorted_params:
            if len(selected_params) < self.pca_save_count:
                selected_params.append((param_name, score))
                cumulative_importance += score

        self.pca_save_param = dict(selected_params)
        actual_percentage = (cumulative_importance / total_importance * 100) if total_importance > 0 else 0
        print(f"Selected {len(self.pca_save_param)}/{len(self.layer_param_blocks)} blocks")
        print(f"Topk important blocks: {list(self.pca_save_param.keys())[:self.pca_save_count]}")
        print(f"Cumulative importance: {actual_percentage:.2f}%")

        # 调试信息
        if actual_percentage < 0 or actual_percentage > 100:
            print(f"Debug - Total importance: {total_importance}")
            print(f"Debug - Cumulative importance: {cumulative_importance}")
            print(f"Debug - Topk scores: {[score for _, score in sorted_params[:self.pca_save_count]]}")

    def freeze_noncritical_params(self):
        """参数冻结"""
        # 获取iter_params_with_lastConvBlockInEncoder()中的参数名称集合
        encoder_param_names = set()
        for name, param in self.net.named_parameters():
            # 检查这个参数是否在iter_params_with_lastConvBlockInEncoder()返回的参数中
            for encoder_param in self.net.iter_params_with_lastConvBlockInEncoder():
                if param is encoder_param:
                    encoder_param_names.add(name)
                    break

        # 统计冻结情况
        frozen_params = 0
        total_params = 0
        frozen_names = []

        for name, param in self.net.named_parameters():
            total_params += param.numel()

            # 只处理在encoder_param_names中的参数
            if name in encoder_param_names:
                # 检查是否应该冻结
                should_freeze = True

                # 检查完整参数名
                if name in self.pca_save_param:
                    should_freeze = False
                # 检查是否是某个参数块的weight部分
                elif name.endswith('.weight'):
                    base_name = name[:-7]
                    if base_name in self.pca_save_param:
                        should_freeze = False
                # 检查是否是某个参数块的bias部分
                elif name.endswith('.bias'):
                    base_name = name[:-5]
                    if base_name in self.pca_save_param:
                        should_freeze = False

                # 对于BatchNorm参数，始终保持可训练
                if 'bn' in name.lower() or 'norm' in name.lower():
                    should_freeze = False

                if should_freeze:
                    param.requires_grad = False
                    frozen_params += param.numel()
                    frozen_names.append(name)
                else:
                    param.requires_grad = True

        print(f"Frozen {frozen_params}/{total_params} parameters ({frozen_params / total_params:.2%})")
        print(f"Frozen {len(frozen_names)} parameter blocks from encoder")
        if len(frozen_names) <= 10:
            print(f"Frozen parameters: {frozen_names}")

    def post_hoc(self,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 name: str = None,
                 network: str = None,
                 excel_save_path: str = None,
                 logger: SummaryWriter = None,
                 verbose=2):

        print('Post Hoc...')
        start = time.perf_counter()
        self.net.fit(train_dataloader)
        end = time.perf_counter()
        print('Post Hoc time: {}s'.format(end - start))

        print('Testing...')
        self.net.eval()
        metric_test = MetricTest()
        with torch.no_grad():
            for data in test_dataloader:
                case_no, slc_idx, resolution = data['case_no'].item(), data['slice'], data['resolution'].item()
                src, tgt = data['src'][0].cuda().float(), data['tgt'][0].cuda().float()
                src_seg, tgt_seg = data['src_seg'][0].cuda().float(), data['tgt_seg'][0].cuda().float()
                nucp_loc_src = data['src_NuPs'][0].cuda().float()
                nucp_loc_tgt = data['tgt_NuPs'][0].cuda().float()

                results_t = self.net.test_laplace(src, tgt, nucp_loc_tgt)
                resultt_s = self.net.test_laplace(tgt, src, nucp_loc_src)

                phis_t = results_t[0]
                phit_s = resultt_s[0]

                warped_src_seg = self.net.transformer(src_seg,
                                                      phis_t,
                                                      mode='nearest')
                warped_tgt_seg = self.net.transformer(tgt_seg,
                                                      phit_s,
                                                      mode='nearest')

                metric_test.testMetrics(src_seg.int(), warped_src_seg.int(),
                                        tgt_seg.int(), warped_tgt_seg.int(),
                                        resolution, case_no, slc_idx)
                metric_test.testFlow(phis_t, phit_s, case_no)

        mean = metric_test.mean()
        if verbose >= 2:
            metric_test.saveAsExcel(network, name, excel_save_path, logger)
        if verbose >= 1:
            metric_test.output()
        return mean, metric_test.details

    def continue_train(self,
                       network,
                       optimizer,
                       train_dataloader: DataLoader,
                       validation_dataloader: DataLoader,
                       save_checkpoint,
                       earlystop: EarlyStopping,
                       logger: SummaryWriter,
                       start_epoch=0,
                       max_epoch=1000,
                       pca_enable=False,
                       pca_save_prop=1,
                       pca_update_frequency=10,
                       pca_warmup_epochs=50):

        logs = []
        earlystop.on_train_begin()

        for e in range(start_epoch, max_epoch * 2):
            start = time.perf_counter()

            # 训练
            self.net.train()
            train_loss_dict = self.trainIter(network, train_dataloader, optimizer)

            # 验证
            self.net.eval()
            validation_dice = self.validationIter(network, validation_dataloader)

            train_loss_dict['Dice'] = validation_dice

            # 保存检查点
            if save_checkpoint:
                save_checkpoint(self.net, e + 1, optimizer=optimizer)

            # 记录日志到Tensorboard
            for key in train_loss_dict:
                logger.add_scalar(key, train_loss_dict[key], e + 1)
            logger.add_scalar('Dice', validation_dice, e + 1)

            logs.append(list(train_loss_dict.values()))

            end = time.perf_counter()

            train_loss_mean_str = ['%s: %f' % (key, np.round(value, 5)) for key, value in train_loss_dict.items()]
            print('Epoch %d(%.2fs): ' % (e + 1, (end - start)), train_loss_mean_str)

            # 每500个epoch保存一次日志到CSV文件
            if (e + 1) % 500 == 0:
                logs = np.round(np.array(logs), 5).tolist()
                with open(os.path.join(logger.log_dir, 'log.csv'), 'a', encoding='utf-8', newline='') as fp:
                    writer = csv.writer(fp)
                    if fp.tell() == 0:
                        title = list(train_loss_dict.keys())
                        writer.writerow(title)
                    writer.writerows(logs)
                logs = []

            # 早停检查
            if earlystop.on_epoch_end(e + 1, validation_dice, self.net, train_loss_dict) and e >= max_epoch:
                if logs != []:
                    logs = np.round(np.array(logs), 5).tolist()
                    with open(os.path.join(logger.log_dir, 'log.csv'), 'a', encoding='utf-8', newline='') as fp:
                        writer = csv.writer(fp)
                        if fp.tell() == 0:
                            title = list(train_loss_dict.keys())
                            writer.writerow(title)
                        writer.writerows(logs)
                break

        return earlystop.best

    def SWAG_continue_train(self,
                            network,
                            optimizer,
                            train_dataloader: DataLoader,
                            validation_dataloader: DataLoader,
                            save_checkpoint,
                            earlystop: EarlyStopping,
                            logger: SummaryWriter,
                            start_epoch=0,
                            max_epoch=1000,
                            c=1,
                            k=1,
                            pca_enable=False,
                            pca_save_prop=1,
                            pca_update_frequency=10):  # 增加更新频率参数

        # SWAG初始化
        iter_param = self.net.iter_params_with_lastConvBlockInEncoder()
        cap_theta = parameters_to_vector(iter_param).clone().detach()
        cap_theta_sq = cap_theta.clone() ** 2
        cap_D = (cap_theta_sq - cap_theta ** 2).unsqueeze(0)

        setattr(self.net, 'cap_theta', cap_theta)
        setattr(self.net, 'cap_theta_sq', cap_theta_sq)
        setattr(self.net, 'cap_D', cap_D)

        # 构建参数映射
        if pca_enable:
            self.build_layer_param_mapping()
            total_params = len(self.layer_param_blocks)
            self.pca_save_count = int(total_params * pca_save_prop)
            print(f"PCA enabled: Total blocks: {total_params}, Saving: {self.pca_save_count}")
            # print(f"Layer name: {self.layer_param_blocks}")

        logs = []
        n = 0
        earlystop.on_train_begin()

        for e in range(start_epoch, max_epoch * 2):
            start = time.perf_counter()

            # 训练
            self.net.train()
            train_loss_dict = self.trainIter(network, train_dataloader, optimizer)
            self.net.eval()

            # SWAG参数收集
            if e % c == 0:
                n += 1
                self.net.collect_model(n, k)

            # 验证
            if e < start_epoch + c * k:
                validation_dice = self.validationIter(network, validation_dataloader)
            else:
                # 使用SWAG均值参数进行验证
                iter_param = self.net.iter_params_with_lastConvBlockInEncoder()
                params_back = parameters_to_vector(iter_param).clone().detach()

                iter_param = self.net.iter_params_with_lastConvBlockInEncoder()
                vector_to_parameters(self.net.cap_theta, iter_param)

                validation_dice = self.validationIter(network, validation_dataloader)

                # PCA更新（在足够的训练后才开始）
                if pca_enable and e % pca_update_frequency == 0 and e > start_epoch + 50:
                    print(f"Updating PCA at epoch {e + 1}")
                    self.compute_pca()

                # 恢复原始参数
                iter_param = self.net.iter_params_with_lastConvBlockInEncoder()
                vector_to_parameters(params_back, iter_param)

            train_loss_dict['Dice'] = validation_dice

            # 保存检查点
            if save_checkpoint:
                save_checkpoint(self.net, e + 1, optimizer=optimizer)

            # 记录日志
            for key in train_loss_dict:
                logger.add_scalar(key, train_loss_dict[key], e + 1)
            logger.add_scalar('Dice', validation_dice, e + 1)

            logs.append(list(train_loss_dict.values()))

            end = time.perf_counter()

            train_loss_mean_str = ['%s: %f' % (key, np.round(value, 5)) for key, value in train_loss_dict.items()]
            print('Epoch %d(%.2fs): ' % (e + 1, (end - start)), train_loss_mean_str)

            # 定期保存日志
            if (e + 1) % 500 == 0:
                logs = np.round(np.array(logs), 5).tolist()
                with open(os.path.join(logger.log_dir, 'log.csv'), 'a', encoding='utf-8', newline='') as fp:
                    writer = csv.writer(fp)
                    if fp.tell() == 0:
                        title = list(train_loss_dict.keys())
                        writer.writerow(title)
                    writer.writerows(logs)
                logs = []

            # 早停检查
            if earlystop.on_epoch_end(e + 1, validation_dice, self.net, train_loss_dict) and e >= max_epoch:
                if logs != []:
                    logs = np.round(np.array(logs), 5).tolist()
                    with open(os.path.join(logger.log_dir, 'log.csv'), 'a', encoding='utf-8', newline='') as fp:
                        writer = csv.writer(fp)
                        if fp.tell() == 0:
                            title = list(train_loss_dict.keys())
                            writer.writerow(title)
                        writer.writerows(logs)
                break

        return earlystop.best

    # def SWAG_test(self,
    #               test_dataloader: DataLoader,
    #               name: str = None,
    #               network: str = None,
    #               excel_save_path: str = None,
    #               logger: SummaryWriter = None,
    #               verbose=2,
    #               k=0,
    #               s=1):
    #
    #     self.net.eval()
    #     metric_test = MetricTest()
    #
    #     with torch.no_grad():
    #         for data in test_dataloader:
    #             case_no, slc_idx, resolution = data['case_no'].item(), data['slice'], data['resolution'].item()
    #             src, tgt = data['src'][0].cuda().float(), data['tgt'][0].cuda().float()
    #             src_seg, tgt_seg = data['src_seg'][0].cuda().float(), data['tgt_seg'][0].cuda().float()
    #             nucp_loc_src = data['src_NuPs'][0].cuda().float()
    #             nucp_loc_tgt = data['tgt_NuPs'][0].cuda().float()
    #
    #             # 确保网络回到eval模式
    #             self.net.eval()
    #
    #             result_t = self.net.SWAG_test(src, tgt, nucp_loc_tgt, k, s)
    #             result_s = self.net.SWAG_test(tgt, src, nucp_loc_src, k, s)
    #
    #             phis_t = result_t[0]
    #             phit_s = result_s[0]
    #
    #             warped_src_seg = self.net.transformer(src_seg, phis_t, mode='nearest')
    #             warped_tgt_seg = self.net.transformer(tgt_seg, phit_s, mode='nearest')
    #
    #             metric_test.testMetrics(src_seg.int(), warped_src_seg.int(), tgt_seg.int(), warped_tgt_seg.int(),
    #                                     resolution, case_no, slc_idx)
    #             metric_test.testFlow(phis_t, phit_s, case_no)
    #
    #     mean = metric_test.mean()
    #     if verbose >= 2:
    #         metric_test.saveAsExcel(network, name, excel_save_path, logger)
    #     if verbose >= 1:
    #         metric_test.output()
    #     return mean, metric_test.details

    def SWAG_test(self,
                  test_dataloader: DataLoader,
                  name: str = None,
                  network: str = None,
                  excel_save_path: str = None,
                  logger: SummaryWriter = None,
                  verbose=2,
                  k=0,
                  s=1):

        self.net.eval()
        metric_test = MetricTest()

        # 用于存储所有样本的方差均值
        all_phis_t_var_means = []
        all_phit_s_var_means = []

        with torch.no_grad():
            for data in test_dataloader:
                case_no, slc_idx, resolution = data['case_no'].item(), data['slice'], data['resolution'].item()
                src, tgt = data['src'][0].cuda().float(), data['tgt'][0].cuda().float()
                src_seg, tgt_seg = data['src_seg'][0].cuda().float(), data['tgt_seg'][0].cuda().float()
                nucp_loc_src = data['src_NuPs'][0].cuda().float()
                nucp_loc_tgt = data['tgt_NuPs'][0].cuda().float()

                # 确保网络回到eval模式
                self.net.eval()

                # 进行s次test，计算方差
                phis_t_list = []
                phit_s_list = []

                for i in range(s):
                    # 执行SWAG测试
                    result_t = self.net.SWAG_test(src, tgt, nucp_loc_tgt, k, s)
                    result_s = self.net.SWAG_test(tgt, src, nucp_loc_src, k, s)

                    phis_t_list.append(result_t[0])
                    phit_s_list.append(result_s[0])

                # 计算方差并取均值
                phis_t_stack = torch.stack(phis_t_list, dim=0)
                phit_s_stack = torch.stack(phit_s_list, dim=0)

                phis_t_var = torch.var(phis_t_stack, dim=0, unbiased=False)  # 使用无偏估计
                phit_s_var = torch.var(phit_s_stack, dim=0, unbiased=False)  # 使用无偏估计

                # 计算每个像素点的方差，获取更详细的统计信息
                phis_t_var_mean = phis_t_var.mean().item()
                phit_s_var_mean = phit_s_var.mean().item()

                # 收集所有样本的方差统计信息
                all_phis_t_var_means.append(phis_t_var_mean)
                all_phit_s_var_means.append(phit_s_var_mean)

                # 输出当前样本的方差统计（提高精度到12位小数）
                # print(f"\n样本 {case_no} 像素点方差统计:")
                # print(f"源到目标方差: 均值={phis_t_var_mean:.12f}, 中位数={phis_t_var_median:.12f}, 最大值={phis_t_var_max:.12f}")
                # print(f"目标到源方差: 均值={phit_s_var_mean:.12f}, 中位数={phit_s_var_median:.12f}, 最大值={phit_s_var_max:.12f}")
                # print(f"平均方差: {(phis_t_var_mean + phit_s_var_mean) / 2.0:.12f}")

                # 取第一次测试结果作为最终形变场
                phis_t = phis_t_list[0]
                phit_s = phit_s_list[0]

                warped_src_seg = self.net.transformer(src_seg, phis_t, mode='nearest')
                warped_tgt_seg = self.net.transformer(tgt_seg, phit_s, mode='nearest')

                metric_test.testMetrics(src_seg.int(), warped_src_seg.int(), tgt_seg.int(), warped_tgt_seg.int(),
                                        resolution, case_no, slc_idx)
                metric_test.testFlow(phis_t, phit_s, case_no)

        # 计算并输出所有样本的平均像素点方差
        if all_phis_t_var_means and all_phit_s_var_means:
            avg_phis_var = np.mean(all_phis_t_var_means)
            avg_phit_var = np.mean(all_phit_s_var_means)
            avg_overall_var = (avg_phis_var + avg_phit_var) / 2.0
            print(f"\n像素点平均方差统计:")
            print(f"源到目标平均方差: {avg_phis_var:.12f}")
            print(f"目标到源平均方差: {avg_phit_var:.12f}")
            print(f"总体平均方差: {avg_overall_var:.12f}\n")

        mean = metric_test.mean()
        if verbose >= 2:
            metric_test.saveAsExcel(network, name, excel_save_path, logger)
        if verbose >= 1:
            metric_test.output()
        return mean, metric_test.details

    def train(self,
              network,
              train_dataloader: DataLoader,
              validation_dataloader: DataLoader,
              save_checkpoint,
              earlystop: EarlyStopping,
              logger: SummaryWriter,
              start_epoch=0,
              max_epoch=1000,
              lr=1e-4):

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        # 计算并输出iter_params_with_lastConvBlockInEncoder的参数总量
        iter_param = self.net.iter_params_with_lastConvBlockInEncoder()
        total_params = sum(p.numel() for p in iter_param)
        print(f"总参数量 (encoder最后卷积块): {total_params}")
        
        # 计算并输出全网络参数总量
        all_params = sum(p.numel() for p in self.net.parameters())
        print(f"总参数量 (全网络): {all_params}")

        logs = []
        earlystop.on_train_begin()
        for e in range(start_epoch, max_epoch * 2):
            start = time.perf_counter()

            # learning
            self.net.train()
            train_loss_dict = self.trainIter(network, train_dataloader, optimizer)
            # validation
            self.net.eval()
            validation_dice = self.validationIter(network, validation_dataloader)

            train_loss_dict['Dice'] = validation_dice

            # save checkpoint
            if save_checkpoint:
                save_checkpoint(self.net, e + 1, optimizer=optimizer)

            # Logger for Tensorboard
            for key in train_loss_dict:
                logger.add_scalar(key, train_loss_dict[key], e + 1)
            logger.add_scalar('Dice', validation_dice, e + 1)

            logs.append(list(train_loss_dict.values()))

            end = time.perf_counter()

            train_loss_mean_str = ['%s: %f' % (key, np.round(value, 5)) for key, value in train_loss_dict.items()]
            print('Epoch %d(%.2fs): ' % (e + 1, (end - start)), train_loss_mean_str)

            if (e + 1) % 500 == 0:
                # Save to excel file
                logs = np.round(np.array(logs), 5).tolist()
                with open(os.path.join(logger.log_dir, 'log.csv'), 'a', encoding='utf-8', newline='') as fp:
                    writer = csv.writer(fp)
                    if fp.tell() == 0:
                        title = list(train_loss_dict.keys())
                        writer.writerow(title)
                    writer.writerows(logs)
                logs = []

            # early stop
            if earlystop.on_epoch_end(e + 1, validation_dice, self.net, train_loss_dict) and e >= max_epoch:
                if logs != []:
                    logs = np.round(np.array(logs), 5).tolist()
                    with open(os.path.join(logger.log_dir, 'log.csv'), 'a', encoding='utf-8', newline='') as fp:
                        writer = csv.writer(fp)
                        if fp.tell() == 0:
                            title = list(train_loss_dict.keys())
                            writer.writerow(title)
                        writer.writerows(logs)
                break

        return earlystop.best

    def trainIter(self,
                  network,
                  dataloader: DataLoader,
                  optimizer: torch.optim.Optimizer) -> dict:

        train_loss_dict = {}
        for data in dataloader:
            src = data['src']['img']
            tgt = data['tgt']['img']
            nucp_loc = data['tgt']['NuPs']

            optimizer.zero_grad()

            if network in ['NuNet', 'NuNetSWAG']:
                loss_dict = self.net.objective(src, tgt, nucp_loc)
            else:
                loss_dict = self.net.objective(src, tgt)

            loss = loss_dict['loss'].mean()
            loss.backward()
            optimizer.step()

            for key, values in loss_dict.items():
                if key not in train_loss_dict:
                    train_loss_dict[key] = [values.mean().item()]
                else:
                    train_loss_dict[key].append(values.mean().item())

        for key in train_loss_dict:
            train_loss_dict[key] = np.mean(train_loss_dict[key])

        return train_loss_dict

    def validationIter(self, network, dataloader: DataLoader):
        dice_list = []
        with torch.no_grad():
            dice_estimator = DiceCoefficientAll()
            for data in dataloader:
                src = data['src'][0].cuda().float()
                tgt = data['tgt'][0].cuda().float()
                # regard all types of segmeant as one
                nucp_loc = data['tgt_NuPs'][0].cuda().float()
                src_seg = data['src_seg'][0].cuda().float()
                tgt_seg = data['tgt_seg'][0].cuda().int()
                case_no = data['case_no'].item()

                if network in ['NuNet', 'NuNetSWAG']:
                    result = self.net.test(src, tgt, nucp_loc)
                else:
                    result = self.net.test(src, tgt)

                phi = result[0]
                warped_src_seg = self.net.transformer(src_seg, phi, mode='nearest')

                dice = dice_estimator(tgt_seg, warped_src_seg.int(), case_no)
                dice_list.append(dice)
            # statistics
            dice_tensor = torch.hstack(dice_list)
            return dice_tensor.mean().item()

    # def test(self,
    #          dataloader: DataLoader,
    #          name: str = None,
    #          network: str = None,
    #          excel_save_path: str = None,
    #          logger: SummaryWriter = None,
    #          verbose=2):

    #     self.net.eval()
    #     metric_test = MetricTest()

    #     with torch.no_grad():
    #         for data in dataloader:
    #             case_no, slc_idx, resolution = data['case_no'].item(), data['slice'], data['resolution'].item()
    #             src, tgt = data['src'][0].cuda().float(), data['tgt'][0].cuda().float()
    #             src_seg, tgt_seg = data['src_seg'][0].cuda().float(), data['tgt_seg'][0].cuda().float()
    #             nucp_loc_src = data['src_NuPs'][0].cuda().float()
    #             nucp_loc_tgt = data['tgt_NuPs'][0].cuda().float()
                
    #             # 确保网络回到eval模式
    #             self.net.eval()

    #             # 判断是否使用生成Mask
    #             if network in ['NuNet', 'NuNetSWAG']:
    #                 results_t = self.net.test(src, tgt, nucp_loc_tgt)
    #                 resultt_s = self.net.test(tgt, src, nucp_loc_src)
    #             else:
    #                 results_t = self.net.test(src, tgt)
    #                 resultt_s = self.net.test(tgt, src)

    #             phis_t = results_t[0]
    #             phit_s = resultt_s[0]

    #             warped_src_seg = self.net.transformer(src_seg, phis_t, mode='nearest')
    #             warped_tgt_seg = self.net.transformer(tgt_seg, phit_s, mode='nearest')

    #             metric_test.testMetrics(src_seg.int(), warped_src_seg.int(), tgt_seg.int(), warped_tgt_seg.int(),
    #                                     resolution, case_no, slc_idx)
    #             metric_test.testFlow(phis_t, phit_s, case_no)

    #     mean = metric_test.mean()
    #     if verbose >= 2:
    #         metric_test.saveAsExcel(network, name, excel_save_path, logger)
    #     if verbose >= 1:
    #         metric_test.output()
    #     return mean, metric_test.details

    def test(self,
             dataloader: DataLoader,
             name: str = None,
             network: str = None,
             excel_save_path: str = None,
             logger: SummaryWriter = None,
             verbose=2):

        self.net.eval()
        metric_test = MetricTest()
        
        # 用于存储所有样本的方差均值
        all_phis_t_var_means = []
        all_phit_s_var_means = []

        with torch.no_grad():
            for data in dataloader:
                case_no, slc_idx, resolution = data['case_no'].item(), data['slice'], data['resolution'].item()
                src, tgt = data['src'][0].cuda().float(), data['tgt'][0].cuda().float()
                src_seg, tgt_seg = data['src_seg'][0].cuda().float(), data['tgt_seg'][0].cuda().float()
                nucp_loc_src = data['src_NuPs'][0].cuda().float()
                nucp_loc_tgt = data['tgt_NuPs'][0].cuda().float()
                
                # 确保网络回到eval模式
                self.net.eval()

                # 进行20次test，计算方差
                phis_t_list = []
                phit_s_list = []
                
                for i in range(20):
                    # 判断是否使用生成Mask
                    if network in ['NuNet', 'NuNetSWAG']:
                        results_t = self.net.test(src, tgt, nucp_loc_tgt)
                        resultt_s = self.net.test(tgt, src, nucp_loc_src)
                    else:
                        results_t = self.net.test(src, tgt)
                        resultt_s = self.net.test(tgt, src)
                    
                    phis_t_list.append(results_t[0])
                    phit_s_list.append(resultt_s[0])
                
                # 计算方差并取均值
                phis_t_stack = torch.stack(phis_t_list, dim=0)
                phit_s_stack = torch.stack(phit_s_list, dim=0)
                
                phis_t_var = torch.var(phis_t_stack, dim=0)
                phit_s_var = torch.var(phit_s_stack, dim=0)
                
                # 计算每个像素点的方差，然后求平均
                phis_t_var_mean = phis_t_var.mean().item()
                phit_s_var_mean = phit_s_var.mean().item()
                
                # 收集所有样本的方差均值
                all_phis_t_var_means.append(phis_t_var_mean)
                all_phit_s_var_means.append(phit_s_var_mean)
                
                # 输出当前样本的方差统计
                print(f"\n样本 {case_no} 像素点方差统计:")
                print(f"源到目标方差: {phis_t_var_mean:.6f}")
                print(f"目标到源方差: {phit_s_var_mean:.6f}")
                print(f"平均方差: {(phis_t_var_mean + phit_s_var_mean) / 2.0:.6f}")
                
                # 取第一次测试结果作为最终形变场
                phis_t = phis_t_list[0]
                phit_s = phit_s_list[0]

                warped_src_seg = self.net.transformer(src_seg, phis_t, mode='nearest')
                warped_tgt_seg = self.net.transformer(tgt_seg, phit_s, mode='nearest')

                metric_test.testMetrics(src_seg.int(), warped_src_seg.int(), tgt_seg.int(), warped_tgt_seg.int(),
                                        resolution, case_no, slc_idx)
                metric_test.testFlow(phis_t, phit_s, case_no)

        # 计算并输出所有样本的平均像素点方差
        if all_phis_t_var_means and all_phit_s_var_means:
            avg_phis_var = np.mean(all_phis_t_var_means)
            avg_phit_var = np.mean(all_phit_s_var_means)
            avg_overall_var = (avg_phis_var + avg_phit_var) / 2.0
            print(f"\n像素点平均方差统计:")
            print(f"源到目标平均方差: {avg_phis_var:.6f}")
            print(f"目标到源平均方差: {avg_phit_var:.6f}")
            print(f"总体平均方差: {avg_overall_var:.6f}\n")

        mean = metric_test.mean()
        if verbose >= 2:
            metric_test.saveAsExcel(network, name, excel_save_path, logger)
        if verbose >= 1:
            metric_test.output()
        return mean, metric_test.details

    def hyperOpt(self,
                 hyperparams,
                 load_checkpoint,
                 n_trials,
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 earlystop: EarlyStopping,
                 logger: SummaryWriter,
                 max_epoch=500,
                 lr=1e-4):
        def objective(trial: optuna.Trial):
            hyperparams = trial.study.user_attrs['hyperparams']
            params_instance = ParamsAll(trial, hyperparams)
            print(params_instance)
            load_checkpoint(self.net, 0)
            self.net.setHyperparam(**params_instance)

            self.train(train_dataloader,
                       validation_dataloader,
                       None,
                       earlystop,
                       None,
                       0,
                       max_epoch,
                       lr,
                       v_step=0,
                       verbose=0)

            res, _ = self.test(test_dataloader, verbose=0)
            print(res)
            return 1 - res['mean']

        self.net.train()
        study = optuna.create_study()
        study.set_user_attr('hyperparams', hyperparams)
        study.optimize(objective, n_trials, n_jobs=1)
        print(study.best_params)
        return study.best_params

    def speedTest(self, dataloader: DataLoader, device_type='gpu'):
        self.net.eval()
        case_time = []
        slice_time = []
        if device_type is 'cpu':
            self.net.cpu()
        with torch.no_grad():
            for data in dataloader:
                if device_type is 'gpu':
                    src = data['src'][0].cuda().float()
                    tgt = data['tgt'][0].cuda().float()
                else:
                    src = data['src'][0].cpu().float()
                    tgt = data['tgt'][0].cpu().float()
                torch.cuda.synchronize()
                start = time.time()
                result = self.net.test(src, tgt)
                torch.cuda.synchronize()
                end = time.time()
                case_time.append(end - start)

                torch.cuda.synchronize()
                for i in range(src.size()[0]):
                    start = time.time()
                    result = self.net.test(src[i:i + 1], tgt[i:i + 1])
                    torch.cuda.synchronize()
                    end = time.time()
                    slice_time.append(end - start)
        case_res = {'mean': np.mean(case_time), 'std': np.std(case_time)}
        slice_res = {'mean': np.mean(slice_time), 'std': np.std(slice_time)}
        print(device_type)
        print('case', '%.3f(%.3f)' % (case_res['mean'], case_res['std']))
        print('slice', '%.3f(%.3f)' % (slice_res['mean'], slice_res['std']))

    def estimate(self, case_data: torch.Tensor):
        self.net.eval()
        with torch.no_grad():
            src = case_data['src'].cuda().float()
            tgt = case_data['tgt'].cuda().float()
            src_seg = case_data['src_seg'].cuda().float()
            tgt_seg = case_data['tgt_seg'].cuda().float()
            slc_idx = case_data['slice']
            if "NU" in type(self.net).__name__:
                result = self.net.testForDraw(src, tgt, tgt_seg)
            else:
                result = self.net.test(src, tgt)

            phi = result[0]
            if len(result) == 3:
                cp_loc = None
            else:
                cp_loc = result[3]
            warped_src = self.net.transformer(src, phi)
            warped_src_seg = self.net.transformer(src_seg, phi, mode='nearest')

            res = {
                'src': src.cpu().numpy()[:, 0, :, :],
                'tgt': tgt.cpu().numpy()[:, 0, :, :],
                'src_seg': src_seg.cpu().numpy()[:, 0, :, :],
                'tgt_seg': tgt_seg.cpu().numpy()[:, 0, :, :],
                'phi': phi.cpu().numpy(),
                'warped_src': warped_src.cpu().numpy()[:, 0, :, :],
                'warped_src_seg': warped_src_seg.cpu().numpy()[:, 0, :, :],
                'slc_idx': slc_idx,
                'cp_loc': cp_loc
            }
            return res

    def cuda(self):
        self.net.cuda()