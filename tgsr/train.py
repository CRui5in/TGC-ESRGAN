import argparse
import datetime
import logging
import math
import os
import random
import time
import torch
import numpy as np
from os import path as osp
from tqdm import tqdm

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed, scandir)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse_options as parse_options_basicsr, copy_opt_file

# 导入我们的TGSRDataset以确保它被注册
from tgsr.data.tgsr_dataset import TGSRDataset

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import cv2

# 使用我们的改进版tensor2img替换原版
from tgsr.utils.visualization_utils import improved_tensor2img as tensor2img

class EnhancedMessageLogger(MessageLogger):
    """增强版消息日志器，改进TensorBoard的分组和可视化"""
    
    def __init__(self, opt, start_iter=0, tb_logger=None):
        super().__init__(opt, start_iter, tb_logger)
        # 添加一个变量来存储当前迭代数，用于解决lint错误
        self.current_iter = start_iter
        
        # 如果是恢复训练，确保TensorBoard从正确的迭代次数开始记录
        if tb_logger is not None and hasattr(tb_logger, '_get_file_writer'):
            writer = tb_logger._get_file_writer()
            if writer is not None and hasattr(writer, 'add_summary'):
                # 添加一个标记，表示这是恢复的训练
                tb_logger.add_text('resume_info', f'从迭代次数 {start_iter} 恢复训练', start_iter)
            
    def __call__(self, log_vars):
        """
        覆盖原方法，改进TensorBoard记录，确保train/val正确分组
        Args:
            log_vars (dict): 包含所有日志变量的字典。
        """
        # 文本日志
        # 使用同样的文本日志格式
        message = f'[{log_vars.pop("epoch"):.4f}][{log_vars.pop("iter"):8d}]'
        
        # 添加学习率
        if 'lrs' in log_vars:
            lrs = log_vars.pop('lrs')
            message += f'[{", ".join([f"{lr:.3e}" for lr in lrs])}]'
            
            # 记录学习率到TensorBoard
            if self.tb_logger is not None:
                for i, lr in enumerate(lrs):
                    self.tb_logger.add_scalar(f'train/lr/lr_{i}', lr, self.current_iter)
            
        # 添加其他训练日志
        for k, v in log_vars.items():
            # 只显示不带train/val前缀的日志名称
            display_key = k.split('/')[-1] if '/' in k else k
            message += f'{display_key}: {v:.4e} '
            
        # 打印日志
        self.logger.info(message)
        
        # TensorBoard日志
        if self.tb_logger is not None:
            # 确保对每个损失都记录其值，并始终将其添加到TensorBoard
            loss_keys_to_ensure = [
                'l_pix', 'l_percep', 'l_style', 'l_g_gan', 'l_total',
                'l_d_real', 'l_d_fake'
            ]
            
            # 整理日志，按照前缀分组
            train_losses = {}
            val_metrics = {}
            train_metrics = {}
            other_metrics = {}
            
            for k, v in log_vars.items():
                if k.startswith('train/'):
                    # 已有train/前缀的，每个迭代都记录
                    self.tb_logger.add_scalar(k, v, self.current_iter)
                    
                    # 提取类别，例如train/losses/l_pix -> losses
                    parts = k.split('/')
                    if len(parts) >= 2:
                        category = parts[1]
                        name = parts[-1] if len(parts) >= 3 else parts[-1]
                        
                        # 按类别分组
                        if category not in train_losses:
                            train_losses[category] = {}
                        train_losses[category][name] = v
                        
                elif k.startswith('val/'):
                    # 已有val/前缀的，每个迭代都记录 
                    self.tb_logger.add_scalar(k, v, self.current_iter)
                    
                    # 提取类别
                    parts = k.split('/')
                    if len(parts) >= 2:
                        category = parts[1]
                        name = parts[-1] if len(parts) >= 3 else parts[-1]
                        
                        # 按类别分组
                        if category not in val_metrics:
                            val_metrics[category] = {}
                        val_metrics[category][name] = v
                
                elif 'psnr' in k.lower() or 'ssim' in k.lower():
                    # 验证指标
                    key = f'val/metrics/{k}'
                    self.tb_logger.add_scalar(key, v, self.current_iter)
                    val_metrics.setdefault('metrics', {})[k] = v
                
                elif any(loss_key in k for loss_key in loss_keys_to_ensure) or 'loss' in k.lower():
                    # 处理所有损失类型 - 确保每个迭代都记录
                    key = f'train/losses/{k}'
                    self.tb_logger.add_scalar(key, v, self.current_iter)
                    train_losses.setdefault('losses', {})[k] = v
                
                else:
                    # 其他指标
                    key = f'other/{k}'
                    self.tb_logger.add_scalar(key, v, self.current_iter)
                    other_metrics[k] = v
            
            # 添加总损失到training/losses组
            if 'losses' in train_losses and len(train_losses['losses']) > 0:
                total_loss = sum(train_losses['losses'].values())
                self.tb_logger.add_scalar('train/losses/total', total_loss, self.current_iter)
        
        # 更新当前迭代次数
        self.current_iter += 1


def parse_options():
    """命令行参数解析"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='路径到配置文件')
    parser.add_argument('--launcher', type=str, default='none', help='启动器 {none, pytorch, slurm}')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--memory_opt', action='store_true', help='启用内存优化')
    parser.add_argument('--reuse_exp_dir', action='store_true', help='继续使用已存在的实验目录')
    args = parser.parse_args()
    
    # 手动解析YAML配置文件
    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    
    with open(args.opt, mode='r', encoding='utf-8') as f:
        opt = yaml.load(f, Loader=Loader)
    
    # 分布式设置
    if args.launcher == 'none':
        opt['dist'] = False
        print('禁用分布式训练')
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('初始化PyTorch分布式训练环境...')
    
    # 内存优化
    if args.memory_opt:
        opt['train']['memory_efficient'] = True
    
    opt['rank'], opt['world_size'] = get_dist_info()
    
    # 随机种子
    seed = opt.get('manual_seed')
    if seed is not None:
        set_random_seed(seed + opt['rank'])
    
    # 设置GPU数量
    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()
    
    # 设置路径
    opt['is_train'] = True
    
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 设置实验目录
    if opt['is_train']:
        experiments_root = opt['path'].get('experiments_root', 
                                         osp.join(root_path, 'experiments', opt['name']))
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
    
    # 设置数据集路径
    for phase, dataset in opt['datasets'].items():
        # 对于多个数据集，例如val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])
    
    # 路径设置
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    
    # 调试模式
    if args.debug:
        opt['val']['val_freq'] = 8
        opt['logger']['print_freq'] = 1
        opt['logger']['save_checkpoint_freq'] = 4
    
    return opt, args


def init_loggers(opt, args=None):
    """初始化各种日志器"""
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    # 初始化tensorboard日志器
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_dir = osp.join('tb_logger', opt['name'])
        os.makedirs(tb_dir, exist_ok=True)
        log_dir = tb_dir
        
        logger.info(f'TensorBoard日志将保存到: {log_dir}')
        
        # 在--reuse-exp-dir模式下，使用追加模式初始化TensorBoard
        if args and args.reuse_exp_dir and os.path.exists(log_dir):
            logger.info(f'在--reuse-exp-dir模式下，尝试追加到现有TensorBoard日志')
            try:
                # 查找现有的事件文件
                event_files = []
                for root, _, files in os.walk(log_dir):
                    for file in files:
                        if file.startswith('events.out.tfevents.'):
                            event_files.append(os.path.join(root, file))
                
                if event_files:
                    logger.info(f'找到{len(event_files)}个现有TensorBoard事件文件，使用最新的文件')
                    # 按文件修改时间排序，获取最新的事件文件
                    event_files.sort(key=os.path.getmtime, reverse=True)
                    latest_event_file = event_files[0]
                    logger.info(f'最新的事件文件是: {latest_event_file}')
                    
                    # 提取这个文件的全局步数，以便我们能够继续记录
                    # 这里我们不直接提取，而是通过当前迭代次数来判断
            except Exception as e:
                logger.warning(f'检查TensorBoard事件文件时出错: {e}，将创建新的日志')
        
        # 创建TensorBoard日志器
        tb_logger = init_tb_logger(log_dir=log_dir)
        
        # 添加配置信息
        tb_logger.add_text('config', dict2str(opt).replace('\n', '  \n'), 0)
        
        # 添加环境信息
        env_info = get_env_info()
        tb_logger.add_text('env_info', str(env_info).replace('\n', '  \n'), 0)
        
        # 记录主要超参数
        if 'network_g' in opt:
            if 'type' in opt['network_g']:
                tb_logger.add_text('network_g', opt['network_g']['type'], 0)
            
            # 定义TensorBoard布局
            layout = {
                "训练与验证": {
                    "损失函数": {
                        "训练损失": ["Multiline", ["Train/losses/l_pix", "Train/losses/l_percep", 
                                          "Train/losses/l_style", "Train/losses/l_g_gan", 
                                          "Train/losses/l_total"]],
                        "判别器损失": ["Multiline", ["Train/losses/l_d_real", "Train/losses/l_d_fake"]]
                    },
                    "训练学习率": {
                        "学习率": ["Multiline", ["Train/learningrate/lr_0", "Train/learningrate/lr_1"]]
                    },
                    "验证指标": {
                        "指标": ["Multiline", ["Val/metrics/psnr", "Val/metrics/ssim"]]
                    }
                },
                "可视化结果": {
                    "训练图像": {
                        "比较图": ["Images", ["Train/images/comparison"]],
                        "单独图像": ["Images", ["Train/images/LQ", "Train/images/SR", "Train/images/GT"]]
                    },
                    "验证图像": {
                        "验证图像": ["Images", ["Val/images/*"]]
                    },
                    "注意力热力图": {
                        "注意力图": ["Images", ["Train/attention/*"]],
                        "热力图": ["Images", ["Train/heatmap/*"]]
                    }
                }
            }
            
            # 尝试添加自定义标量布局
            try:
                # 获取TensorBoard的SummaryWriter实例
                writer = None
                
                # 尝试不同的属性名
                for attr_name in ['_writer', 'writer', 'file_writer', 'summary_writer']:
                    if hasattr(tb_logger, attr_name):
                        writer = getattr(tb_logger, attr_name)
                        if hasattr(writer, 'add_custom_scalars'):
                            writer.add_custom_scalars(layout)
                            logger.info('成功添加TensorBoard自定义布局')
                            break
                
                if writer is None:
                    logger.warning('无法获取TensorBoard的Writer实例，自定义布局未添加')
            except Exception as e:
                logger.warning(f'添加TensorBoard布局时出错: {str(e)}')
            
            logger.info(f'TensorBoard初始化完成: {log_dir}')
    
    # 初始化wandb日志器
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                    is not None) and 'debug' not in opt['name']:
        init_wandb_logger(opt)
    
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    """创建训练和验证数据加载器"""
    # 确保有 gpu_ids
    if 'gpu_ids' not in opt:
        num_gpu = opt.get('num_gpu', 1)
        if num_gpu == 0:
            num_gpu = torch.cuda.device_count()
        opt['gpu_ids'] = list(range(num_gpu))
        logger.info(f'自动创建 gpu_ids: {opt["gpu_ids"]}')
    
    # 创建训练数据集
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=len(opt['gpu_ids']),
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
                
            num_iter_per_epoch = math.ceil(len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(f'训练数据集大小: {len(train_set)}')
            logger.info(f'每个epoch总迭代次数: {num_iter_per_epoch}')
            logger.info(f'总训练epoch: {total_epochs}')
            
            if opt['val']['val_freq'] > 0:
                logger.info(f"每 {opt['val']['val_freq']} 轮迭代执行一次验证")
        
        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set,
                dataset_opt,
                num_gpu=len(opt['gpu_ids']),
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(f'验证数据集大小: {len(val_set)}')
    
    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    """主训练函数"""
    # 解析参数
    opt, args = parse_options()
    
    # 创建实验目录
    if args.reuse_exp_dir and os.path.exists(opt['path']['experiments_root']):
        logger_exp = get_root_logger()
        logger_exp.info(f"继续使用已存在的实验目录: {opt['path']['experiments_root']}")
        # 确保子目录存在
        os.makedirs(os.path.join(opt['path']['experiments_root'], 'models'), exist_ok=True)
        os.makedirs(os.path.join(opt['path']['experiments_root'], 'training_states'), exist_ok=True)
        os.makedirs(os.path.join(opt['path']['experiments_root'], 'visualization'), exist_ok=True)
    else:
        make_exp_dirs(opt)
        
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        if args.reuse_exp_dir and os.path.exists(os.path.join('tb_logger', opt['name'])):
            logger_tb = get_root_logger()
            logger_tb.info(f"继续使用已存在的TensorBoard日志目录: {os.path.join('tb_logger', opt['name'])}")
        else:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))
    
    # 初始化日志器 - 传递args参数
    logger, tb_logger = init_loggers(opt, args)
    
    # 创建数据加载器
    train_loader, train_sampler, val_loader, total_epochs, total_iters = create_train_val_dataloader(opt, logger)
    
    # 创建模型
    model = build_model(opt)
    start_epoch = 0
    current_iter = 0
    
    # 恢复训练
    if opt['path'].get('resume_state'):
        resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(opt['gpu_ids'][0]))
        check_resume(opt, resume_state['iter'])
        model.resume_training(resume_state)  # 恢复训练状态
        current_iter = resume_state['iter']
        start_epoch = resume_state['epoch']
        logger.info(f"从迭代轮次 {current_iter} 恢复训练")
    
    # 创建消息日志器（使用增强版）
    msg_logger = EnhancedMessageLogger(opt, current_iter, tb_logger)
    
    # 创建预取数据加载器
    prefetcher = None
    if opt.get('use_prefetch', False):
        logger.info('使用 CUDA 预加载数据...')
        prefetcher = CUDAPrefetcher(train_loader, opt)
    else:
        logger.info('使用 CPU 预加载数据...')
        prefetcher = CPUPrefetcher(train_loader)
    
    # 内存优化
    memory_efficient = opt['train'].get('memory_efficient', False)
    if memory_efficient:
        logger.info("启用内存优化模式")
    
    # 计算每个epoch的迭代次数
    dataset_opt = opt['datasets']['train']
    iter_per_epoch = math.ceil(
        len(train_loader.dataset) * dataset_opt.get('dataset_enlarge_ratio', 1) / 
        (dataset_opt['batch_size_per_gpu'] * opt['world_size'])
    )
    
    # 训练循环
    logger.info(f'开始训练，总迭代次数: {total_iters}, 总轮次: {total_epochs}')
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        
        # 添加tqdm进度条，创建带进度条的迭代器
        # 计算当前epoch预期的迭代次数，用于设置进度条总长度
        expected_iter = min(iter_per_epoch, total_iters - current_iter)
        pbar = tqdm(total=expected_iter, desc=f'Epoch {epoch}/{total_epochs}', 
                   dynamic_ncols=True, unit='iter')
        
        while train_data is not None:
            current_iter += 1
            if current_iter > total_iters:
                break
                
            # 更新学习率，第一次执行时会初始化
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            # 计算时间
            iter_start_time = time.time()
            
            # 训练一个step
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            
            # 计算时间
            iter_time = time.time() - iter_start_time
            
            # # 训练日志
            # if current_iter % opt['logger']['print_freq'] == 0:
            #     log_vars = {'epoch': epoch, 'iter': current_iter}
            #     log_vars.update({'lrs': model.get_current_learning_rate()})
            #     log_vars.update({'time': iter_time})
            #     log_vars.update(model.get_current_log())
            #     message = logger.train_message(log_vars)
            #     logger.info(message)
            
            # 记录损失到TensorBoard
            tb_log_freq = opt['logger'].get('tb_log_freq', opt['logger']['print_freq'])
            if tb_logger and current_iter % tb_log_freq == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                tb_logger_kwargs = dict(global_step=current_iter)
                tb_logger.add_scalar('epoch', epoch, **tb_logger_kwargs)
                # 记录当前日志到TensorBoard，使用模型内部的log_current方法
                model.log_current(current_iter, tb_logger)
                
            # 每隔一定迭代次数记录当前训练图像
            if tb_logger and current_iter % opt['logger'].get('save_training_image_freq', 1000) == 0:
                try:
                    # 获取当前训练批次的图像
                    lq = model.lq[:1]  # 只取第一张图片
                    gt = model.gt[:1]
                    sr = model.output[:1]
                    
                    # 转换为numpy图像以便添加到TensorBoard
                    lq_img = tensor2img(lq, rgb2bgr=False, min_max=(0, 1))
                    sr_img = tensor2img(sr, rgb2bgr=False, min_max=(0, 1))
                    gt_img = tensor2img(gt, rgb2bgr=False, min_max=(0, 1))
                    
                    # 将LQ图像放大到与SR图像相同的尺寸
                    if lq_img.shape[0] != sr_img.shape[0] or lq_img.shape[1] != sr_img.shape[1]:
                        lq_img = cv2.resize(lq_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    
                    # 确保GT图像与SR尺寸相同
                    if gt_img.shape[0] != sr_img.shape[0] or gt_img.shape[1] != sr_img.shape[1]:
                        gt_img = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    
                    # 添加文本标记以便区分图像
                    if hasattr(model, '_add_text_marker'):
                        lq_img = model._add_text_marker(lq_img, 'LQ')
                        sr_img = model._add_text_marker(sr_img, 'SR')
                        gt_img = model._add_text_marker(gt_img, 'GT')
                    
                    # 横向拼接三个图像并添加水平空白间隔
                    h, w = sr_img.shape[:2]
                    spacer = np.ones((h, 10, 3), dtype=np.uint8) * 255  # 白色水平空白间隔
                    comparison = np.concatenate(
                        [lq_img, spacer, sr_img, spacer, gt_img], 
                        axis=1  # 使用axis=1进行水平拼接
                    )
                    
                    # 添加到TensorBoard
                    tb_logger.add_image('train/images/comparison', comparison, current_iter, dataformats='HWC')
                    
                    # 也添加单独的图像
                    tb_logger.add_image('train/images/LQ', lq_img, current_iter, dataformats='HWC')
                    tb_logger.add_image('train/images/SR', sr_img, current_iter, dataformats='HWC')
                    tb_logger.add_image('train/images/GT', gt_img, current_iter, dataformats='HWC')
                    
                    # 生成简单的热力图 - 直接使用差异图
                    diff_map = np.abs(sr_img.astype(np.float32) - gt_img.astype(np.float32))
                    diff_map = diff_map.mean(axis=2)  # 转为单通道
                    diff_map = diff_map / diff_map.max() * 255  # 归一化到0-255
                    
                    # 应用热力图颜色映射
                    heatmap = cv2.applyColorMap(diff_map.astype(np.uint8), cv2.COLORMAP_JET)
                    
                    # 将热力图与SR图像混合
                    overlay = cv2.addWeighted(sr_img, 0.7, heatmap, 0.3, 0)
                    
                    # 添加到TensorBoard
                    tb_logger.add_image('train/attention/diff_map', overlay, current_iter, dataformats='HWC')
                    tb_logger.add_image('train/heatmap/diff_map', heatmap, current_iter, dataformats='HWC')
                    
                    # 记录日志
                    logger.info(f'已添加训练图像和热力图到TensorBoard，当前迭代: {current_iter}')
                except Exception as e:
                    logger.error(f"记录训练图像到TensorBoard时出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 更新迭代计数
                current_iter += 1
            
            # 保存模型
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('保存模型...')
                model.save(epoch, current_iter)
            
            # 验证
            if opt['val']['val_freq'] > 0 and current_iter % opt['val']['val_freq'] == 0:
                # 减少内存使用
                if memory_efficient:
                    torch.cuda.empty_cache()
                    
                # 验证前先更新进度条描述
                pbar.set_description(f'Epoch {epoch}/{total_epochs} (Validating...)')
                
                # 确保保存图像用于可视化
                model.validation(val_loader, current_iter, tb_logger, save_img=True)
                
                # 验证后恢复进度条描述
                pbar.set_description(f'Epoch {epoch}/{total_epochs}')
                
                # 减少内存使用
                if memory_efficient:
                    torch.cuda.empty_cache()
            
            # 测试
            if 'test' in opt and 'test_freq' in opt['test'] and opt['test']['test_freq'] > 0 and current_iter % opt['test']['test_freq'] == 0:
                # 减少内存使用
                if memory_efficient:
                    torch.cuda.empty_cache()
                
                # 获取测试数据加载器
                test_set = build_dataset(opt['datasets']['test'])
                test_loader = build_dataloader(
                    test_set,
                    opt['datasets']['test'],
                    num_gpu=len(opt['gpu_ids']),
                    dist=opt['dist'],
                    sampler=None,
                    seed=opt['manual_seed'])
                
                logger.info(f"开始在完整测试集上进行测试，当前迭代: {current_iter}")
                pbar.set_description(f'Epoch {epoch}/{total_epochs} (Testing...)')
                
                # 执行测试
                model.validation(test_loader, current_iter, tb_logger, save_img=False, is_test=True)
                
                # 添加测试结果到TensorBoard
                if tb_logger:
                    try:
                        # 获取当前测试批次的图像
                        lq = model.lq[:1]  # 只取第一张图片
                        gt = model.gt[:1]
                        sr = model.output[:1]
                        
                        # 转换为numpy图像
                        lq_img = tensor2img(lq, rgb2bgr=False, min_max=(0, 1))
                        sr_img = tensor2img(sr, rgb2bgr=False, min_max=(0, 1))
                        gt_img = tensor2img(gt, rgb2bgr=False, min_max=(0, 1))
                        
                        # 将LQ图像放大到与SR图像相同的尺寸
                        if lq_img.shape[0] != sr_img.shape[0] or lq_img.shape[1] != sr_img.shape[1]:
                            lq_img = cv2.resize(lq_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                        
                        # 确保GT图像与SR尺寸相同
                        if gt_img.shape[0] != sr_img.shape[0] or gt_img.shape[1] != sr_img.shape[1]:
                            gt_img = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                        
                        # 添加文本标记
                        if hasattr(model, '_add_text_marker'):
                            lq_img = model._add_text_marker(lq_img, 'LQ')
                            sr_img = model._add_text_marker(sr_img, 'SR')
                            gt_img = model._add_text_marker(gt_img, 'GT')
                        
                        # 横向拼接三个图像并添加水平空白间隔
                        h, w = sr_img.shape[:2]
                        spacer = np.ones((h, 10, 3), dtype=np.uint8) * 255  # 白色水平空白间隔
                        comparison = np.concatenate(
                            [lq_img, spacer, sr_img, spacer, gt_img], 
                            axis=1  # 使用axis=1进行水平拼接
                        )
                        
                        # 添加到TensorBoard
                        tb_logger.add_image('test/images/comparison', comparison, current_iter, dataformats='HWC')
                        tb_logger.add_image('test/images/LQ', lq_img, current_iter, dataformats='HWC')
                        tb_logger.add_image('test/images/SR', sr_img, current_iter, dataformats='HWC')
                        tb_logger.add_image('test/images/GT', gt_img, current_iter, dataformats='HWC')
                        
                        # 生成差异热力图
                        diff_map = np.abs(sr_img.astype(np.float32) - gt_img.astype(np.float32))
                        diff_map = diff_map.mean(axis=2)  # 转为单通道
                        diff_map = diff_map / diff_map.max() * 255  # 归一化到0-255
                        
                        # 应用热力图颜色映射
                        heatmap = cv2.applyColorMap(diff_map.astype(np.uint8), cv2.COLORMAP_JET)
                        
                        # 将热力图与SR图像混合
                        overlay = cv2.addWeighted(sr_img, 0.7, heatmap, 0.3, 0)
                        
                        # 添加到TensorBoard
                        tb_logger.add_image('test/attention/diff_map', overlay, current_iter, dataformats='HWC')
                        tb_logger.add_image('test/heatmap/diff_map', heatmap, current_iter, dataformats='HWC')
                        
                        # 记录测试指标
                        if hasattr(model, 'get_current_log'):
                            test_log = model.get_current_log()
                            for k, v in test_log.items():
                                if 'psnr' in k.lower() or 'ssim' in k.lower():
                                    tb_logger.add_scalar(f'test/metrics/{k}', v, current_iter)
                        
                        logger.info(f'已添加测试图像和指标到TensorBoard，当前迭代: {current_iter}')
                    except Exception as e:
                        logger.error(f"记录测试图像到TensorBoard时出错: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # 测试后恢复进度条描述
                pbar.set_description(f'Epoch {epoch}/{total_epochs}')
                
                # 减少内存使用
                if memory_efficient:
                    torch.cuda.empty_cache()
            
            train_data = prefetcher.next()
            pbar.update(1)  # 更新进度条
        
        # 一个epoch结束，关闭进度条
        pbar.close()
        
        if current_iter > total_iters:
            break
    
    # 保存最终模型
    logger.info('完成训练，保存最终模型...')
    model.save(epoch=-1, current_iter=-1)  # -1标记为最终模型
    
    if tb_logger:
        tb_logger.close()
    
    if prefetcher:
        del prefetcher


if __name__ == '__main__':
    main() 