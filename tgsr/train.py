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

def parse_options():
    """命令行参数解析"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='路径到配置文件')
    parser.add_argument('--launcher', type=str, default='none', help='启动器 {none, pytorch, slurm}')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0, help='分布式训练的本地排名')
    parser.add_argument('--memory_opt', action='store_true', help='启用内存优化')
    parser.add_argument('--reuse_exp_dir', action='store_true', help='继续使用已存在的实验目录')
    args = parser.parse_args()
    
    # 从环境变量中获取分布式训练参数（用于torchrun）
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    
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
            # 更新分布式参数
            if 'dist_params' in opt:
                opt['dist_params']['rank'] = args.local_rank
                opt['dist_params']['local_rank'] = args.local_rank
                opt['dist_params']['launcher'] = args.launcher
            else:
                opt['dist_params'] = {
                    'backend': 'nccl', 
                    'rank': args.local_rank,
                    'local_rank': args.local_rank,
                    'launcher': args.launcher,
                    'port': 29500
                }
            
            # 初始化分布式
            init_dist(args.launcher)
            print(f'初始化PyTorch分布式训练环境... rank: {args.local_rank}')
    
    # 内存优化
    if args.memory_opt:
        opt['train']['memory_efficient'] = True
    
    # 获取分布式信息
    opt['rank'], opt['world_size'] = get_dist_info()
    
    # 根据实际分布式情况更新参数
    if opt['dist']:
        # 根据rank和world_size自动调整GPU IDs
        if 'gpu_ids' in opt:
            if opt['world_size'] > 1:
                opt['gpu_ids'] = list(range(opt['world_size']))
            else:
                opt['gpu_ids'] = [opt['rank']]  # 单GPU情况下使用指定的rank
        print(f'GPU IDs: {opt["gpu_ids"]}')
    
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
    
    # 只有在rank=0的主进程中初始化TensorBoard
    if opt['rank'] == 0 and opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
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
        tb_logger.add_text('Config', dict2str(opt).replace('\n', '  \n'), 0)
        
        # 添加环境信息
        env_info = get_env_info()
        tb_logger.add_text('Environment', str(env_info).replace('\n', '  \n'), 0)
        
        # 记录主要超参数
        if 'network_g' in opt:
            if 'type' in opt['network_g']:
                tb_logger.add_text('Model', opt['network_g']['type'], 0)
            
            # 定义TensorBoard布局
            layout = {
                "训练与验证": {
                    "损失函数": {
                        "训练损失": ["Multiline", ["Train/Losses/l_pix", "Train/Losses/l_percep", 
                                          "Train/Losses/l_style", "Train/Losses/l_g_gan", 
                                          "Train/Losses/l_total"]],
                        "判别器损失": ["Multiline", ["Train/Losses/l_d_real", "Train/Losses/l_d_fake"]]
                    },
                    "学习率": {
                        "学习率": ["Multiline", ["Train/learning_rate"]]
                    },
                    "验证指标": {
                        "指标": ["Multiline", ["Validation/Metrics/psnr", "Validation/Metrics/ssim"]]
                    },
                    "测试指标": {
                        "指标": ["Multiline", ["Test/Metrics/psnr", "Test/Metrics/ssim"]]
                    }
                },
                "可视化结果": {
                    "训练图像": {
                        "单独图像": ["Images", ["Train/Images/LQ", "Train/Images/SR", "Train/Images/GT"]]
                    },
                    "验证图像": {
                        "单独图像": ["Images", ["Validation/Images/LQ", "Validation/Images/SR", "Validation/Images/GT"]]
                    },
                    "测试图像": {
                        "单独图像": ["Images", ["Test/Images/LQ", "Test/Images/SR", "Test/Images/GT"]]
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
    
    # 初始化wandb日志器 - 同样只在rank=0的主进程中初始化
    if opt['rank'] == 0 and (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
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
    
    # 显示分布式训练信息
    rank, world_size = get_dist_info()
    print(f"分布式训练信息 - Rank: {rank}, World Size: {world_size}, GPU: {torch.cuda.current_device()}")
    
    # 创建实验目录
    if args.reuse_exp_dir and os.path.exists(opt['path']['experiments_root']):
        logger_exp = get_root_logger()
        logger_exp.info(f"继续使用已存在的实验目录: {opt['path']['experiments_root']}")
        # 确保子目录存在
        os.makedirs(os.path.join(opt['path']['experiments_root'], 'models'), exist_ok=True)
        os.makedirs(os.path.join(opt['path']['experiments_root'], 'training_states'), exist_ok=True)
        os.makedirs(os.path.join(opt['path']['experiments_root'], 'visualization'), exist_ok=True)
    else:
        # 只在主进程中创建目录
        if opt['rank'] == 0:
            make_exp_dirs(opt)
        
    # 对于分布式训练，确保所有进程都等待主进程创建目录
    if opt.get('dist', False):
        torch.distributed.barrier()
        
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        if args.reuse_exp_dir and os.path.exists(os.path.join('tb_logger', opt['name'])):
            logger_tb = get_root_logger()
            logger_tb.info(f"继续使用已存在的TensorBoard日志目录: {os.path.join('tb_logger', opt['name'])}")
        else:
            # 只在主进程中创建TensorBoard目录
            if opt['rank'] == 0:
                mkdir_and_rename(osp.join('tb_logger', opt['name']))
    
    # 初始化日志器 - 传递args参数
    logger, tb_logger = init_loggers(opt, args)
    
    # 记录重要的训练参数
    logger.info(f'分布式训练: {opt.get("dist", False)}, 使用进程: {opt["rank"]}/{opt["world_size"]}')
    if opt.get("dist", False):
        logger.info(f'分布式后端: {opt.get("dist_params", {}).get("backend", "未设置")}')
    logger.info(f'GPU IDs: {opt.get("gpu_ids", [])}')
    logger.info(f'使用DDP: {opt.get("use_ddp", False)}')
    logger.info(f'混合精度训练: {opt["train"].get("fp16", False)}')
    logger.info(f'TensorBoard记录器状态: {"已启用" if tb_logger else "未启用"}')
    logger.info(f'EMA: {opt["train"].get("use_ema", False)}')
    if opt["train"].get("use_ema", False):
        logger.info(f'EMA衰减率: {opt["train"].get("ema_decay", 0.999)}')
    logger.info(f'文本编码器: {opt.get("text_encoder", {}).get("name", "未设置")}')
    
    # 记录梯度累积配置
    accumulation_steps = opt["train"].get("accumulation_steps", 1)
    if accumulation_steps > 1:
        logger.info(f'使用梯度累积，步数: {accumulation_steps}')
        effective_batch_size = opt["datasets"]["train"]["batch_size_per_gpu"] * accumulation_steps * opt["world_size"]
        logger.info(f'等效批量大小: {effective_batch_size} (batch_size_per_gpu * accumulation_steps * world_size)')
    else:
        logger.info(f'未使用梯度累积')
        logger.info(f'批量大小: {opt["datasets"]["train"]["batch_size_per_gpu"] * opt["world_size"]} (batch_size_per_gpu * world_size)')
    
    # 创建数据加载器
    train_loader, train_sampler, val_loader, total_epochs, total_iters = create_train_val_dataloader(opt, logger)
    
    # 创建模型
    model = build_model(opt)
    start_epoch = 0
    current_iter = 0
    resume_state = False
    
    # 恢复训练
    if opt['path'].get('resume_state'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device)
        )
        check_resume(opt, resume_state['iter'])
        model.resume_training(resume_state)  # 恢复训练状态
        current_iter = resume_state['iter']
        start_epoch = resume_state['epoch']
        logger.info(f"从迭代轮次 {current_iter} 恢复训练")
        resume_state = True
    
    
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
        # 设置epoch用于数据采样
        train_sampler.set_epoch(epoch)
        
        # 重置预取器
        prefetcher.reset()
        train_data = prefetcher.next()
        
        # 添加tqdm进度条，创建带进度条的迭代器
        # 计算当前epoch预期的迭代次数，用于设置进度条总长度
        expected_iter = min(iter_per_epoch, total_iters - current_iter)
        
        # 只在主进程中显示进度条
        if opt['rank'] == 0:
            # 计算当前epoch的起始迭代次数
            epoch_start_iter = current_iter
            pbar = tqdm(total=expected_iter, 
                      initial=current_iter - epoch_start_iter,
                      desc=f'Epoch {epoch}/{total_epochs} (Iter {current_iter}/{total_iters})', 
                      dynamic_ncols=True, 
                      unit='iter')
        
            if resume_state:
                pbar.update(current_iter)
                resume_state = False

        # 训练循环
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
            
            # 记录损失到TensorBoard - 仅在主进程中
            if opt['rank'] == 0:
                tb_log_freq = opt['logger'].get('tb_log_freq', opt['logger']['print_freq'])
                if tb_logger is not None and current_iter % tb_log_freq == 0:
                    model.log_current(current_iter, tb_logger)
            
            # 更新进度条 - 仅在主进程中
            if opt['rank'] == 0:
                pbar.update(1)
                # 更新进度条描述，显示当前迭代次数和总迭代次数
                pbar.set_description(f'Epoch {epoch}/{total_epochs} (Iter {current_iter}/{total_iters})')
            
            # 保存模型 - 仅在主进程中
            if opt['rank'] == 0 and current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('保存模型...')
                model.save(epoch, current_iter)
            
            # 验证 - 仅在特定迭代次数执行，并且current_iter > 0
            if opt['rank'] == 0 and opt['val']['val_freq'] > 0 and current_iter % opt['val']['val_freq'] == 0:
                # 减少内存使用
                if memory_efficient:
                    torch.cuda.empty_cache()
                    
                # 验证前先更新进度条描述
                if opt['rank'] == 0:
                    pbar.set_description(f'Epoch {epoch}/{total_epochs} (Validating...)')
                
                # 确保保存图像用于可视化
                model.validation(val_loader, current_iter, tb_logger, save_img=True)
                
                # 验证后恢复进度条描述
                if opt['rank'] == 0:
                    pbar.set_description(f'Epoch {epoch}/{total_epochs}')
                
                # 减少内存使用
                if memory_efficient:
                    torch.cuda.empty_cache()
            
            # 测试 - 仅在特定迭代次数执行，并且current_iter > 0
            if opt['rank'] == 0 and 'test' in opt and 'test_freq' in opt['test'] and opt['test']['test_freq'] > 0 and current_iter % opt['test']['test_freq'] == 0:
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
                
                if opt['rank'] == 0:
                    pbar.set_description(f'Epoch {epoch}/{total_epochs} (Testing...)')
                
                # 执行测试
                model.validation(test_loader, current_iter, tb_logger, save_img=False, is_test=True)
                
                # 添加测试结果到TensorBoard - 仅在主进程中
                if opt['rank'] == 0 and tb_logger is not None:
                    try:
                        # 不再重复添加测试图像和指标，由model.test()和model.validation()函数处理
                        logger.info(f'已添加测试图像和指标到TensorBoard，当前迭代: {current_iter}')
                    except Exception as e:
                        logger.error(f"记录测试图像到TensorBoard时出错: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # 测试后恢复进度条描述
                if opt['rank'] == 0:
                    pbar.set_description(f'Epoch {epoch}/{total_epochs}')
                
                # 减少内存使用
                if memory_efficient:
                    torch.cuda.empty_cache()
            
            train_data = prefetcher.next()
        
        # 一个epoch结束，关闭进度条
        if opt['rank'] == 0:
            pbar.close()
        
        if current_iter > total_iters:
            break
    
    # 保存最终模型 - 仅在主进程中
    if opt['rank'] == 0:
        logger.info('完成训练，保存最终模型...')
        model.save(epoch=-1, current_iter=-1)  # -1标记为最终模型
    
    # 等待所有进程
    if opt.get('dist', False):
        torch.distributed.barrier()
    
    # 清理资源 - 所有进程都执行
    if tb_logger is not None:
        tb_logger.close()
    
    if prefetcher:
        del prefetcher


if __name__ == '__main__':
    main() 