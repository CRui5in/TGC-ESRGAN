import os
import torch
import torch.multiprocessing as mp
import psutil
import gc
import logging
from basicsr.train import train_pipeline
from basicsr.utils.options import dict2str, parse

def log_memory_usage():
    """记录当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f'内存使用: {memory_info.rss / 1024 / 1024:.2f} MB')
    if torch.cuda.is_available():
        logging.info(f'GPU内存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB')

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # 解析命令行参数
    opt = parse()
    
    # 设置多进程启动方法
    if opt.get('multiprocessing_start_method', 'spawn') == 'spawn':
        mp.set_start_method('spawn', force=True)
    
    # 记录初始内存使用情况
    log_memory_usage()
    
    # 训练流程
    train_pipeline(opt)
    
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 记录最终内存使用情况
    log_memory_usage()

if __name__ == '__main__':
    main() 