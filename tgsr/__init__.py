import importlib
import os
import os.path as osp

# 导入版本信息
try:
    from .version import __version__, __gitsha__
except ImportError:
    __version__ = '0.1.0'
    __gitsha__ = 'unknown'

# 自动导入所有模块
def import_all_modules_for_register():
    """导入所有模块以正确注册BASICSR模块."""
    file_dir = osp.dirname(osp.abspath(__file__))
    
    # 导入archs
    archs_dir = osp.join(file_dir, 'archs')
    if osp.exists(archs_dir):
        _modules = [
            osp.splitext(file_name)[0] for file_name in os.listdir(archs_dir)
            if file_name.endswith('.py') and not file_name.startswith('_')
        ]
        _modules.sort()
        for module in _modules:
            importlib.import_module(f'tgsr.archs.{module}')
    
    # 导入models
    models_dir = osp.join(file_dir, 'models')
    if osp.exists(models_dir):
        _modules = [
            osp.splitext(file_name)[0] for file_name in os.listdir(models_dir)
            if file_name.endswith('.py') and not file_name.startswith('_')
        ]
        _modules.sort()
        for module in _modules:
            importlib.import_module(f'tgsr.models.{module}')
    
    # 导入data
    data_dir = osp.join(file_dir, 'data')
    if osp.exists(data_dir):
        _modules = [
            osp.splitext(file_name)[0] for file_name in os.listdir(data_dir)
            if file_name.endswith('.py') and not file_name.startswith('_')
        ]
        _modules.sort()
        for module in _modules:
            importlib.import_module(f'tgsr.data.{module}')

# 导入所有模块以完成注册
import_all_modules_for_register() 