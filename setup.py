#!/usr/bin/env python

from setuptools import find_packages, setup

import os
import subprocess
import time
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

version_file = 'tgsr/version.py'


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# 该文件由setup.py自动生成
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    sha = get_hash()
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join(SHORT_VERSION.split('.'))

    with open(version_file, 'w') as f:
        f.write(content.format(SHORT_VERSION, sha, VERSION_INFO))


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


if __name__ == '__main__':
    write_version_py()
    setup(
        name='tgsr',
        version=get_version(),
        description='Text-Guided Super-Resolution 基于文本引导的超分辨率模型',
        long_description=open('README.md', encoding='utf8').read(),
        long_description_content_type='text/markdown',
        author='TGSR Contributors',
        author_email='your_email@example.com',
        keywords='computer vision, super resolution, text-guided',
        url='https://github.com/yourusername/TGSR',
        packages=find_packages(exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        license='MIT License',
        setup_requires=['torch>=1.7'],
        install_requires=get_requirements(),
        cmdclass={
            'build_ext': BuildExtension,
        },
        zip_safe=False) 