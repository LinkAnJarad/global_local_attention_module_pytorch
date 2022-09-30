from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Unoffical Implementation of the Global Local Attention Module (GLAM) in PyTorch'

setup(
    name="global_local_attention_module_pytorch",
    version=VERSION,
    author="Link An Jarad",
    description=DESCRIPTION,
    url="https://github.com/LinkAnJarad/global_local_attention_module_pytorch",
    packages=find_packages(),
    install_requires=['torch']
)
