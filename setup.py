from setuptools import setup, find_packages

setup(
    name='torch-gamma',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
    ],
    author='KevinZonda',
    author_email='realkevin@tutanota.com',
    description='PyTorch γ is a high level library inspired by Ruby',
    keywords="pytorch, machine learning",
    url='https://github.com/KevinZonda/pytorch-gamma'
)