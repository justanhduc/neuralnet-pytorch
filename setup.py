from setuptools import setup, find_packages
import codecs
import os


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neuralnet-pytorch',
    version='0.0.1a',
    description='A high-level library on top of Pytorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/justanhduc/neuralnet-pytorch',
    author='Duc Nguyen',
    author_email='adnguyen@yonsei.ac.kr',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5'
    ],
    packages=find_packages(exclude=['examples']),
    install_requires=['matplotlib', 'scipy', 'numpy'],
    project_urls={
        'Bug Reports': 'https://github.com/justanhduc/neuralnet-pytorch/issues',
        'Source': 'https://github.com/justanhduc/neuralnet-pytorch/',
    },
)