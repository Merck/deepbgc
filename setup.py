#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages
import os
from io import open

install_requires = [
    'argparse',
    'biopython>=1.78', # Bio.Alphabet was removed then
    'scikit-learn==0.21.3', # sklearn with random forest compatibility
    'pandas==0.24.1',
    'numpy==1.21.0',
    'keras==2.2.4',
    'tensorflow==1.15.4',
    'matplotlib==2.2.3',
    'appdirs>=1.4.3',
    'scipy==1.2.0',
    'protobuf<4' # Needed to support loading old model pickle files
]

about = {}
# Read version number from deepbgc.__version__.py (see PEP 396)
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'deepbgc', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), about)

# Read contents of readme file into string
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='deepbgc',
    version=about['__version__'],
    description='DeepBGC - Biosynthetic Gene Cluster detection and classification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='David Příhoda, Geoffrey Hannigan',
    packages=find_packages(exclude=('test','test.*')),
    author_email='david.prihoda1@merck.com',
    license='MIT',
    python_requires=">=3.6,<3.8",
    install_requires=install_requires,
    keywords='biosynthetic gene clusters, bgc detection, deep learning, pfam2vec',
    extras_require={
        'hmm': ['hmmlearn>=0.2.1,<0.2.7']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
    ],
    include_package_data=True,
    url='https://github.com/Merck/DeepBGC',
    entry_points={
        'console_scripts': ['deepbgc = deepbgc.main:main']
    }
)
