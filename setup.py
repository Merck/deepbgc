import os
from setuptools import setup, find_packages

install_requires = [
    'argparse',
    'biopython',
    'scikit-learn',
    'pandas',
    'keras',
    'tensorflow',
    'hmmlearn',
    'matplotlib'
]

def get_version():
    for line in open(os.path.join('deepbgc', '__init__.py'), 'r'):
        if line.startswith('VERSION'):
            break

    _, _, version_str = line.strip().split()
    return version_str[1:-1]

setup(
    name='deepbgc',
    version=get_version(),
    description='DeepBGC - Biosynthetic Gene Cluster detection and classification',
    long_description=open('README.md', 'r').read(),
    author='David Příhoda, Geoffrey Hannigan',
    packages=find_packages(),
    author_email='david.prihoda1@merck.com',
    license='MIT',
    install_requires=install_requires,
    keywords='biosynthetic gene clusters, bgc detection, deep learning, pfam2vec',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
    ],
    include_package_data=True,
    url='http://www.merck.com',
    entry_points={
        'console_scripts': ['deepbgc = deepbgc.main:main']
    }
)
