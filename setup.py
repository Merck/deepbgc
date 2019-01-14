from setuptools import setup, find_packages
from deepbgc import VERSION

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

setup(
    name='deepbgc',
    version=VERSION,
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
