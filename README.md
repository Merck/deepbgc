# DeepBGC: Biosynthetic Gene Cluster detection and classification

DeepBGC detects BGCs in bacterial and fungal genomes using deep learning. 
DeepBGC employs a Bidirectional Long Short-Term Memory Recurrent Neural Network 
and a word2vec-like vector embedding of Pfam protein domains. 
Product class and activity of detected BGCs is predicted using a Random Forest classifier.

[![PyPI license](https://img.shields.io/pypi/l/deepbgc.svg)](https://pypi.python.org/pypi/deepbgc/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deepbgc.svg?color=green&label=pypi%20downloads)
![GitHub Releases](https://img.shields.io/github/downloads/Merck/deepbgc/latest/total.svg?label=GitHub%20downloads)
[![PyPI version](https://badge.fury.io/py/deepbgc.svg)](https://badge.fury.io/py/deepbgc)

![DeepBGC architecture](images/deepbgc.architecture.png?raw=true "DeepBGC architecture")

## Install using pip

- Install Python version 2.7+ or 3.4+
- Install Prodigal and put the `prodigal` binary it on your PATH: https://github.com/hyattpd/Prodigal/releases
- Install HMMER and put the `hmmscan` and `hmmpress` binaries on your PATH: http://hmmer.org/download.html
- Run `pip install deepbgc` to install the DeepBGC package      

## Use DeepBGC

### Download models and Pfam database

Before you can use DeepBGC, download trained models and Pfam database:

```bash
deepbgc download
```

You can display downloaded dependencies and models using:

```bash
deepbgc info
```

### Detection and classification

![DeepBGC pipeline](images/deepbgc.pipeline.png?raw=true "DeepBGC pipeline")

Detect and classify BGCs in a genomic sequence. 
Proteins and Pfam domains are detected automatically if not already annotated (HMMER and Prodigal needed)

```bash
# Show command help docs
deepbgc pipeline --help

# Detect and classify BGCs in mySequence.fa using DeepBGC algorithm and save the output to mySequence directory.
deepbgc pipeline mySequence.fa
```

This will produce a directory with multiple files and a README.txt with file descriptions.

#### Example output

See the [DeepBGC Example Result Notebook](https://nbviewer.jupyter.org/urls/github.com/Merck/deepbgc/releases/download/v0.1.0/DeepBGC_Example_Result.ipynb).
Data can be downloaded on the [releases page](https://github.com/Merck/deepbgc/releases)

![Detected BGC Regions](images/deepbgc.bgc.png?raw=true "Detected BGC regions")

### Model training

You can train your own BGC detection and classification models, see `deepbgc train --help` for documentation and examples.

DeepBGC positives, negatives and other training and validation data can be found on the [releases page](https://github.com/Merck/deepbgc/releases).
