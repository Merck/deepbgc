# DeepBGC: Biosynthetic Gene Cluster detection and classification

DeepBGC detects BGCs in bacterial and fungal genomes using deep learning. 
DeepBGC employs a Bidirectional Long Short-Term Memory Recurrent Neural Network 
and a word2vec-like vector embedding of Pfam protein domains. 
Product class and activity of detected BGCs is predicted using a Random Forest classifier.

[![BioConda Install](https://img.shields.io/conda/dn/bioconda/deepbgc.svg?style=flag&label=BioConda%20install&color=green)](https://anaconda.org/bioconda/deepbgc) 
![PyPI - Downloads](https://img.shields.io/pypi/dm/deepbgc.svg?color=green&label=PyPI%20downloads)
[![PyPI license](https://img.shields.io/pypi/l/deepbgc.svg)](https://pypi.python.org/pypi/deepbgc/)
[![PyPI version](https://badge.fury.io/py/deepbgc.svg)](https://badge.fury.io/py/deepbgc)
[![CI](https://api.travis-ci.org/Merck/deepbgc.svg?branch=master)](https://travis-ci.org/Merck/deepbgc)

![DeepBGC architecture](images/deepbgc.architecture.png?raw=true "DeepBGC architecture")

## 📌 News 📌

- **DeepBGC 0.1.23**: Predicted BGCs can now be uploaded for visualization in **antiSMASH** using a JSON output file
  - Install and run DeepBGC as usual based on instructions below
  - Upload `antismash.json` from the DeepBGC output folder using "Upload extra annotations" on the [antiSMASH](https://antismash.secondarymetabolites.org/) page
  - Predicted BGC regions and their prediction scores will be displayed alongside antiSMASH BGCs
 
## Publications

A deep learning genome-mining strategy for biosynthetic gene cluster prediction <br>
Geoffrey D Hannigan,  David Prihoda et al., Nucleic Acids Research, gkz654, https://doi.org/10.1093/nar/gkz654


## Install using conda (recommended)

- Install Bioconda by following Step 1 and 2 from: https://bioconda.github.io/
- Add `conda-forge` channel using `conda config --add channels conda-forge`
- Run `conda install deepbgc` to install DeepBGC and all of its dependencies    

## Install using pip (if conda is not available)

If you don't mind installing the HMMER and Prodigal dependencies manually, you can also install DeepBGC using pip:

- Install Python version 3.6 or 3.7 (Note: **Python 3.8 is not supported** due to Tensorflow < 2.0 dependency)
- Install Prodigal and put the `prodigal` binary it on your PATH: https://github.com/hyattpd/Prodigal/releases
- Install HMMER and put the `hmmscan` and `hmmpress` binaries on your PATH: http://hmmer.org/download.html
- Run `pip install deepbgc` to install DeepBGC   

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

# Detect and classify BGCs in mySequence.fa using DeepBGC detector.
deepbgc pipeline mySequence.fa

# Detect and classify BGCs in mySequence.fa using custom DeepBGC detector trained on your own data.
deepbgc pipeline --detector path/to/myDetector.pkl mySequence.fa
```

This will produce a `mySequence` directory with multiple files and a README.txt with file descriptions.

See [Train DeepBGC on your own data](#train-deepbgc-on-your-own-data) section below for more information about training a custom detector or classifier.

#### Example output

See the [DeepBGC Example Result Notebook](https://nbviewer.jupyter.org/urls/github.com/Merck/deepbgc/releases/download/v0.1.0/DeepBGC_Example_Result.ipynb).
Data can be downloaded on the [releases page](https://github.com/Merck/deepbgc/releases)

![Detected BGC Regions](images/deepbgc.bgc.png?raw=true "Detected BGC regions")

## Train DeepBGC on your own data

You can train your own BGC detection and classification models, see `deepbgc train --help` for documentation and examples.

Training and validation data can be found in [release 0.1.0](https://github.com/Merck/deepbgc/releases/tag/v0.1.0) and [release 0.1.5](https://github.com/Merck/deepbgc/releases/tag/v0.1.5). You will need:
- Positive (BGC) training data - In most cases, this is your own BGC training set, see "Preparing training data" section below
- Negative (Non-BGC) training data - Needed for BGC detection. You can use `GeneSwap_Negatives.pfam.tsv` from release https://github.com/Merck/deepbgc/releases/tag/v0.1.0
- Validation data - Needed for BGC detection. Contigs with annotated BGC and non-BGC regions. A working example can be downloaded from https://github.com/Merck/deepbgc/releases/tag/v0.1.5
- Trained Pfam2vec vectors - "Vocabulary" converting Pfam IDs to meaningful numeric vectors, you can reuse previously trained `pfam2vec.csv` results from https://github.com/Merck/deepbgc/releases/tag/v0.1.0
- JSON configuration files - See JSON section below

If you have any questions about using or training DeepBGC, feel free to submit an issue.

### Preparing training data

The training examples need to be prepared in Pfam TSV format, which can be prepared from your sequence
using `deepbgc prepare`. 

First, you will need to manually add an `in_cluster` column that will contain 0 for pfams outside a BGC 
and 1 for pfams inside a BGC. We recommend preparing a separate negative TSV and positive TSV file, 
where the column will be equal to all 0 or 1 respectively. 

Finally, you will need to manually add a `sequence_id` column ,
which will identify a continuous sequence of Pfams from a single sample (BGC or negative sequence).
The samples are shuffled during training to present the model with a random order of positive and negative samples.
Pfams with the same `sequence_id` value will be kept together. For example, if your training set contains multiple BGCs, the `sequence_id` column should contain the BGC ID.

**! New in version 0.1.17 !** You can now prepare *protein* FASTA sequences into a Pfam TSV file using `deepbgc prepare --protein`.


### JSON model training template files

DeepBGC is using JSON template files to define model architecture and training parameters. All templates can be downloaded in [release 0.1.0](https://github.com/Merck/deepbgc/releases/tag/v0.1.0).

JSON template for DeepBGC LSTM **detector** with pfam2vec is structured as follows:
```
{
  "type": "KerasRNN", - Model architecture (KerasRNN/DiscreteHMM/GeneBorderHMM)
  "build_params": { - Parameters for model architecture
    "batch_size": 16, - Number of splits of training data that is trained in parallel 
    "hidden_size": 128, - Size of vector storing the LSTM inner state
    "stateful": true - Remember previous sequence when training next batch
  },
  "fit_params": {
    "timesteps": 256, - Number of pfam2vec vectors trained in one batch
    "validation_size": 0, - Fraction of training data to use for validation (if validation data is not provided explicitly). Use 0.2 for 20% data used for testing.
    "verbose": 1, - Verbosity during training
    "num_epochs": 1000, - Number of passes over your training set during training. You probably want to use a lower number if not using early stopping on validation data.
    "early_stopping" : { - Stop model training when at certain validation performance
      "monitor": "val_auc_roc", - Use validation AUC ROC to observe performance
      "min_delta": 0.0001, - Stop training when the improvement in the last epochs did not improve more than 0.0001
      "patience": 20, - How many of the last epochs to check for improvement
      "mode": "max" - Stop training when given metric stops increasing (use "min" for decreasing metrics like loss)
    },
    "shuffle": true, - Shuffle samples in each epoch. Will use "sequence_id" field to group pfam vectors belonging to the same sample and shuffle them together 
    "optimizer": "adam", - Optimizer algorithm
    "learning_rate": 0.0001, - Learning rate
    "weighted": true - Increase weight of less-represented class. Will give more weight to BGC training samples if the non-BGC set is larger.
  },
  "input_params": {
    "features": [ - Array of features to use in model, see deepbgc/features.py
      {
        "type": "ProteinBorderTransformer" - Add two binary flags for pfam domains found at beginning or at end of protein
      },
      {
        "type": "Pfam2VecTransformer", - Convert pfam_id field to pfam2vec vector using provided pfam2vec table
        "vector_path": "#{PFAM2VEC}" - PFAM2VEC variable is filled in using command line argument --config
      }
    ]
  }
}
```

JSON template for Random Forest **classifier** is structured as follows:
```
{
  "type": "RandomForestClassifier", - Type of classifier (RandomForestClassifier)
  "build_params": {
    "n_estimators": 100, - Number of trees in random forest
    "random_state": 0 - Random seed used to get same result each time
  },
  "input_params": {
    "sequence_as_vector": true, - Convert each sample into a single vector
    "features": [
      {
        "type": "OneHotEncodingTransformer" - Convert each sequence of Pfams into a single binary vector (Pfam set)
      }
    ]
  }
}
```

### Using your trained model

Since version `0.1.10` you can provide a direct path to the detector or classifier model like so:
```bash
deepbgc pipeline \
    mySequence.fa \
    --detector path/to/myDetector.pkl \
    --classifier path/to/myClassifier.pkl 
```
