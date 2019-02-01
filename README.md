# DeepBGC: Biosynthetic Gene Cluster detection and classification.

## Install DeepBGC

- Run `pip install deepbgc` to install the `deepbgc` python module.

## Prerequisities

- Install Python 3.6 (version 3.7 is not supported by TensorFlow yet)
- Install Prodigal and put the `prodigal` binary it on your PATH: https://github.com/hyattpd/Prodigal/releases
- Install HMMER and put the `hmmscan` and `hmmpress` binaries on your PATH: http://hmmer.org/download.html
- Download and **extract** Pfam database from: ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam31.0/Pfam-A.hmm.gz

## Use DeepBGC

### Detection

Detect BGCs in a genomic sequence.

```bash
# Show detection help
deepbgc detect --help

# Detect BGCs in a nucleotide sequence
deepbgc detect --model DeepBGCDetector_v0.0.1.pkl --pfam Pfam-A.hmm --output myCandidates/ myInputSequence.fa

# Detect BGCs with >0.9 score in existing Pfam CSV sequence
deepbgc detect --model myModel.pkl --output myStrictCandidates/ -s 0.9 myCandidates/myCandidates.pfam.csv

```

### Classification

Classify BGCs into one or more classes.

```bash
# Show classification help
deepbgc classify --help

# Predict biosynthetic class of detected BGCs
deepbgc classify --model RandomForestMIBiGClasses_v0.0.1.pkl --output myCandidates/myCandidates.classes.csv myCandidates/myCandidates.candidates.csv

```

### Trained Models

The trained model files can be found in the GitHub code release [here](https://github.com/Merck/deepbgc/releases).
