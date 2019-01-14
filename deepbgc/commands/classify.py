import pandas as pd
from deepbgc.commands.base import BaseCommand
import os
import pickle
import numpy as np

SCORE_COLUMN = 'deepbgc_score'

class ClassifyCommand(BaseCommand):
    command = 'classify'
    help = """Classify BGCs into one or more classes.
    
Examples:
    
  deepbgc classify --model myClassifier.pkl --output classes.csv inputSequence.fa
  """

    def __init__(self, args):
        super().__init__(args)
        self.output_path = args.output
        self.input_path = args.input
        self.model_path = args.model

    @classmethod
    def add_subparser(cls, subparsers):
        parser = super().add_subparser(subparsers)

        parser.add_argument('-o', '--output', required=True, help="Output CSV file path.")
        parser.add_argument('-m', '--model', required=True, help="Trained classification model file path.")
        parser.add_argument(dest='input', help="Input candidate CSV file path.")

    def run(self):
        candidates = pd.read_csv(self.input_path)
        if 'candidate_hash' not in candidates.columns:
            raise AttributeError('Input CSV is not a candidate CSV file, "candidate_hash" column should be present.')

        candidates = candidates.set_index('candidate_hash')

        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)

        vectors = domain_set_vectors(candidates)

        predictions = predict_classes(vectors, model)
        predictions.to_csv(self.output_path, index=False)
        print('Saved {} predictions to {}'.format(len(predictions), self.output_path))


def domain_set_vectors(candidates):
    candidate_pfam_ids = [pfam_ids.split(';') for pfam_ids in candidates['pfam_ids']]
    unique_pfam_ids = sorted(list(set([p for ids in candidate_pfam_ids for p in ids])))
    print('Getting domain set vectors for {} candidates with {} unique Pfam IDs...'.format(len(candidates), len(unique_pfam_ids)))
    vectors = pd.DataFrame(np.zeros((len(candidates), len(unique_pfam_ids))), columns=unique_pfam_ids)
    for i, pfam_ids in enumerate(candidate_pfam_ids):
        vectors.iloc[i][pfam_ids] = 1
    return vectors


def predict_classes(samples, model, add_classes_list=True):
    # Set missing columns to 0
    if not hasattr(model, 'input_columns'):
        raise AttributeError('Trained model does not contain the "input_columns" attribute.')
    if not hasattr(model, 'label_columns'):
        raise AttributeError('Trained model does not contain the "label_columns" attribute.')

    missing_columns = set(model.input_columns).difference(samples.columns)
    for col in missing_columns:
        samples[col] = 0
    #print('Missing columns:\n{}'.format(sorted(list(missing_columns))))
    print('Warning: Setting {} missing columns to 0'.format(len(missing_columns)))
    samples = samples[model.input_columns]

    results = np.array([r[:,1] for r in model.predict_proba(samples.values)]).transpose()
    predictions = pd.DataFrame(results, index=samples.index, columns=model.label_columns)
    if add_classes_list:
        predictions['classes'] = [';'.join(model.label_columns[x >= 0.5]) for x in results]

    return predictions

def sequence_id_from_filename(path):
    """
    Create a basic sequence_id from a file name without extension
    :param path: Path of file
    :return: file name without extension that can be used as sequence_id
    """
    return os.path.splitext(os.path.basename(path))[0]

