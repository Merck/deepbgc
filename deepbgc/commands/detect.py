import pandas as pd
from deepbgc.commands.base import BaseCommand
from deepbgc.converter import SequenceToPfamCSVConverter
import os
from deepbgc.detector import DeepBGCDetector

SCORE_COLUMN = 'deepbgc_score'

class DetectCommand(BaseCommand):
    command = 'detect'
    help = """Detect BGCs in a genomic sequence.
    
Examples:
    
  # Detect BGCs in FASTA sequence with default settings
  deepbgc detect --model myModel.pkl --output myDetections/ --pfam Pfam-A.hmm inputSequence.fa
    
  # Detect BGCs with >0.9 score in existing Pfam CSV sequence
  deepbgc detect --model myModel.pkl --output myStrictDetections/ -s 0.9 myDetections/myDetections.pfam.csv
  """

    def __init__(self, args):
        super().__init__(args)
        self.output_path = args.output
        self.output_basename = os.path.basename(self.output_path)
        self.input_path = args.input
        self.model_path = args.model
        self.score_threshold = args.score
        self.converter = SequenceToPfamCSVConverter(db_path=args.pfam)

    @classmethod
    def add_subparser(cls, subparsers):
        parser = super().add_subparser(subparsers)

        parser.add_argument('-o', '--output', required=True, help="Output folder path.")
        parser.add_argument('-m', '--model', required=True, help="Trained detection model file path.")
        parser.add_argument('-p', '--pfam', required=False, help="Pfam DB (Pfam-A.hmm) file path.")
        parser.add_argument('-s', '--score', default=0.5, type=float, help="Average protein-wise DeepBGC score threshold for extracting BGC regions from domain sequences.")
        parser.add_argument(dest='input', help="Input pfam CSV file path.")

    def _outpath(self, suffix, extension):
        return os.path.join(self.output_path, '{}.{}.{}'.format(self.output_basename, suffix, extension))

    def run(self):
        try:
            os.makedirs(self.output_path, exist_ok=True)
        except FileExistsError:
            raise AttributeError("Output directory already exists: {}".format(self.output_path))
        except Exception as e:
            raise AttributeError("Output directory not writable: {}".format(self.output_path), e)

        domain_path = self._outpath('pfam', 'csv')
        if not self.converter.convert(self.input_path, domain_path):
            # Input was already a pfam CSV file, use original path
            domain_path = self.input_path

        domains = pd.read_csv(domain_path)
        detector = DeepBGCDetector(model=self.model_path)

        candidates = detector.detect(domains, score_threshold=self.score_threshold)

        cand_path = self._outpath('candidates', 'csv')
        candidates.to_csv(cand_path, index=False)
        print('Saved {} detected BGCs to {}'.format(len(candidates), cand_path))


def sequence_id_from_filename(path):
    """
    Create a basic sequence_id from a file name without extension
    :param path: Path of file
    :return: file name without extension that can be used as sequence_id
    """
    return os.path.splitext(os.path.basename(path))[0]

