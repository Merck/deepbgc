from __future__ import (
    print_function,
    division,
    absolute_import,
)

import logging

from deepbgc import util
from deepbgc.command.base import BaseCommand
from deepbgc.models.wrapper import SequenceModelWrapper


class TrainCommand(BaseCommand):
    command = 'train'
    help = """Train a BGC detector/classifier on a set of BGC samples.
    
Examples:
    
  # Train a detector using pre-processed samples in Pfam CSV format. 
  deepbgc train --model deepbgc.json --output MyDeepBGCDetector.pkl BGCs.pfam.tsv negatives.pfam.tsv
    
  # Train a BGC classifier using a TSV classes file and a set of BGC samples in Pfam TSV format and save the trained classifier to a file. 
  deepbgc train --model random_forest.json --output MyDeepBGCClassifier.pkl --classes path/to/BGCs.classes.csv BGCs.pfam.tsv
  """

    def add_arguments(self, parser):
        parser.add_argument("-m", "--model", dest="model", required=True,
                            help="Path to JSON model config file.")
        parser.add_argument('-t', '--target', required=False, default='in_cluster',
                            help="Target column to predict in sequence prediction.")
        parser.add_argument('-o', '--output', required=True,
                            help="Output trained model file path.")
        parser.add_argument('-l', '--log', required=False,
                            help="Progress log output path (e.g. TensorBoard).")
        parser.add_argument('-c', '--classes', required=False,
                            help="Class TSV file path - train a sequence classifier "
                                 "using provided classes (binary columns), indexed by sequence_id column.")
        parser.add_argument("--config", nargs=2, action='append', default=[],
                            help="Variables in model JSON file to replace (e.g. --config PFAM2VEC path/to/pfam2vec.csv).")
        parser.add_argument('-v', '--validation', action='append', required=False,
                            help="Validation sequence file path. Repeat to specify multiple files.")
        parser.add_argument("--verbose", dest="verbose", required=False, default=2, type=int,
                            help="Verbosity level (0=none, 1=progress bar, 2=once per epoch).", metavar="INT")
        parser.add_argument(dest='inputs', nargs='+', help="Training sequences (Pfam TSV) file paths.")

    def run(self, inputs, output, model, target, classes, config, log, validation, verbose):

        pipeline = SequenceModelWrapper.from_config(model, vars=dict(config))

        if classes:
            class_df = util.read_compatible_csv(classes).set_index('sequence_id').astype('int8')
            train_samples, train_y = util.read_samples_with_classes(inputs, class_df)
            logging.info('Training samples:\n%s', train_y.sum())

            validation_samples, validation_y = util.read_samples_with_classes(validation, class_df)
            if len(validation_y):
                logging.info('Validation samples:\n%s', validation_y.sum())
        else:
            train_samples, train_y = util.read_samples(inputs, target)
            validation_samples, validation_y = util.read_samples(validation, target)
        pipeline.fit(
            samples=train_samples,
            y=train_y,
            debug_progress_path=log,
            validation_samples=validation_samples,
            validation_y=validation_y,
            verbose=verbose
        )

        pipeline.save(output)

        if log:
            logging.info('Progress log saved to: %s', log)
        logging.info('Trained model saved to: %s', output)


