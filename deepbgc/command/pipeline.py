from __future__ import (
    print_function,
    division,
    absolute_import,
)

import logging

import deepbgc.util
from deepbgc.command.base import BaseCommand
import os
from deepbgc import util
from Bio import SeqIO

from deepbgc.output.bgc_genbank import BGCGenbankWriter
from deepbgc.output.evaluation.pr_plot import PrecisionRecallPlotWriter
from deepbgc.output.evaluation.roc_plot import ROCPlotWriter
from deepbgc.output.readme import ReadmeWriter
from deepbgc.pipeline.annotator import DeepBGCAnnotator
from deepbgc.pipeline.detector import DeepBGCDetector
from deepbgc.pipeline.classifier import DeepBGCClassifier
from deepbgc.output.genbank import GenbankWriter
from deepbgc.output.evaluation.bgc_region_plot import BGCRegionPlotWriter
from deepbgc.output.cluster_tsv import ClusterTSVWriter
from deepbgc.output.evaluation.pfam_score_plot import PfamScorePlotWriter
from deepbgc.output.pfam_tsv import PfamTSVWriter
from deepbgc.output.antismash_json import AntismashJSONWriter


class PipelineCommand(BaseCommand):
    command = 'pipeline'

    help = """Run DeepBGC pipeline: Preparation, BGC detection, BGC classification and generate the report directory.
    
Examples:
    
  # Show detailed help 
  deepbgc pipeline --help 
    
  # Detect BGCs in a nucleotide FASTA sequence using DeepBGC model 
  deepbgc pipeline sequence.fa
  
  # Detect BGCs using the ClusterFinder GeneBorder detection model and a higher score threshold
  deepbgc pipeline --detector clusterfinder_geneborder --score 0.8 sequence.fa
  
  # Add additional clusters detected using DeepBGC model with a strict score threshold
  deepbgc pipeline --continue --output sequence/ --label deepbgc_90_score --score 0.9 sequence/sequence.full.gbk
  """

    LOG_FILENAME = 'LOG.txt'
    PLOT_DIRNAME = 'evaluation'
    TMP_DIRNAME = 'tmp'

    def add_arguments(self, parser):

        parser.add_argument(dest='inputs', nargs='+', help="Input sequence file path (FASTA, GenBank, Pfam CSV).")

        parser.add_argument('-o', '--output', required=False, help="Custom output directory path.")
        parser.add_argument('--limit-to-record', action='append', help="Process only specific record ID. Can be provided multiple times.")
        parser.add_argument('--minimal-output', dest='is_minimal_output', action='store_true', default=False,
                            help="Produce minimal output with just the GenBank sequence file.")
        group = parser.add_argument_group('BGC detection options', '')
        no_models_message = 'run "deepbgc download" to download models'
        detector_names = util.get_available_models('detector')
        group.add_argument('-d', '--detector', dest='detectors', action='append', default=[],
                           help="Trained detection model name ({}). "
                                "Can be provided multiple times (-d first -d second).".format(', '.join(detector_names) or no_models_message))
        group.add_argument('--no-detector', action='store_true', help="Disable BGC detection.")
        group.add_argument('-l', '--label', dest='labels', action='append', default=[], help="Label for detected clusters (equal to --detector by default). "
                                                                                             "If multiple detectors are provided, a label should be provided for each one.")
        group.add_argument('-s', '--score', default=0.5, type=float,
                            help="Average protein-wise DeepBGC score threshold for extracting BGC regions from Pfam sequences.")
        group.add_argument('--merge-max-protein-gap', default=0, type=int, help="Merge detected BGCs within given number of proteins.")
        group.add_argument('--merge-max-nucl-gap', default=0, type=int, help="Merge detected BGCs within given number of nucleotides.")
        group.add_argument('--min-nucl', default=1, type=int, help="Minimum BGC nucleotide length.")
        group.add_argument('--min-proteins', default=1, type=int, help="Minimum number of proteins in a BGC.")
        group.add_argument('--min-domains', default=1, type=int, help="Minimum number of protein domains in a BGC.")
        group.add_argument('--min-bio-domains', default=0, type=int, help="Minimum number of known biosynthetic protein domains in a BGC (from antiSMASH ClusterFinder).")

        group = parser.add_argument_group('BGC classification options', '')
        classifier_names = util.get_available_models('classifier')
        group.add_argument('-c', '--classifier', dest='classifiers', action='append', default=[],
                            help="Trained classification model name ({}). "
                                 "Can be provided multiple times (-c first -c second).".format(', '.join(classifier_names) or no_models_message))
        group.add_argument('--no-classifier', action='store_true', help="Disable BGC classification.")
        group.add_argument('--classifier-score', default=0.5, type=float,
                            help="DeepBGC classification score threshold for assigning classes to BGCs (inclusive).")

    def run(self, inputs, output, detectors, no_detector, labels, classifiers, no_classifier,
            is_minimal_output, limit_to_record, score, classifier_score, merge_max_protein_gap, merge_max_nucl_gap, min_nucl,
            min_proteins, min_domains, min_bio_domains):
        if not detectors:
            detectors = ['deepbgc']
        if not classifiers:
            classifiers = ['product_class', 'product_activity']
        if not output:
            # if not specified, set output path to name of first input file without extension
            output, _ = os.path.splitext(os.path.basename(os.path.normpath(inputs[0])))

        if not os.path.exists(output):
            os.mkdir(output)

        # Save log to LOG.txt file
        logger = logging.getLogger('')
        logger.addHandler(logging.FileHandler(os.path.join(output, self.LOG_FILENAME)))

        # Define report dir paths
        tmp_path = os.path.join(output, self.TMP_DIRNAME)
        evaluation_path = os.path.join(output, self.PLOT_DIRNAME)
        output_file_name = os.path.basename(os.path.normpath(output))

        steps = []
        steps.append(DeepBGCAnnotator(tmp_dir_path=tmp_path))
        if not no_detector:
            if not labels:
                labels = [None] * len(detectors)
            elif len(labels) != len(detectors):
                raise ValueError('A separate label should be provided for each of the detectors: {}'.format(detectors))

            for detector_name, label in zip(detectors, labels):
                steps.append(DeepBGCDetector(
                    detector=detector_name,
                    label=label,
                    score_threshold=score,
                    merge_max_protein_gap=merge_max_protein_gap,
                    merge_max_nucl_gap=merge_max_nucl_gap,
                    min_nucl=min_nucl,
                    min_proteins=min_proteins,
                    min_domains=min_domains,
                    min_bio_domains=min_bio_domains
                ))

        writers = []
        writers.append(GenbankWriter(out_path=os.path.join(output, output_file_name+'.full.gbk')))
        #writers.append(AntismashJSONWriter(out_path=os.path.join(output, output_file_name + '.antismash.json')))
        is_evaluation = False
        if not is_minimal_output:
            writers.append(BGCGenbankWriter(out_path=os.path.join(output, output_file_name+'.bgc.gbk')))
            writers.append(ClusterTSVWriter(out_path=os.path.join(output, output_file_name+'.bgc.tsv')))
            writers.append(PfamTSVWriter(out_path=os.path.join(output, output_file_name+'.pfam.tsv')))

            is_evaluation = True
            writers.append(PfamScorePlotWriter(out_path=os.path.join(evaluation_path, output_file_name + '.score.png')))
            writers.append(BGCRegionPlotWriter(out_path=os.path.join(evaluation_path, output_file_name+'.bgc.png')))
            writers.append(ROCPlotWriter(out_path=os.path.join(evaluation_path, output_file_name+'.roc.png')))
            writers.append(PrecisionRecallPlotWriter(out_path=os.path.join(evaluation_path, output_file_name+'.pr.png')))

        writers.append(ReadmeWriter(out_path=os.path.join(output, 'README.txt'), root_path=output, writers=writers))

        if not no_classifier:
            for classifier_name in classifiers:
                steps.append(DeepBGCClassifier(classifier=classifier_name, score_threshold=classifier_score))

        # Create temp and evaluation dir
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        if is_evaluation:
            if not os.path.exists(evaluation_path):
                os.mkdir(evaluation_path)

        record_idx = 0
        for input_path in inputs:
            fmt = deepbgc.util.guess_format(input_path)
            if not fmt:
                raise NotImplementedError("Sequence file type not recognized: {}, ".format(input_path),
                                          "Please provide a GenBank or FASTA sequence "
                                          "with an appropriate file extension.")
            records = SeqIO.parse(input_path, fmt)
            for record in records:
                if limit_to_record and record.id not in limit_to_record:
                    logging.debug('Skipping record %s not matching filter %s', record.id, limit_to_record)
                    continue

                record_idx += 1
                logging.info('='*80)
                logging.info('Processing record #%s: %s', record_idx, record.id)
                for step in steps:
                    step.run(record)

                logging.info('Saving processed record %s', record.id)
                for writer in writers:
                    writer.write(record)

        logging.info('=' * 80)
        for step in steps:
            step.print_summary()

        for writer in writers:
            writer.close()

        logging.info('='*80)
        logging.info('Saved DeepBGC result to: {}'.format(output))
