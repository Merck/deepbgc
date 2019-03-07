from __future__ import (
    print_function,
    division,
    absolute_import,
)

import logging

from deepbgc import util
from deepbgc.command.base import BaseCommand
from Bio import SeqIO
import os
import shutil

from deepbgc.output.genbank import GenbankWriter
from deepbgc.output.pfam_tsv import PfamTSVWriter
from deepbgc.pipeline.annotator import DeepBGCAnnotator


class PrepareCommand(BaseCommand):
    command = 'prepare'
    help = """Prepare genomic sequence by annotating proteins and Pfam domains.
    
Examples:
    
  # Show detailed help 
  deepbgc prepare --help 
    
  # Detect proteins and pfam domains in a FASTA sequence and save the result as GenBank file 
  deepbgc prepare --output sequence.prepared.gbk sequence.fa
  """

    def add_arguments(self, parser):
        parser.add_argument(dest='inputs', nargs='+', help="Input sequence file path(s) (FASTA/GenBank).")
        group = parser.add_argument_group('required arguments', '')
        group.add_argument('--output-gbk', required=False, help="Output GenBank file path.")
        group.add_argument('--output-tsv', required=False, help="Output TSV file path.")

    def run(self, inputs, output_gbk, output_tsv):
        first_output = output_gbk or output_tsv
        if not first_output:
            raise ValueError('Specify at least one of --output-gbk or --output-tsv')

        tmp_dir_path = first_output + '.tmp'
        logging.debug('Using TMP dir: %s', tmp_dir_path)
        if not os.path.exists(tmp_dir_path):
            os.mkdir(tmp_dir_path)

        prepare_step = DeepBGCAnnotator(tmp_dir_path=tmp_dir_path)

        writers = []
        if output_gbk:
            writers.append(GenbankWriter(out_path=output_gbk))
        if output_tsv:
            writers.append(PfamTSVWriter(out_path=output_tsv))

        num_records = 0
        for input_path in inputs:
            fmt = util.guess_format(input_path)
            if not fmt:
                raise NotImplementedError("Sequence file type not recognized: {}, ".format(input_path),
                                          "Please provide a GenBank or FASTA sequence "
                                          "with an appropriate file extension.")
            records = SeqIO.parse(input_path, fmt)
            for record in records:
                prepare_step.run(record)
                for writer in writers:
                    writer.write(record)
                num_records += 1

        logging.debug('Removing TMP directory: %s', tmp_dir_path)
        shutil.rmtree(tmp_dir_path)

        prepare_step.print_summary()

        for writer in writers:
            writer.close()

        logging.info('Saved %s fully annotated records to %s', num_records, first_output)

