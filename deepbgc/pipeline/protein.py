from __future__ import (
    print_function,
    division,
    absolute_import,
)
import subprocess
import os
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
import logging
from distutils.spawn import find_executable


class ProdigalProteinRecordAnnotator(object):
    def __init__(self, record, tmp_path_prefix):
        self.record = record
        self.tmp_path_prefix = tmp_path_prefix

    def annotate(self):
        logging.info('Finding genes in record: %s', self.record.id)

        nucl_path = self.tmp_path_prefix + '.prodigal.nucl.fa'
        SeqIO.write(self.record, nucl_path, 'fasta')

        protein_path = self.tmp_path_prefix + '.prodigal.proteins.fa'

        if not find_executable('prodigal'):
            raise Exception("Prodigal needs to be installed and available on PATH in order to detect genes.")

        logging.debug('Detecting genes using Prodigal...')

        p = subprocess.Popen(
            ['prodigal', '-i', nucl_path, '-a', protein_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        out, err = p.communicate()
        if p.returncode or not os.path.exists(protein_path):
            logging.warning('== Prodigal Error: ================')
            logging.warning(err.strip())
            logging.warning('== End Prodigal Error. ============')

            if 'Sequence must be' in err:
                logging.warning('No proteins detected in short sequence, moving on.')
            elif os.stat(protein_path).st_size == 0:
                raise ValueError("Prodigal produced empty output, make sure to use a DNA sequence.")
            else:
                raise ValueError("Unexpected error detecting genes using Prodigal")

        proteins = SeqIO.parse(protein_path, 'fasta')
        for protein in proteins:
            splits = protein.description.split('#')
            try:
                start = int(splits[1]) - 1
                end = int(splits[2])
                strand = int(splits[3])
            except Exception as e:
                raise ValueError('Invalid Prodigal protein description: "{}"'.format(protein.description), e)
            location = FeatureLocation(start, end, strand=strand)
            protein = SeqFeature(location=location, id=protein.id, type="CDS",
                        qualifiers={'locus_tag': ['{}_{}'.format(self.record.id, protein.id)]})
            self.record.features.append(protein)

