from __future__ import (
    print_function,
    division,
    absolute_import,
)
import subprocess
import os

import pandas as pd

from deepbgc.data import PFAM_DB_FILE_NAME, PFAM_DB_VERSION, PFAM_CLANS_FILE_NAME
from Bio import SeqIO, SearchIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
import numpy as np
from deepbgc import util
import logging
from distutils.spawn import find_executable
from datetime import datetime


class HmmscanPfamRecordAnnotator(object):
    def __init__(self, record, tmp_path_prefix, max_evalue=0.01, db_path=None, clans_path=None):
        self.record = record
        self.tmp_path_prefix = tmp_path_prefix
        self.db_path = db_path or util.get_downloaded_file_path(PFAM_DB_FILE_NAME, versioned=False)
        self.clans_path = clans_path or util.get_downloaded_file_path(PFAM_CLANS_FILE_NAME, versioned=False)
        self.max_evalue = max_evalue

    def _write_proteins(self, proteins, protein_path):
        records = []
        for feature in proteins:
            translation = feature.extract(self.record.seq).translate()
            records.append(SeqRecord(translation, util.get_protein_id(feature), description=''))
        SeqIO.write(records, protein_path, 'fasta')

    def _get_pfam_loc(self, query_start, query_end, feature):
        if feature.strand == 1:
            start = feature.location.start + 3 * query_start
            end = feature.location.start + 3 * query_end
        elif feature.strand == -1:
            end = feature.location.end - 3 * query_start
            start = feature.location.end - 3 * query_end
        else:
            raise ValueError('Invalid strand for feature: {}'.format(feature))
        return FeatureLocation(start, end, strand=feature.strand)

    def _run_hmmscan(self, protein_path, domtbl_path):
        if not find_executable('hmmscan'):
            raise RuntimeError("HMMER hmmscan needs to be installed and available on PATH "
                               "in order to detect Pfam domains.")

        p = subprocess.Popen(
            ['hmmscan', '--nobias', '--domtblout', domtbl_path, self.db_path, protein_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        out, err = p.communicate()
        if p.returncode or not os.path.exists(domtbl_path):
            logging.warning('== HMMER hmmscan Error: ================')
            logging.warning(err.strip())
            logging.warning('== End HMMER hmmscan Error. ============')
            raise Exception("Unexpected error detecting protein domains using HMMER hmmscan")

    def annotate(self):

        proteins = util.get_protein_features(self.record)
        proteins_by_id = util.get_proteins_by_id(proteins)
        domtbl_path = self.tmp_path_prefix + '.pfam.domtbl.txt'

        if not proteins:
            logging.warning('No	proteins in sequence %s, skipping protein domain detection', self.record.id)
            return

        if util.is_valid_hmmscan_output(domtbl_path):
            cached = True
            logging.info('Reusing already existing HMMER hmmscan result: %s', domtbl_path)
        else:
            cached = False
            protein_path = self.tmp_path_prefix + '.pfam.proteins.fa'

            # Write proteins to fasta file
            self._write_proteins(proteins, protein_path)

            logging.info('Detecting Pfam domains in "%s" using HMMER hmmscan, this might take a while...', self.record.id)
            start_time = datetime.now()
            self._run_hmmscan(protein_path, domtbl_path)

            logging.info('HMMER hmmscan Pfam detection done in %s', util.print_elapsed_time(start_time))

        # Read domain matches in all proteins
        queries = SearchIO.parse(domtbl_path, 'hmmscan3-domtab')

        # Read descriptions from Pfam clan TSV
        pfam_descriptions = self._get_pfam_descriptions()

        # Extract all matched domain hits
        num = 0
        pfam_ids = set()
        for query in queries:
            if cached and query.id not in proteins_by_id:
                raise ValueError('Found invalid protein ID "{}" in cached HMMER hmmscan result for record "{}", '
                                     'disable caching or delete the file: {}'.format(query.id, self.record.id, domtbl_path))
            protein = proteins_by_id.get(query.id)
            for hit in query.hits:
                best_index = np.argmin([hsp.evalue for hsp in hit.hsps])
                best_hsp = hit.hsps[best_index]
                pfam_id = hit.accession
                evalue = float(best_hsp.evalue)
                if evalue > self.max_evalue:
                    continue
                location = self._get_pfam_loc(best_hsp.query_start, best_hsp.query_end, protein)
                qualifiers = {
                    'db_xref': [pfam_id],
                    'evalue': evalue,
                    'locus_tag': [query.id],
                    'database': [PFAM_DB_VERSION],
                }
                short_pfam_id = pfam_id.rsplit('.', 1)[0]
                description = pfam_descriptions.get(short_pfam_id)
                if description:
                    qualifiers['description'] = [description]
                pfam = SeqFeature(
                    location=location,
                    id=pfam_id,
                    type="PFAM_domain",
                    qualifiers=qualifiers
                )
                self.record.features.append(pfam)
                num += 1
                pfam_ids.add(pfam_id)

        util.sort_record_features(self.record)
        logging.info('Added %s Pfam domains (%s unique PFAM_IDs)', num, len(pfam_ids))

    def _get_clans(self):
        clans = pd.read_csv(self.clans_path, sep='\t', header=None)
        clans.columns = ['pfam_id', 'clan_id', 'clan_name', 'pfam_name', 'description']
        return clans.set_index('pfam_id')

    def _get_pfam_descriptions(self):
        return self._get_clans()['description'].to_dict()
