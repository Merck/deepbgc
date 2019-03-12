import logging

from deepbgc import util
from deepbgc.output.writer import TSVWriter


class PfamTSVWriter(TSVWriter):

    @classmethod
    def get_description(cls):
        return 'Table of Pfam domains (pfam_id) from given sequence (sequence_id) in genomic order, with BGC detection scores'

    @classmethod
    def get_name(cls):
        return 'pfam-tsv'

    def record_to_df(self, record):
        df = util.create_pfam_dataframe(record, add_scores=True, add_in_cluster=True)
        logging.debug('Writing %s Pfams to: %s', len(df), self.out_path)
        return df
