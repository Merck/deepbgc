from deepbgc import util
from deepbgc.output.writer import TSVWriter
import logging


class ClusterTSVWriter(TSVWriter):
    @classmethod
    def get_description(cls):
        return 'Table of detected BGCs and their properties'

    @classmethod
    def get_name(cls):
        return 'bgc-tsv'

    def record_to_df(self, record):
        df = util.create_cluster_dataframe(record, add_classification=True)
        df.insert(0, 'sequence_id', record.id)
        logging.debug('Writing %s BGCs to: %s', len(df), self.out_path)
        return df
