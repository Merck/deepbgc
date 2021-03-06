from deepbgc import util
from deepbgc.output.writer import OutputWriter
import logging
import pandas as pd

class ClusterTSVWriter(OutputWriter):
    @classmethod
    def get_description(cls):
        return 'Table of detected BGCs and their properties'

    @classmethod
    def get_name(cls):
        return 'bgc-tsv'

    def __init__(self, out_path):
        super(ClusterTSVWriter, self).__init__(out_path)
        self.dfs = []

    def record_to_df(self, record):
        df = util.create_cluster_dataframe(record, add_classification=True)
        df.insert(0, 'sequence_id', record.id)
        return df

    def write(self, record):
        df = self.record_to_df(record)
        if df.empty:
            return
        self.dfs.append(df)

    def close(self):
        df = pd.concat(self.dfs, sort=False)
        logging.debug('Writing %s BGCs to: %s', len(df), self.out_path)
        df.to_csv(self.out_path, index=False, sep='\t')
