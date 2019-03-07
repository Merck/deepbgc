from Bio import SeqIO

from deepbgc import util
from deepbgc.output.writer import OutputWriter
import os


class BGCGenbankWriter(OutputWriter):

    def __init__(self, out_path):
        super(BGCGenbankWriter, self).__init__(out_path)
        self.fd = open(self.out_path, 'w')

    @classmethod
    def get_description(cls):
        return 'Sequences and features of all detected BGCs in GenBank format'

    @classmethod
    def get_name(cls):
        return 'genbank'

    def write(self, record):
        clusters = util.get_cluster_features(record)
        for cluster in clusters:
            cluster_record = util.extract_cluster_record(cluster, record)
            SeqIO.write(cluster_record, self.fd, 'genbank')

    def close(self):
        self.fd.close()
