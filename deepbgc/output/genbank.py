from Bio import SeqIO
from deepbgc.output.writer import OutputWriter
import os


class GenbankWriter(OutputWriter):
    def __init__(self, out_path):
        super(GenbankWriter, self).__init__(out_path)
        self.tmp_out_path = self.out_path + '.part'
        self.fd = open(self.tmp_out_path, 'w')

    @classmethod
    def get_description(cls):
        return 'Fully annotated input sequence with proteins, Pfam domains (PFAM_domain features) and BGCs (cluster features)'

    @classmethod
    def get_name(cls):
        return 'genbank'

    def write(self, record):
        SeqIO.write(record, self.fd, 'genbank')

    def close(self):
        self.fd.close()
        # Move file.gbk.part to file.gbk
        os.rename(self.tmp_out_path, self.out_path)
