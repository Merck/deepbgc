from deepbgc.pipeline.protein import ProdigalProteinRecordAnnotator
from Bio import SeqIO
from deepbgc import util
from test.test_util import get_test_file, assert_sorted_features
import os


def test_integration_protein_annotator(tmpdir):
    tmpdir = str(tmpdir)
    tmppath = os.path.join(tmpdir, 'test')
    records = SeqIO.parse(get_test_file('BGC0000015.fa'), format='fasta')
    record = next(records)

    annotator = ProdigalProteinRecordAnnotator(record=record, tmp_path_prefix=tmppath)
    annotator.annotate()
    proteins = util.get_protein_features(record)

    assert len(proteins) == 18

    protein = proteins[0]
    assert protein.location.start == 3
    assert protein.location.end == 1824
    assert protein.id == 'BGC0000015.1_1'
    assert protein.qualifiers.get('locus_tag') == ['BGC0000015.1_BGC0000015.1_1']

    assert_sorted_features(record)
