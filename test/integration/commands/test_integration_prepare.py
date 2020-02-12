from Bio import SeqIO
from deepbgc.main import run
from test.test_util import get_test_file, assert_sorted_features
import os
from deepbgc import util
import pandas as pd
import pytest


@pytest.mark.parametrize("input_file", [
    "BGC0000015.fa",
    "BGC0000015.fa.gz",
])
def test_integration_prepare_default(tmpdir, input_file):
    tmpdir = str(tmpdir)
    outgbk = os.path.join(tmpdir, 'outfile.gbk')
    outtsv = os.path.join(tmpdir, 'outfile.tsv')
    run(['prepare', '--output-gbk', outgbk, '--output-tsv', outtsv, get_test_file(input_file)])

    records = list(SeqIO.parse(outgbk, 'genbank'))

    # Check GenBank file
    assert len(records) == 2

    record = records[0]
    assert_sorted_features(record)
    proteins = util.get_protein_features(record)
    pfams = util.get_pfam_features(record)

    assert len(proteins) == 18
    print([util.get_protein_id(f) for f in proteins])
    assert len(pfams) == 111

    record = records[1]
    assert_sorted_features(record)
    proteins = util.get_protein_features(record)
    pfams = util.get_pfam_features(record)

    assert len(proteins) == 27
    assert len(pfams) == 36

    # Check Pfam TSV file
    domains = pd.read_csv(outtsv, sep='\t')
    records = domains.groupby('sequence_id')

    assert len(records) == 2

    record = records.get_group('BGC0000015.1')
    print(record['protein_id'].unique())
    # some of the proteins do not have any Pfam domains so they are not present
    assert len(record['protein_id'].unique()) == 17
    assert len(record) == 111

    record = records.get_group('BGC0000015.2')
    # some of the proteins do not have any Pfam domains so they are not present
    assert len(record['protein_id'].unique()) == 11
    assert len(record) == 36


def test_integration_prepare_protein(tmpdir):
    tmpdir = str(tmpdir)
    outgbk = os.path.join(tmpdir, 'outfile.gbk')
    outtsv = os.path.join(tmpdir, 'outfile.tsv')
    run(['prepare', '--protein', '--output-gbk', outgbk, '--output-tsv', outtsv, get_test_file('BGC0000015.protein.fa')])

    # Check GenBank file
    records = list(SeqIO.parse(outgbk, 'genbank'))

    assert len(records) == 1

    record = records[0]
    assert_sorted_features(record)
    proteins = util.get_protein_features(record)
    pfams = util.get_pfam_features(record)

    assert len(proteins) == 30
    print([util.get_protein_id(f) for f in proteins])
    assert len(pfams) == 155

    # Check Pfam TSV file
    domains = pd.read_csv(outtsv, sep='\t')
    records = domains.groupby('sequence_id')

    assert len(records) == 1

    record = records.get_group('BGC0000015.protein')
    print(record['protein_id'].unique())
    # some of the proteins do not have any Pfam domains so they are not present
    assert len(record['protein_id'].unique()) == 30
    assert len(record) == 155
