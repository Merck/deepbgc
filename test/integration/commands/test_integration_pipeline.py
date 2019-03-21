from Bio import SeqIO

from deepbgc import util
from deepbgc.main import run
from test.test_util import get_test_file
import os


def test_integration_pipeline_default(tmpdir):
    tmpdir = str(tmpdir)
    report_dir = os.path.join(tmpdir, 'report')
    run(['pipeline', '--output', report_dir, get_test_file('BGC0000015.fa')])

    files = os.listdir(report_dir)
    for file in files:
        print(file)

    assert 'README.txt' in files
    assert 'report.bgc.gbk' in files
    assert 'report.bgc.tsv' in files
    assert 'report.full.gbk' in files
    assert 'report.pfam.tsv' in files

    evaluation_dir = os.path.join(report_dir, 'evaluation')
    files = os.listdir(evaluation_dir)
    for file in files:
        print(file)

    assert 'report.bgc.png' in files
    assert 'report.score.png' in files

    records = list(SeqIO.parse(os.path.join(report_dir, 'report.full.gbk'), 'genbank'))
    assert len(records) == 2

    record = records[0]
    cluster_features = util.get_cluster_features(record)
    assert len(cluster_features) >= 1

    record = records[1]
    cluster_features = util.get_cluster_features(record)
    assert len(cluster_features) >= 1

    cluster_records = list(SeqIO.parse(os.path.join(report_dir, 'report.bgc.gbk'), 'genbank'))
    assert len(cluster_records) >= 2

