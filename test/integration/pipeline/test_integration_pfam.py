from deepbgc.pipeline.pfam import HmmscanPfamRecordAnnotator
from Bio import SeqIO
from deepbgc import util
from test.test_util import get_test_file, assert_sorted_features
import os


def test_integration_pfam_annotator(tmpdir):
    tmpdir = str(tmpdir)
    tmppath = os.path.join(tmpdir, 'test')
    records = SeqIO.parse(get_test_file('BGC0000015.gbk'), format='genbank')
    record = next(records)

    annotator = HmmscanPfamRecordAnnotator(
        record=record,
        tmp_path_prefix=tmppath,
        db_path=get_test_file('Pfam-A.PF00005.hmm'),
        clans_path=get_test_file('Pfam-A.PF00005.clans.tsv')
    )
    annotator.annotate()
    pfams = util.get_pfam_features(record)

    assert len(pfams) == 2

    pfam = pfams[0]
    assert pfam.location.start == 249
    assert pfam.location.end == 696
    assert pfam.location.strand == -1
    assert pfam.qualifiers.get('PFAM_ID') == ['PF00005']
    assert pfam.qualifiers.get('locus_tag') == ['AAK73498.1']
    assert pfam.qualifiers.get('description') == ['ABC transporter']
    assert pfam.qualifiers.get('database') == ['Pfam-A.31.0.hmm']

    assert_sorted_features(record)