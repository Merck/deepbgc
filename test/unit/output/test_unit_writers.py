from deepbgc.output.bgc_genbank import BGCGenbankWriter
from deepbgc.output.evaluation.pr_plot import PrecisionRecallPlotWriter
from deepbgc.output.evaluation.roc_plot import ROCPlotWriter
from deepbgc.output.genbank import GenbankWriter
from deepbgc.output.evaluation.bgc_region_plot import BGCRegionPlotWriter
from deepbgc.output.cluster_tsv import ClusterTSVWriter
from deepbgc.output.evaluation.pfam_score_plot import PfamScorePlotWriter
from deepbgc.output.pfam_tsv import PfamTSVWriter
from deepbgc.output.antismash_json import AntismashJSONWriter
from deepbgc import util
from deepbgc.data import PFAM_DB_VERSION
import collections
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Alphabet import generic_dna
import os
import pytest


class WriterTest:
    def __init__(self, cls, path):
        self.cls = cls
        self.path = path

    def __str__(self):
        return str(self.cls)

    def __repr__(self):
        return str(self.cls)


WRITERS = [
    BGCGenbankWriter,
    PrecisionRecallPlotWriter,
    ROCPlotWriter,
    GenbankWriter,
    BGCRegionPlotWriter,
    ClusterTSVWriter,
    PfamScorePlotWriter,
    PfamTSVWriter,
    AntismashJSONWriter
]

@pytest.fixture
def processed_record(detector_name='deepbgc', detector_label='deepbgc', score_threshold=0.5):
    comment_key = util.format_detector_meta_key(detector_label)
    record = SeqRecord(Seq('ACTGCTCGACTGATT', alphabet=generic_dna))
    record.annotations['structured_comment'] = collections.OrderedDict()
    record.annotations['structured_comment'][comment_key] = collections.OrderedDict(
        name=detector_name,
        label=detector_label,
        score_threshold=score_threshold
    )
    # Add protein features
    record.features.append(SeqFeature(FeatureLocation(0, 2), type='CDS', qualifiers={'locus_tag': ['A']}))
    record.features.append(SeqFeature(FeatureLocation(2, 5), type='CDS', qualifiers={'locus_tag': ['B']}))
    record.features.append(SeqFeature(FeatureLocation(5, 8), type='CDS', qualifiers={'locus_tag': ['C']}))
    # Add pfam features
    score_column = util.format_bgc_score_column(detector_name)
    qualifiers = {score_column: [0.4], 'db_xref': ['PF00001'], 'locus_tag': ['A'], 'database': [PFAM_DB_VERSION]}
    record.features.append(SeqFeature(FeatureLocation(0, 2), type=util.PFAM_FEATURE, qualifiers=qualifiers))
    qualifiers = {score_column: [0.7], 'db_xref': ['PF00002'], 'locus_tag': ['B'], 'database': [PFAM_DB_VERSION]}
    record.features.append(SeqFeature(FeatureLocation(2, 5), type=util.PFAM_FEATURE, qualifiers=qualifiers))
    qualifiers = {score_column: [0.6], 'db_xref': ['PF00003'], 'locus_tag': ['C'], 'database': [PFAM_DB_VERSION]}
    record.features.append(SeqFeature(FeatureLocation(5, 8), type=util.PFAM_FEATURE, qualifiers=qualifiers))
    # Add BGC features
    qualifiers = { score_column: ['0.6'], 'detector': [detector_name], 'detector_label': [detector_label]}
    record.features.append(SeqFeature(FeatureLocation(0, 5), type='cluster', qualifiers=qualifiers))
    qualifiers = { 'detector': ['annotated'], 'detector_label': ['annotated']}
    record.features.append(SeqFeature(FeatureLocation(2, 8), type='cluster', qualifiers=qualifiers))
    return record


@pytest.mark.parametrize("writer_cls", WRITERS)
def test_unit_writer_full_record(tmpdir, writer_cls, processed_record):
    out_path = os.path.join(str(tmpdir), 'file.png')
    writer = writer_cls(out_path=out_path)
    writer.write(processed_record)
    writer.close()
    assert os.path.exists(out_path)


@pytest.mark.parametrize("writer_cls", WRITERS)
def test_unit_writer_no_record(tmpdir, writer_cls):
    out_path = os.path.join(str(tmpdir), 'file.png')
    writer = writer_cls(out_path=out_path)
    writer.close()


@pytest.mark.parametrize("writer_cls", WRITERS)
def test_unit_writer_no_features(tmpdir, writer_cls, processed_record):
    out_path = os.path.join(str(tmpdir), 'file.png')
    processed_record.features = []
    writer = writer_cls(out_path=out_path)
    writer.write(processed_record)
    writer.close()


@pytest.mark.parametrize("writer_cls", WRITERS)
def test_unit_writer_single_feature(tmpdir, writer_cls, processed_record):
    out_path = os.path.join(str(tmpdir), 'file.png')
    cds_features = util.get_protein_features(processed_record)
    pfam_features = util.get_pfam_features(processed_record)
    cluster_features = util.get_cluster_features(processed_record)
    processed_record.features = cds_features[:1] + pfam_features[:1] + cluster_features[:1]
    writer = writer_cls(out_path=out_path)
    writer.write(processed_record)
    writer.close()
