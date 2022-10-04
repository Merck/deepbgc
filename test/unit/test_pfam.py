import pytest
from pathlib import Path
from io import StringIO
import pandas as pd
from copy import deepcopy
from Bio import SearchIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
from deepbgc.pipeline.pfam import HmmscanPfamRecordAnnotator
import deepbgc.util as bgc_util
from tempfile import TemporaryDirectory


@pytest.fixture
def mock_find_exe(mocker):
    mock = mocker.patch('deepbgc.pipeline.pfam.find_executable')
    mock.return_value = 'hmmscan'
    mock.start()
    yield mock
    mock.stop()


@pytest.fixture
def mock_popen(mocker):

    class MockPopen:

        returncode = 0

        def __init__(self, *arg, **kwargs):
            pass

        @staticmethod
        def communicate():
            return '', ''

    mock = mocker.patch('deepbgc.pipeline.pfam.subprocess.Popen')
    mock.return_value = MockPopen
    mock.start()
    yield mock
    mock.stop()


@pytest.fixture
def mock_searchio(mocker):
    mock = mocker.patch('deepbgc.pipeline.pfam.SearchIO')
    mock.start()
    yield mock
    mock.stop()


@pytest.fixture
def mock_read_csv(mocker):
    mock = mocker.patch('deepbgc.pipeline.pfam.pd.read_csv')
    mock.return_value = pd.DataFrame().from_dict(
        {'pfam_id': [], 'clan_id': [], 'clan_name': [], 'pfam_name': [], 'description': []},
    )
    mock.start()
    yield mock
    mock.stop()


def test_basic_scenario(mock_find_exe, mock_popen, mock_searchio, mock_read_csv):
    """
    basic scenario where there are two CDS with different protein_id qualifiers
    """

    record = SeqRecord(
        Seq("atgc" * 50),
        id="contig_1",
        name="contig_1",
        description="mock record",
        annotations={"molecule_type": "DNA"},
        features=[
            SeqFeature(
                FeatureLocation(2, 11, strand=1),
                id='FOOBAR_1000',
                type="CDS",
                qualifiers={
                    'locus_tag': ['FOOBAR_1000'],
                    'protein_id': ['AAK73500.1'],
                }
            ),
            SeqFeature(
                FeatureLocation(15, 36, strand=1),
                id='FOOBAR_1005',
                type="CDS",
                qualifiers={
                    'locus_tag': ['FOOBAR_1005'],
                    'protein_id': ['AAK73498.1'],
                }
            ),
        ]
    )

    expected_record = deepcopy(record)
    expected_record.features += [
            SeqFeature(
                FeatureLocation(5, 11, strand=1),
                type="PFAM_domain",
                id='PF06970.10',
                qualifiers={
                    'db_xref': ['PF06970.10'],
                    'evalue': [1.2e-28],
                    'locus_tag': ['AAK73500.1'],
                    'database': ['31.0'],
                }
            ),
            SeqFeature(
                FeatureLocation(18, 36, strand=1),
                type="PFAM_domain",
                id='PF01076.18',
                qualifiers={
                    'db_xref': ['PF01076.18'],
                    'evalue': [6.9e-52],
                    'locus_tag': ['AAK73498.1'],
                    'database': ['31.0'],
                }
            ),
    ]

    expected_record.features = sorted(
        expected_record.features,
        key=lambda feature: (feature.location.start, -feature.location.end, bgc_util.FEATURE_ORDER.get(feature.type, 0))
    )

    data = \
"""#                                                                            --- full sequence --- -------------- this domain -------------   hmm coord   ali coord   env coord
# target name        accession   tlen query name           accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target
#------------------- ---------- ----- -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ----- ----- ---- ---------------------
RepA_N               PF06970.10    12 AAK73500.1         -            12   8.1e-29   99.3   0.0   1   1     7e-33   1.2e-28   98.8   0.0     2    12    2    12    2    12 0.99 Replication initiator protein A (RepA) N-terminus
Mob_Pre              PF01076.18   12 AAK73498.1       -            12   6.9e-52  176.0   1.7   1   2   4.9e-55   6.9e-52  176.0   1.7     2   12     2   12     2   12 0.98 Plasmid recombination enzyme
Mob_Pre              PF01076.18   12 AAK73498.1      -            12   6.9e-52  176.0   1.7   2   2         3   4.1e+03   -2.7   0.1    2    12   2   12   2   12 0.74 Plasmid recombination enzyme
"""
    mock_searchio.parse.return_value = SearchIO.parse(StringIO(data), 'hmmscan3-domtab')

    assert record.format('genbank') != expected_record.format('genbank')

    # record fixing happening before running pfam annotation
    bgc_util.fix_record_locus(record)
    bgc_util.fix_duplicate_cds(record)
    bgc_util.fix_dna_alphabet(record)

    with TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir).joinpath('hmmscan_out.pfam.domtbl.txt')
        out_file.touch()
        tmp_path_prefix = str(out_file.parent.joinpath(out_file.stem.split('.')[0]))
        annotator = HmmscanPfamRecordAnnotator(record, tmp_path_prefix, db_path='/mock/db', clans_path='/mock/clans')

        assert isinstance(annotator, HmmscanPfamRecordAnnotator)

        annotator.annotate()

        assert record.format('genbank') == expected_record.format('genbank')


def test_same_protein_id(mock_find_exe, mock_popen, mock_searchio, mock_read_csv):
    """
    scenario where there are two CDS with same protein_id qualifiers
    """
    record = SeqRecord(
        Seq("atgc" * 50),
        id="contig_1",
        name="contig_1",
        description="mock record",
        annotations={"molecule_type": "DNA"}, 
        features=[
            SeqFeature(
                FeatureLocation(2, 11, strand=1),
                id='FOOBAR_1000',
                type="CDS",
                qualifiers={
                    'locus_tag': ['FOOBAR_1000'],
                    'protein_id': ['AAK73500.1'],
                }
            ),
            SeqFeature(
                FeatureLocation(15, 36, strand=1),
                id='FOOBAR_1005',
                type="CDS",
                qualifiers={
                    'locus_tag': ['FOOBAR_1005'],
                    'protein_id': ['AAK73500.1'],
                }
            ),
        ]
    )

    expected_record = deepcopy(record)
    del expected_record.features[-1].qualifiers['protein_id']
    expected_record.features[-1].qualifiers['unique_protein_id'] = 'AAK73500.1_1'
    expected_record.features += [
            SeqFeature(
                FeatureLocation(5, 11, strand=1),
                type="PFAM_domain",
                id='PF06970.10',
                qualifiers={
                    'db_xref': ['PF06970.10'],
                    'evalue': [1.2e-28],
                    'locus_tag': ['AAK73500.1'],
                    'database': ['31.0'],
                }
            ),
            SeqFeature(
                FeatureLocation(18, 36, strand=1),
                type="PFAM_domain",
                id='PF01076.18',
                qualifiers={
                    'db_xref': ['PF01076.18'],
                    'evalue': [6.9e-52],
                    'locus_tag': ['AAK73500.1_1'],
                    'database': ['31.0'],
                }
            ),
    ]

    expected_record.features = sorted(
        expected_record.features,
        key=lambda feature: (feature.location.start, -feature.location.end, bgc_util.FEATURE_ORDER.get(feature.type, 0))
    )

    data = \
"""#                                                                            --- full sequence --- -------------- this domain -------------   hmm coord   ali coord   env coord
# target name        accession   tlen query name           accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target
#------------------- ---------- ----- -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ----- ----- ---- ---------------------
RepA_N               PF06970.10    12 AAK73500.1         -            12   8.1e-29   99.3   0.0   1   1     7e-33   1.2e-28   98.8   0.0     2    12    2    12    2    12 0.99 Replication initiator protein A (RepA) N-terminus
Mob_Pre              PF01076.18   12 AAK73500.1_1       -            12   6.9e-52  176.0   1.7   1   2   4.9e-55   6.9e-52  176.0   1.7     2   12     2   12     2   12 0.98 Plasmid recombination enzyme
Mob_Pre              PF01076.18   12 AAK73500.1_1      -            12   6.9e-52  176.0   1.7   2   2         3   4.1e+03   -2.7   0.1    2    12   2   12   2   12 0.74 Plasmid recombination enzyme
"""
    mock_searchio.parse.return_value = SearchIO.parse(StringIO(data), 'hmmscan3-domtab')

    assert record.format('genbank') != expected_record.format('genbank')

    # record fixing happening before running pfam annotation
    bgc_util.fix_record_locus(record)
    bgc_util.fix_duplicate_cds(record)
    bgc_util.fix_dna_alphabet(record)

    with TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir).joinpath('hmmscan_out.pfam.domtbl.txt')
        out_file.touch()
        tmp_path_prefix = str(out_file.parent.joinpath(out_file.stem.split('.')[0]))
        annotator = HmmscanPfamRecordAnnotator(record, tmp_path_prefix, db_path='/mock/db', clans_path='/mock/clans')

        assert isinstance(annotator, HmmscanPfamRecordAnnotator)

        annotator.annotate()

        assert record.format('genbank') == expected_record.format('genbank')

