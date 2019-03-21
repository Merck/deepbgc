import logging

from deepbgc.main import run
import os
from Bio.SeqRecord import SeqRecord


def test_unit_pipeline_default(tmpdir, mocker):
    tmpdir = str(tmpdir)
    mocker.patch('os.mkdir')
    mocker.patch('deepbgc.command.pipeline.logging.FileHandler')
    mock_seqio = mocker.patch('deepbgc.command.pipeline.deepbgc.command.pipeline.SeqIO')

    record1 = SeqRecord('ABC')
    record2 = SeqRecord('DEF')
    mock_seqio.parse.return_value = [record1, record2]

    mock_annotator = mocker.patch('deepbgc.command.pipeline.DeepBGCAnnotator')
    mock_classifier = mocker.patch('deepbgc.command.pipeline.DeepBGCClassifier')
    mock_detector = mocker.patch('deepbgc.command.pipeline.DeepBGCDetector')

    writer_paths = [
        'deepbgc.command.pipeline.BGCRegionPlotWriter',
        'deepbgc.command.pipeline.ClusterTSVWriter',
        'deepbgc.command.pipeline.PfamScorePlotWriter',
        'deepbgc.command.pipeline.PfamTSVWriter',
        'deepbgc.command.pipeline.GenbankWriter',
        'deepbgc.command.pipeline.BGCGenbankWriter',
        'deepbgc.command.pipeline.ReadmeWriter'
        # Note: We are mocking classes imported in deepbgc.command.pipeline, not at their original location!
    ]
    writers = [mocker.patch(path) for path in writer_paths]

    report_dir = os.path.join(tmpdir, 'report')
    report_tmp_dir = os.path.join(report_dir, 'tmp')
    run([
        'pipeline', 
        '--output', report_dir, 
        '--detector', 'mydetector',
        '--label', 'mylabel',
        '--score', '0.1',
        '--merge-max-protein-gap', '8',
        '--merge-max-nucl-gap', '9',
        '--min-nucl', '10',
        '--min-proteins', '20',
        '--min-domains', '30',
        '--min-bio-domains', '40',
        '--classifier', 'myclassifier1',
        '--classifier', 'myclassifier2',
        '--classifier-score', '0.2',
        'mySequence.gbk'
    ])

    os.mkdir.assert_any_call(report_dir)
    os.mkdir.assert_any_call(report_tmp_dir)

    mock_annotator.assert_called_with(tmp_dir_path=report_tmp_dir)
    mock_classifier.assert_any_call(
        classifier='myclassifier1', 
        score_threshold=0.2
    )
    mock_classifier.assert_any_call(
        classifier='myclassifier2', 
        score_threshold=0.2
    )
    mock_detector.assert_called_with(
        detector='mydetector',
        label='mylabel',
        score_threshold=0.1,
        merge_max_protein_gap=8,
        merge_max_nucl_gap=9,
        min_nucl=10,
        min_proteins=20,
        min_domains=30,
        min_bio_domains=40
    )

    assert mock_annotator.return_value.run.call_count == 2     # Two records
    assert mock_detector.return_value.run.call_count == 2    # Two records
    assert mock_classifier.return_value.run.call_count == 4  # Two records for each of the two classifiers

    mock_annotator.return_value.print_summary.assert_called_once_with()
    mock_detector.return_value.print_summary.assert_called_once_with()
    assert mock_classifier.return_value.print_summary.call_count == 2  # For each of the two classifiers

    for writer in writers:
        assert writer.return_value.write.call_count == 2  # Two records
        writer.return_value.close.assert_called_once_with()

    # Remove logging handlers to avoid affecting other tests
    logger = logging.getLogger('')
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
