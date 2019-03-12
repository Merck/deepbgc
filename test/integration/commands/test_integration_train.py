import pandas as pd
import pytest

from deepbgc.main import main
from deepbgc.models.wrapper import SequenceModelWrapper
from test.test_util import get_test_file
import os


def test_integration_train_detect_fail_fasta():
    # Should fail due to unprocessed input sequence
    with pytest.raises(NotImplementedError):
        main(['train', '--model', get_test_file('clusterfinder_geneborder_test.json'), '--output', 'bar.pkl', get_test_file('BGC0000015.fa')])


@pytest.mark.parametrize("model", [
    "deepbgc_test.json",
    "clusterfinder_geneborder_test.json",
])
def test_integration_train_detect(model, tmpdir):
    tmpdir = str(tmpdir)
    out_path = os.path.join(tmpdir, 'model.pkl')
    main([
        'train',
        '--model', get_test_file(model),
        '--config', 'PFAM2VEC', get_test_file('pfam2vec.test.tsv'),
        '--output', out_path,
        get_test_file('BGC0000015.pfam.csv'),
        get_test_file('negative.pfam.csv')
    ])

    assert os.path.exists(out_path)

    model = SequenceModelWrapper.load(out_path)

    pos_domains = pd.read_csv(get_test_file('BGC0000015.pfam.csv'))
    neg_domains = pd.read_csv(get_test_file('negative.pfam.csv'))

    pos_prediction = model.predict(pos_domains)
    neg_prediction = model.predict(neg_domains)

    assert isinstance(pos_prediction, pd.Series)
    assert isinstance(neg_prediction, pd.Series)

    assert pos_prediction.index.equals(pos_domains.index)
    assert neg_prediction.index.equals(neg_domains.index)

    assert pos_prediction.mean() > 0.5
    assert neg_prediction.mean() < 0.5


def test_integration_train_classify(tmpdir):
    tmpdir = str(tmpdir)
    out_path = os.path.join(tmpdir, 'model.pkl')
    main([
        'train',
        '--model', get_test_file('random_forest_test.json'),
        '--classes', get_test_file('BGC0000015.classes.csv'),
        '--output', out_path,
        get_test_file('BGC0000015.pfam.csv')
    ])

    assert os.path.exists(out_path)

    model = SequenceModelWrapper.load(out_path)

    domains = pd.read_csv(get_test_file('BGC0000015.pfam.csv'))

    classes = model.predict([sample for _, sample in domains.groupby('sequence_id')])

    assert isinstance(classes, pd.DataFrame)
    assert list(classes.columns) == ['class1', 'class2', 'class3', 'class4']

    assert len(classes.index) == 2

    assert list(classes.iloc[0] > 0.5) == [True, False, True, False]
    assert list(classes.iloc[1] > 0.5) == [False, True, False, True]
