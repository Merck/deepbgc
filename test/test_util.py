import os


def get_test_file(path):
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(here, 'data', path)


def assert_sorted_features(record):
    prev_start = None
    prev_end = None
    for feature in record.features:
        if prev_start is not None:
            # Features starting sooner should be first
            assert feature.location.start >= prev_start
            if feature.location.start == prev_start:
                # For two features starting at same loci, the longer feature should be first
                assert feature.location.end <= prev_end
        prev_start = feature.location.start
        prev_end = feature.location.end
