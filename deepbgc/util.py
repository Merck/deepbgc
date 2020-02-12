from __future__ import (
    print_function,
    division,
    absolute_import,
)

import logging
import pandas as pd
import collections
import os
import glob
import numpy as np
import re
import gzip

import six
from Bio import SeqIO
from Bio.Alphabet import SingleLetterAlphabet, generic_dna
from appdirs import user_data_dir
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib2 import urlopen
    from shutil import copyfileobj
    # We cannot use Python 2 urlretrieve since it does not work with SSL behind our custom proxy
    def urlretrieve(url, path):
        with open(path, 'wb') as outfile:
            copyfileobj(urlopen(url), outfile)

import hashlib
import subprocess
from distutils.spawn import find_executable
import shutil
from datetime import datetime
import gzip

SCORE_SUFFIX = '_score'
STRUCTURED_COMMENT_DETECTOR_PREFIX = 'deepbgc_detector_'
STRUCTURED_COMMENT_CLASSIFIER_PREFIX = 'deepbgc_classifier_'
DEEPBGC_DOWNLOADS_DIR = 'DEEPBGC_DOWNLOADS_DIR'
DEEPBGC_DATA_RELEASE_VERSION = 'DEEPBGC_DATA_RELEASE_VERSION'
PFAM_FEATURE = 'PFAM_domain'
EXTENSIONS_FASTA = ['.fa', '.fna', '.fasta']
EXTENSIONS_GENBANK = ['.gbk', '.gb', '.genbank']
EXTENSIONS_CSV = ['.csv', '.tsv']

def get_protein_features(record):
    return get_features_of_type(record, 'CDS')


def get_proteins_by_id(protein_features):
    return {protein_id: feature for feature in protein_features for protein_id in get_protein_ids(feature)}


def get_features_of_type(record, feature_type):
    return [feature for feature in record.features if feature.type == feature_type]


FEATURE_ORDER = {
    'cluster': -3,
    'gene': -2,
    'CDS': -1
}
def sort_record_features(record):
    record.features = sorted(record.features, key=lambda feature: (feature.location.start, -feature.location.end, FEATURE_ORDER.get(feature.type, 0)))


def get_pfam_features(record):
    pfams = get_features_of_type(record, PFAM_FEATURE)
    from deepbgc.data import PFAM_DB_VERSION
    filtered_pfams = [pfam for pfam in pfams if pfam.qualifiers.get('database') == [PFAM_DB_VERSION]]
    num_incompatible = len(pfams) - len(filtered_pfams)
    if num_incompatible:
        # TODO: enable users to run model with older incompatible Pfam DB versions?
        logging.warning('Ignoring {} incompatible Pfam features with database different than "{}"'.format(num_incompatible, PFAM_DB_VERSION))
    return filtered_pfams


def get_pfam_feature_ids(record):
    pfam_features = get_pfam_features(record)
    return [get_pfam_id(feature) for feature in pfam_features if get_pfam_id(feature)]


# Taken from antiSMASH ClusterFinder module
ANTISMASH_BIO_PFAMS = frozenset(["PF00109", "PF02801", "PF08659", "PF00378", "PF08541", "PF08545", "PF02803", "PF00108",
                          "PF02706", "PF03364", "PF08990", "PF00501", "PF00668", "PF08415", "PF00975", "PF03061",
                          "PF00432", "PF00494", "PF03936", "PF01397", "PF00432", "PF04275", "PF00348", "PF02401",
                          "PF04551", "PF00368", "PF00534", "PF00535", "PF02922", "PF01041", "PF00128", "PF00908",
                          "PF02719", "PF04321", "PF01943", "PF02806", "PF02350", "PF02397", "PF04932", "PF01075",
                          "PF00953", "PF01050", "PF03033", "PF01501", "PF05159", "PF04101", "PF02563", "PF08437",
                          "PF02585", "PF01721", "PF02052", "PF02674", "PF03515", "PF04369", "PF08109", "PF08129",
                          "PF09221", "PF09683", "PF10439", "PF11420", "PF11632", "PF11758", "PF12173", "PF04738",
                          "PF04737", "PF04604", "PF05147", "PF08109", "PF08129", "PF08130", "PF00155", "PF00202",
                          "PF00702", "PF06339", "PF04183", "PF10331", "PF03756", "PF00106", "PF01370", "PF00107",
                          "PF08240", "PF00441", "PF02770", "PF02771", "PF08028", "PF01408", "PF02894", "PF00984",
                          "PF00725", "PF03720", "PF03721", "PF07993", "PF02737", "PF00903", "PF00037", "PF04055",
                          "PF00171", "PF00067", "PF01266", "PF01118", "PF02668", "PF00248", "PF01494", "PF01593",
                          "PF03992", "PF00355", "PF01243", "PF00384", "PF01488", "PF00857", "PF04879", "PF08241",
                          "PF08242", "PF00698", "PF00483", "PF00561", "PF00583", "PF01636", "PF01039", "PF00288",
                          "PF00289", "PF02786", "PF01757", "PF02785", "PF02409", "PF01553", "PF02348", "PF00891",
                          "PF01596", "PF04820", "PF02522", "PF08484", "PF08421"])


def filter_biosynthetic_pfam_ids(pfam_ids):
    return ANTISMASH_BIO_PFAMS.intersection(pfam_ids)


def get_cluster_features(record, detector=None):
    features = get_features_of_type(record, 'cluster')
    features += get_features_of_type(record, 'cand_cluster')
    if detector:
        features = [f for f in features if f.qualifiers.get('detector', [None])[0] == detector]
    return features


def get_protein_ids(feature):
    ids = []
    if feature.qualifiers.get('unique_protein_id'):
        ids += feature.qualifiers['unique_protein_id']
    if feature.qualifiers.get('protein_id'):
        ids += feature.qualifiers['protein_id']
    if feature.qualifiers.get('locus_tag'):
        ids += feature.qualifiers['locus_tag']
    if feature.qualifiers.get('gene'):
        ids += feature.qualifiers['gene']
    if not ids:
        raise NotImplementedError('No recognized protein ID for feature: {}'.format(feature))
    return ids


def get_protein_id(feature):
    return get_protein_ids(feature)[0]


def format_detector_meta_key(detector_label):
    return STRUCTURED_COMMENT_DETECTOR_PREFIX + detector_label


def format_classifier_meta_key(classifier_name):
    return STRUCTURED_COMMENT_CLASSIFIER_PREFIX + classifier_name


def get_record_detector_meta(record):
    comments = record.annotations.get('structured_comment', {})
    return {value['label']: value for comment_key, value in comments.items() if comment_key.startswith(STRUCTURED_COMMENT_DETECTOR_PREFIX)}


def get_record_detector_names(record):
    return sorted(list(set([meta['name'] for meta in get_record_detector_meta(record).values()])))


def get_record_classifier_meta(record):
    comments = record.annotations.get('structured_comment', {})
    return {value['name']: value for comment_key, value in comments.items() if comment_key.startswith(STRUCTURED_COMMENT_CLASSIFIER_PREFIX)}


def get_record_classifier_names(record):
    return sorted(list(set([meta['name'] for meta in get_record_classifier_meta(record).values()])))


def extract_cluster_record(cluster_feature, record):
    """
    Extract cluster record, remove pfams that belong to proteins that are not fully present
    in the cluster (e.g. overlapping protein on complimentary strand)

    :param cluster_feature: cluster feature
    :param record: sequence record
    :return: extracted cluster region from the record
    """

    cluster_record = cluster_feature.extract(record)
    cluster_record.id = cluster_feature.qualifiers.get('bgc_candidate_id', ['unknown_cluster_id'])[0]
    cluster_record.description = ''
    cluster_record.annotations['source'] = record.annotations.get('source', '')
    cluster_record.annotations['organism'] = record.annotations.get('organism', '')
    
    proteins_by_id = get_proteins_by_id(get_protein_features(cluster_record))
    # Remove pfams with protein not fully inside cluster borders (therefore not present in cluster_record)
    cluster_record.features = [f for f in cluster_record.features
                               if not(f.type == PFAM_FEATURE and get_pfam_protein_id(f) not in proteins_by_id)]
    return cluster_record


def create_pfam_dataframe(record, add_scores=True, add_in_cluster=False):
    # FIXME: make sure columns are provided even with an empty DataFrame to avoid KeyErrors down the line
    proteins_by_id = get_proteins_by_id(get_protein_features(record))
    pfam_features = get_pfam_features(record)
    detector_names = get_record_detector_names(record) if add_scores else []
    cluster_locations = [f.location for f in get_cluster_features(record, 'annotated')] if add_in_cluster else []
    df = create_pfam_dataframe_from_features(pfam_features, proteins_by_id, detector_names, cluster_locations)
    df.insert(0, 'sequence_id', record.id)
    # No clusters were found, set in_cluster to 0
    if add_in_cluster and 'in_cluster' not in df.columns:
        df['in_cluster'] = 0
    return df


def create_pfam_dataframe_from_features(pfam_features, proteins_by_id, detector_names=[], cluster_locations=[]):
    return pd.DataFrame([create_pfam_dict(pfam, proteins_by_id, detector_names, cluster_locations) for pfam in pfam_features])


def get_pfam_protein_id(pfam_feature):
    return pfam_feature.qualifiers.get('locus_tag')[0]


def get_pfam_id(feature):
    if feature.qualifiers.get('db_xref'):
        return feature.qualifiers.get('db_xref')[0].split('.')[0]
    return None


def create_pfam_dict(pfam, proteins_by_id, detector_names, cluster_locations):
    locus_tag = get_pfam_protein_id(pfam)
    if locus_tag not in proteins_by_id:
        logging.warning('Available protein IDs: \n%s', proteins_by_id.keys())
        raise ValueError('Got pfam with missing protein ID "{}": {}'.format(locus_tag, pfam))
    protein = proteins_by_id.get(locus_tag)

    pfam_dict = collections.OrderedDict(
        protein_id=locus_tag,
        gene_start=protein.location.start,
        gene_end=protein.location.end,
        gene_strand=protein.strand,
        pfam_id=get_pfam_id(pfam)
    )

    if cluster_locations:
        pfam_dict['in_cluster'] = int(any(pfam.location.start in loc for loc in cluster_locations))

    # Add BGC score for each detector model
    for detector_name in detector_names:
        score_column = format_bgc_score_column(detector_name)
        pfam_dict[score_column] = float(pfam.qualifiers[score_column][0])

    return pfam_dict


def create_cluster_dataframe(record, add_pfams=True, add_proteins=True, add_classification=False):
    # FIXME: make sure columns are provided even with an empty DataFrame to avoid KeyErrors down the line
    cluster_features = get_cluster_features(record)
    classifier_names = get_record_classifier_names(record) if add_classification else []
    df = pd.DataFrame([create_cluster_dict(cluster, record, add_pfams=add_pfams, add_proteins=add_proteins, classifier_names=classifier_names) for cluster in cluster_features])
    if df.empty:
        return df
    # Move protein_ids and pfam_ids columns to the end
    if add_proteins:
        move_column_to_end(df, 'protein_ids')
    if add_pfams:
        move_column_to_end(df, 'bio_pfam_ids')
        move_column_to_end(df, 'pfam_ids')
    return df


def move_column_to_end(df, column):
    values = df[column]
    del df[column]
    df[column] = values


def create_cluster_dict(cluster, record, add_pfams=True, add_proteins=True, classifier_names=[]):
    start = int(cluster.location.start)
    end = int(cluster.location.end)
    cluster_record = extract_cluster_record(cluster, record)
    proteins = get_protein_features(cluster_record)
    pfam_ids = get_pfam_feature_ids(cluster_record)
    bio_pfam_ids = filter_biosynthetic_pfam_ids(pfam_ids)
    bgc_candidate_id = cluster.qualifiers.get('bgc_candidate_id', [None])[0]
    tool_name = cluster.qualifiers.get('tool', ['unspecified'])[0]
    detector_name = cluster.qualifiers.get('detector', [tool_name])[0]

    cluster_dict = collections.OrderedDict()
    cluster_dict['detector'] = detector_name
    cluster_dict['detector_version'] = cluster.qualifiers.get('detector_version', [None])[0]
    cluster_dict['detector_label'] = cluster.qualifiers.get('detector_label', [detector_name])[0]
    cluster_dict['bgc_candidate_id'] = bgc_candidate_id
    cluster_dict['nucl_start'] = int(cluster.location.start)
    cluster_dict['nucl_end'] = int(cluster.location.end)
    cluster_dict['nucl_length'] = end-start
    cluster_dict['num_proteins'] = len(proteins)
    cluster_dict['num_domains'] = len(pfam_ids)
    cluster_dict['num_bio_domains'] = len(bio_pfam_ids)

    score_column = format_bgc_score_column(detector_name)
    if cluster.qualifiers.get(score_column):
        cluster_dict[score_column] = cluster.qualifiers[score_column][0]

    if add_pfams:
        cluster_dict['pfam_ids'] = ';'.join(pfam_ids)
        cluster_dict['bio_pfam_ids'] = ';'.join(bio_pfam_ids)

    if add_proteins:
        protein_ids = [get_protein_id(protein) for protein in proteins]
        cluster_dict['protein_ids'] = ';'.join(protein_ids)

    for classifier_name in classifier_names:
        score_column = format_classification_score_column(classifier_name)
        class_column = format_classification_column(classifier_name)
        cluster_dict[class_column] = cluster.qualifiers.get(class_column, [None])[0]
        if cluster.qualifiers.get(score_column):
            scores = decode_class_score_string(cluster.qualifiers[score_column][0])
            cluster_dict.update(scores)

    return cluster_dict


def sanitize_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def format_bgc_score_column(model_name):
    return model_name + SCORE_SUFFIX


def get_classifier_prefix(model_name):
    return model_name.split('-')[0]


def format_classification_column(model_name):
    return get_classifier_prefix(model_name)


def format_classification_score_column(model_name):
    return format_classification_column(model_name) + SCORE_SUFFIX


def encode_class_score_string(scores):
    return ','.join(['{}={:.2f}'.format(cls, score) for cls, score in scores.items()])


def decode_class_score_string(string):
    pairs = [pair.split('=') for pair in string.split(',')]
    return pd.Series([p[1] for p in pairs], [p[0] for p in pairs])


def get_data_release_version():
    from deepbgc.data import DATA_RELEASE_VERSION
    return os.environ.get(DEEPBGC_DATA_RELEASE_VERSION, DATA_RELEASE_VERSION)


def get_default_downloads_dir():
    return user_data_dir("deepbgc", version="data")


def get_downloads_dir(versioned=True):
    downloads_dir = os.environ.get(DEEPBGC_DOWNLOADS_DIR)
    data_release_version = get_data_release_version()
    if not downloads_dir:
        downloads_dir = get_default_downloads_dir()
    version = data_release_version if versioned else 'common'
    return os.path.join(downloads_dir, version)


def get_downloaded_file_path(filename, dirname='', check_exists=True, versioned=True):
    path = os.path.join(get_downloads_dir(versioned=versioned), dirname, filename)
    if check_exists and not os.path.exists(path):
        dir_msg = 'Or set {} env var.'.format(DEEPBGC_DOWNLOADS_DIR)
        downloads_dir = os.environ.get(DEEPBGC_DOWNLOADS_DIR)
        if downloads_dir:
            dir_msg = 'For custom {}: "{}"'.format(DEEPBGC_DOWNLOADS_DIR, downloads_dir)
        raise ValueError('File "{}" is not downloaded yet'.format(filename),
                         'Use "deepbgc download" to download all dependencies',
                         dir_msg)
    return path


def format_model_name_from_path(detector_path):
    return os.path.splitext(os.path.basename(detector_path))[0]


def file_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_files(downloads):
    downloads_dir = get_downloads_dir(versioned=True)
    if not os.path.exists(downloads_dir):
        logging.info('Creating download directory: %s', os.path.abspath(downloads_dir))
        os.makedirs(downloads_dir)

    versioned_downloads_dir = get_downloads_dir(versioned=False)
    if not os.path.exists(versioned_downloads_dir):
        os.mkdir(versioned_downloads_dir)

    for download in downloads:
        url = download.get('url')
        checksum = download.get('checksum')
        filename = download.get('target')
        dirname = download.get('dir', '')
        gzipped = download.get('gzip', False)
        versioned = download.get('versioned', True)
        after_function = download.get('after')
        if dirname:
            dirpath = get_downloaded_file_path(filename='', dirname=dirname, check_exists=False,
                                                    versioned=versioned)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
        target_path = get_downloaded_file_path(filename, dirname=dirname, check_exists=False, versioned=versioned)
        logging.info('=' * 80)
        download_file(url=url, target_path=target_path, checksum=checksum, gzipped=gzipped)
        # Run after_function even if file was already downloaded
        # for example if pressing Pfam DB was interrupted last time, try again
        if after_function:
            after_function(target_path)

    logging.info('=' * 80)
    logging.info('All downloads finished')


def download_file(url, target_path, checksum, gzipped=False):
    logging.info('Checking: %s', target_path)

    if os.path.exists(target_path) and checksum == file_md5(target_path):
        logging.info('File already downloaded: %s', target_path)
        return False

    download_path = target_path
    if gzipped:
        download_path = target_path + '.gz'

    logging.info('Downloading: %s', url)
    urlretrieve(url, download_path)

    if gzipped:
        logging.info('Unzipping to: %s', target_path)
        with gzip.open(download_path, 'rt') as f_in:
            with open(target_path, 'wt') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(download_path)

    downloaded_checksum = file_md5(target_path)
    if downloaded_checksum != checksum:
        raise ValueError('File MD5 checksum "{}" does not match expected checksum "{}". Please try again.'.format(downloaded_checksum, checksum),
                         'While downloading from url: "{}"'.format(url))

    logging.info('Downloaded to: %s', target_path)
    return True

def get_downloaded_models_dir(model_type):
    assert model_type in ['detector', 'classifier']
    return os.path.join(get_downloads_dir(), model_type)

def get_available_models(model_type):
    available_paths = glob.glob(os.path.join(get_downloaded_models_dir(model_type), '*.pkl'))
    return [format_model_name_from_path(path) for path in available_paths]


def get_model_path(model_name, model_type):
    model_dir = get_downloaded_models_dir(model_type)
    path = os.path.join(model_dir, model_name+'.pkl')
    if not os.path.exists(path):
        available_names = get_available_models(model_type)
        if available_names:
            raise ValueError('Model "{}" does not exist in path {}'.format(model_name, path), 'Available models: {}'.format(', '.join(available_names)))
        raise ValueError('DeepBGC models directory does not exist yet: "{}". '.format(model_dir),
                         'Run "deepbgc download" to download trained models',
                         'Or set {} env var'.format(DEEPBGC_DOWNLOADS_DIR))
    return path


def run_hmmpress(path, force=False):
    if not force and os.path.exists(path+'.h3f'):
        logging.info('File already pressed, skipping: %s', path)
        return False

    if not find_executable('hmmpress'):
        raise RuntimeError("HMMER hmmpress needs to be installed and available on PATH "
                           "in order to prepare Pfam database.")

    logging.info('Pressing Pfam DB with HMMER hmmpress...')
    subprocess.call(['hmmpress', path])

    pressed_db_path = path + '.h3f'
    if not os.path.exists(pressed_db_path):
        raise RuntimeError("Unexpected error running HMMpress on Pfam DB")
    return True


def fix_record_locus(record):
    MAX_LEN = 19
    if record.name and len(record.name) > MAX_LEN:
        new_locus = record.name[:MAX_LEN-3] + '...'
        logging.warning('Trimming long record locus "{}" to "{}"'.format(record.name, new_locus))
        record.name = new_locus


def fix_duplicate_cds(record):
    # Don't allow multiple CDS for single gene
    protein_ids = set()
    for feature in get_protein_features(record):
        protein_id = get_protein_id(feature)

        # Find new unique proteid ID if needed
        new_protein_id = protein_id
        i = 1
        while new_protein_id in protein_ids:
            new_protein_id = '{}_{}'.format(protein_id, i)
            i += 1

        if protein_id != new_protein_id:
            logging.warning('Setting new unique_protein_id %s for CDS %s', new_protein_id, protein_id)
            feature.qualifiers['unique_protein_id'] = [new_protein_id]

        protein_ids.add(new_protein_id)


def fix_dna_alphabet(record):
    if type(record.seq.alphabet) == SingleLetterAlphabet:
        logging.warning('Updating record alphabet to generic_dna')
        record.seq.alphabet = generic_dna
    record.seq.alphabet = generic_dna


def read_compatible_csv(path):
    sep = '\t' if path.endswith('.tsv') else ','
    # Backwards-compatible with old "contig_id" column instead of "sequence_id"
    return pd.read_csv(path, sep=sep).rename(columns={'contig_id': 'sequence_id'})


def read_pfam_csv(path):
    df = read_compatible_csv(path)
    if 'pfam_id' not in df.columns:
        raise ValueError('File is not a Pfam CSV sequence, missing "pfam_id" column: {}'.format(path))
    if 'evalue' in df.columns:
        # Make sure we are not loading the old version of Pfam CSVs where e-value filtering was not yet performed
        raise ValueError('Old Pfam CSV format with the "evalue" column is not supported anymore. Create the Pfam CSV again using "deepbgc pfam" command or filter on e-value yourself and remove the column.')
    return df


def read_samples(paths, target_column=None):
    """
    Read multiple Pfam CSV file paths and return a list of DataFrames, one per each unique 'sequence_id' column value, along with its target column values.
    Will return tuple (samples_list, y_list).
    :param paths: List of Pfam TSV / GenBank file paths.
    :param target_column: Target column.
    :return: Tuple (samples_list, y_list), where samples_list is a list of DataFrames and y_list is a list of Series.
    """
    all_samples = []
    all_y = []
    if isinstance(paths, six.string_types):
        paths = [paths]

    for sample_path in paths or []:
        fmt = guess_format(sample_path, accept_csv=True)
        if fmt == 'csv':
            domains = read_pfam_csv(sample_path)
            if target_column and target_column not in domains.columns:
                raise ValueError('Sample does not contain target column "{}": {}'.format(target_column, sample_path))
        elif fmt == 'genbank':
            target_in_cluster = (target_column == 'in_cluster')
            if target_column and not target_in_cluster:
                raise NotImplementedError('Only "in_cluster" target is supported '
                                          'when using GenBank samples: {}'.format(sample_path))
            domains = pd.concat([create_pfam_dataframe(record, add_scores=False, add_in_cluster=target_in_cluster)
                                 for record in SeqIO.parse(sample_path, fmt)])
        else:
            raise NotImplementedError('Samples have to be provided in Pfam TSV or annotated GenBank format, got: {}'.format(sample_path))
        samples = [sample for sample_id, sample in domains.groupby('sequence_id')]
        logging.info('Loaded %s samples and %s domains from %s', len(samples), len(domains), sample_path)
        all_samples += samples
        if target_column:
            all_y += [sample[target_column] for sample in samples]
    if not target_column:
        return all_samples

    # Check that both positive and negative samples were provided
    unique_y = set([y for sample_y in all_y for y in sample_y.unique()])
    if len(unique_y) == 1:
        raise ValueError('Got target variable with only one value {} in: {}'.format(unique_y, paths),
                         'At least two values are required to train a model. ',
                         'Did you provide positive and negative samples?')

    return all_samples, all_y


def read_samples_with_classes(sample_paths, classes):
    if not sample_paths:
        return [], []
    samples = pd.concat(read_samples(sample_paths)).groupby('sequence_id')
    common_sample_ids = np.intersect1d(list(samples.groups.keys()), classes.index)
    if not len(common_sample_ids):
        raise ValueError('No overlap found between classes and samples. Classes should be indexed by sequence_id.')
    sample_classes = classes.loc[common_sample_ids]

    num_total = len(samples)
    num_missing = num_total - len(common_sample_ids)
    if num_missing:
        logging.warning('Warning: Removing %s/%s samples with missing class from %s', num_missing, num_total, sample_paths)
    samples_with_class = [samples.get_group(sample_id) for sample_id in common_sample_ids]

    logging.info('Loaded %s samples from %s', len(samples_with_class), sample_paths)
    return samples_with_class, sample_classes


def guess_format(file_path, accept_csv=False):
    _, ext = os.path.splitext(file_path.lower())
    if ext in EXTENSIONS_FASTA:
        return 'fasta'
    elif ext in EXTENSIONS_GENBANK:
        return 'genbank'
    elif accept_csv and ext in EXTENSIONS_CSV:
        return 'csv'

    extensions = EXTENSIONS_GENBANK + EXTENSIONS_FASTA
    if accept_csv:
        extensions += EXTENSIONS_CSV
    raise NotImplementedError("Sequence file type not recognized: '{}', ".format(file_path),
                              "Please provide a sequence "
                              "with an appropriate file extension "
                              "({}), ending with .gz if gzipped.".format(', '.join(extensions)))


def is_valid_hmmscan_output(domtbl_path):
    if not os.path.exists(domtbl_path):
        return False
    line = ''
    with open(domtbl_path, 'r') as infile:
        for line in infile:
            pass
    if '# [ok]' not in line:
        logging.warning('Not using existing but incomplete HMMER hmmscan output: %s', domtbl_path)
        return False
    return True


def print_elapsed_time(start_time):
    s = (datetime.now() - start_time).total_seconds()
    return '{:.0f}h{:.0f}m{:.0f}s'.format(s//3600, (s//60) % 60, s % 60)

def create_faux_record_from_proteins(proteins, id):
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    record = SeqRecord(seq=Seq(''), id=id)
    start = 0
    end = 0
    max_protein_id_len = 45
    for protein in proteins:
        nucl_length = len(protein.seq) * 3
        end += nucl_length
        feature = SeqFeature(
            location=FeatureLocation(start, end, strand=1),
            type="CDS",
            qualifiers={
                'protein_id': [protein.id[:max_protein_id_len]],
                'translation': [str(protein.seq)]
            }
        )
        start += nucl_length
        record.features.append(feature)
    return record

class SequenceParser(object):
    def __init__(self, file_path, protein=False):
        self.file_path = file_path
        self.fd = None
        self.fmt = None
        self.protein = protein

    def __enter__(self):
        if self.file_path.lower().endswith('.gz'):
            self.fmt = guess_format(self.file_path[:-3])
            self.fd = gzip.open(self.file_path, 'rt')
        else:
            self.fmt = guess_format(self.file_path)
            self.fd = open(self.file_path, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fd is not None:
            self.fd.close()

    def parse(self):
        records = SeqIO.parse(self.fd, self.fmt)
        if self.protein:
            if self.fmt != 'fasta':
                raise ValueError('Only FASTA format can be parsed in --protein mode, got: {}'.format(self.file_path))
            name, _ = os.path.splitext(os.path.basename(self.file_path))
            return [create_faux_record_from_proteins(records, id=name)]
        return records
