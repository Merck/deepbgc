import logging

import numpy as np
from datetime import datetime
from Bio.SeqFeature import SeqFeature, FeatureLocation
from deepbgc.models.wrapper import SequenceModelWrapper
from deepbgc import util
from deepbgc.pipeline.step import PipelineStep
import collections
import six
import os

class DeepBGCDetector(PipelineStep):
    def __init__(self, detector, label=None, score_threshold=0.5, merge_max_protein_gap=0,
                 merge_max_nucl_gap=0, min_nucl=1, min_proteins=1, min_domains=1, min_bio_domains=0):
        self.score_threshold = score_threshold
        if detector is None or not isinstance(detector, six.string_types):
            raise ValueError('Expected detector name or path, got {}'.format(detector))
        if (os.path.exists(detector) or os.path.sep in detector) and not os.path.isdir(detector):
            model_path = detector
            # Set detector name to filename without suffix
            detector, _ = os.path.splitext(os.path.basename(detector))
        else:
            model_path = util.get_model_path(detector, 'detector')

        self.detector_name = detector
        self.detector_label = label or self.detector_name
        self.score_column = util.format_bgc_score_column(self.detector_name)
        self.merge_max_protein_gap = merge_max_protein_gap
        self.merge_max_nucl_gap = merge_max_nucl_gap
        self.min_nucl = min_nucl
        self.min_proteins = min_proteins
        self.min_domains = min_domains
        self.min_bio_domains = min_bio_domains
        self.model = SequenceModelWrapper.load(model_path)
        self.num_detected = 0

    def run(self, record):
        logging.info('Detecting BGCs using %s model in %s', self.detector_label, record.id)

        protein_features = util.get_protein_features(record)
        proteins_by_id = util.get_proteins_by_id(protein_features)
        pfam_features = util.get_pfam_features(record)

        if not len(pfam_features):
            logging.warning('Warning: No Pfam domains in record %s, skipping BGC detection', record.id)
            return

        # Filter out previous clusters detected with the same detector label
        num_prev_features = len(record.features)
        record.features = [f for f in record.features if
                           not(f.type == 'cluster' and f.qualifiers.get('detector_label') == [self.detector_label])]
        num_removed_features = num_prev_features - len(record.features)
        if num_removed_features:
            logging.warning('Warning: Removed %s previously clusters detected clusters with same label "%s". '
                  'Use --label DeepBGCMyLabel to preserve original clusters and add second set of clusters detected '
                  'with same model but different parameters.', num_removed_features, self.detector_label)

        # Create DataFrame with Pfam sequence
        pfam_sequence = util.create_pfam_dataframe_from_features(pfam_features, proteins_by_id)

        # Predict BGC score of each Pfam
        pfam_sequence[self.score_column] = self.model.predict(pfam_sequence)

        # Get average BGC score for each protein
        protein_scores = pfam_sequence.groupby('protein_id', sort=False)[self.score_column].mean()

        # Add score to all Pfam features
        for i, feature in enumerate(pfam_features):
            feature.qualifiers[self.score_column] = ['{:.5f}'.format(pfam_sequence[self.score_column].iloc[i])]

        # Add score to all protein features
        for protein_id, score in protein_scores.items():
            proteins_by_id[protein_id].qualifiers[self.score_column] = ['{:.5f}'.format(score)]

        clusters = []
        active_proteins = []
        gap_proteins = []

        # Create a list of cluster features by merging consecutive proteins with score satisfying given threshold
        # Neighboring clusters within given number of nucleotides/proteins are merged
        for protein in protein_features:
            if self.score_column not in protein.qualifiers:
                # TODO: Should proteins with no Pfam domains also be considered?
                # Current protein did not have any Pfam domains, therefore it has no BGC score, ignore it
                continue
            score = float(protein.qualifiers[self.score_column][0])
            # Inactive protein, add to gap
            if score < self.score_threshold:
                gap_proteins.append(protein)
                # We just changed from active to inactive, add current list of active proteins as a cluster
                if active_proteins:
                    clusters.append(active_proteins)
                    active_proteins = []
            # Active protein
            else:
                # If no cluster is open, check if we should merge with the previous cluster
                if not active_proteins and clusters:
                    prev_cluster_proteins = clusters[-1]
                    prev_end = prev_cluster_proteins[-1].location.end
                    if len(gap_proteins) <= self.merge_max_protein_gap or \
                            (protein.location.start - prev_end) <= self.merge_max_nucl_gap:
                        # Remove previous candidate and continue where it left off
                        clusters = clusters[:-1]
                        active_proteins = prev_cluster_proteins + gap_proteins

                # Add current protein to cluster
                active_proteins.append(protein)
                gap_proteins = []

        # Last protein was active, add list of active proteins as a cluster
        if active_proteins:
            clusters.append(active_proteins)

        # Add detected clusters as features
        record_num_detected = 0
        for cluster_proteins in clusters:
            start = cluster_proteins[0].location.start
            end = cluster_proteins[-1].location.end
            candidate_id = '{}_{}-{}.1'.format(record.id, int(start), int(end))

            if self.min_nucl > 1:
                nucl_length = end - start
                if nucl_length < self.min_nucl:
                    logging.debug('Skipping cluster %s with %s < %s nucleotides', candidate_id, nucl_length, self.min_nucl)
                    continue

            if self.min_proteins > 1:
                num_proteins = len(cluster_proteins)
                if num_proteins < self.min_proteins:
                    logging.debug('Skipping cluster %s with %s < %s proteins', candidate_id, num_proteins, self.min_proteins)
                    continue

            if self.min_domains > 1 or self.min_bio_domains > 0:
                pfam_ids = util.get_pfam_feature_ids(record)
                num_domains = len(pfam_features)
                if num_domains < self.min_domains:
                    logging.debug('Skipping cluster %s with %s < %s protein domains', candidate_id, num_domains, self.min_domains)
                    continue
                num_bio_domains = len(util.filter_biosynthetic_pfam_ids(pfam_ids))
                if num_bio_domains < self.min_bio_domains:
                    logging.debug('Skipping cluster %s with %s < %s known biosynthetic protein domains', candidate_id, num_bio_domains, self.min_bio_domains)
                    continue

            scores = [float(feature.qualifiers[self.score_column][0]) for feature in cluster_proteins]
            location = FeatureLocation(start, end)
            qualifiers = {
                self.score_column: ['{:.5f}'.format(np.mean(scores))],
                'detector': [self.detector_name],
                'detector_label': [self.detector_label],
                'detector_version': [self.model.version],
                'detector_version_timestamp': [self.model.timestamp],
                'product': ['{}_putative'.format(self.detector_name)],
                'bgc_candidate_id': [candidate_id]
            }
            record.features.append(SeqFeature(
                location=location,
                type="cluster",
                qualifiers=qualifiers
            ))
            record_num_detected += 1
            self.num_detected += 1

        # Sort all features by location
        util.sort_record_features(record)

        # Add detector metadata to the record as a structured comment
        if 'structured_comment' not in record.annotations:
            record.annotations['structured_comment'] = {}
        comment_key = util.format_detector_meta_key(self.detector_label)
        record.annotations['structured_comment'][comment_key] = collections.OrderedDict(
            name=self.detector_name,
            label=self.detector_label,
            version=self.model.version,
            version_timestamp=self.model.timestamp,
            detection_timestamp_utc=datetime.utcnow().isoformat(),
            score_threshold=self.score_threshold,
            merge_max_nucl_gap=self.merge_max_nucl_gap,
            merge_max_protein_gap=self.merge_max_protein_gap,
            min_proteins=self.min_proteins,
            min_domains=self.min_domains,
            min_bio_domains=self.min_bio_domains
        )
        logging.info('Detected %s BGCs using %s model in %s', record_num_detected, self.detector_label, record.id)

    def print_summary(self):
        logging.info('Detected %s total BGCs using %s model', self.num_detected, self.detector_label)
