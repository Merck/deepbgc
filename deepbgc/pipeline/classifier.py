import collections
from datetime import datetime
import logging
from deepbgc import util
import pandas as pd
from deepbgc.models.wrapper import SequenceModelWrapper
from deepbgc.pipeline.step import PipelineStep
import six
import os

class DeepBGCClassifier(PipelineStep):

    def __init__(self, classifier, score_threshold=0.5):
        if classifier is None or not isinstance(classifier, six.string_types):
            raise ValueError('Expected classifier name or path, got {}'.format(classifier))
        if (os.path.exists(classifier) or os.path.sep in classifier) and not os.path.isdir(classifier):
            classifier_path = classifier
            # Set classifier name to filename without suffix
            classifier, _ = os.path.splitext(os.path.basename(classifier))
        else:
            classifier_path = util.get_model_path(classifier, 'classifier')
        self.classifier_name = classifier
        self.score_threshold = score_threshold
        self.model = SequenceModelWrapper.load(classifier_path)
        self.total_class_counts = pd.Series()

    def run(self, record):
        cluster_features = util.get_cluster_features(record)
        if not len(cluster_features):
            return

        logging.info('Classifying %s BGCs using %s model in %s', len(cluster_features), self.classifier_name, record.id)

        # Create list of DataFrames with Pfam sequences (one for each cluster)
        cluster_pfam_sequences = []
        for feature in cluster_features:
            cluster_record = util.extract_cluster_record(feature, record)
            cluster_pfam_sequences.append(util.create_pfam_dataframe(cluster_record, add_scores=False))

        # Predict BGC score of each Pfam
        class_scores = self.model.predict(cluster_pfam_sequences)

        predicted_classes = []
        # Annotate classes to all cluster features
        for i, feature in enumerate(cluster_features):
            scores = class_scores.iloc[i]
            # Add predicted score for each class
            score_column = util.format_classification_score_column(self.classifier_name)
            feature.qualifiers[score_column] = [util.encode_class_score_string(scores)]
            # Add classes with score over given threshold
            new_classes = list(class_scores.columns[scores >= self.score_threshold])
            class_column = util.format_classification_column(self.classifier_name)
            all_classes = new_classes
            if feature.qualifiers.get(class_column):
                prev_classes = feature.qualifiers.get(class_column)[0].split('-')
                all_classes = sorted(list(set(all_classes + prev_classes)))
            if all_classes:
                feature.qualifiers[class_column] = ['-'.join(all_classes)]
            predicted_classes += new_classes or ['no confident class']

        # Add detector metadata to the record as a structured comment
        if 'structured_comment' not in record.annotations:
            record.annotations['structured_comment'] = {}

        comment_key = util.format_classifier_meta_key(self.classifier_name)
        record.annotations['structured_comment'][comment_key] = collections.OrderedDict(
            name=self.classifier_name,
            version=self.model.version,
            version_timestamp=self.model.timestamp,
            classification_timestamp_utc=datetime.utcnow().isoformat(),
            score_threshold=self.score_threshold
        )

        class_counts = pd.Series(predicted_classes).value_counts()
        self.total_class_counts = self.total_class_counts.add(class_counts, fill_value=0)

    def print_summary(self):
        # Print class counts
        sorted_counts = self.total_class_counts.sort_values(ascending=False).astype('int64')
        class_list = '\n'.join(' {}: {}'.format(cls, count) for cls, count in sorted_counts.items())
        logging.info('Number of BGCs with predicted %s: \n%s', self.classifier_name, class_list)





