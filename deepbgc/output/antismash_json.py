from deepbgc.output.writer import OutputWriter
from deepbgc import util
from deepbgc import __version__
import json
import collections

ANTISMASH_SUBREGION_LABEL_MAX_LENGTH = 20


class AntismashJSONWriter(OutputWriter):

    def __init__(self, out_path):
        super(AntismashJSONWriter, self).__init__(out_path)
        self.record_ids = []
        self.record_subregions = []
        self.record_protoclusters = []
        self.tool_meta = collections.OrderedDict()

    @classmethod
    def get_description(cls):
        return 'AntiSMASH JSON file for sideloading.'

    @classmethod
    def get_name(cls):
        return 'antismash-json'

    def write(self, record):
        cluster_features = util.get_cluster_features(record)
        classifier_names = util.get_record_classifier_names(record)
        subregions = []
        protoclusters = []
        for cluster in cluster_features:
            subregion = self._create_cluster_json(cluster, classifier_names=classifier_names)
            subregions.append(subregion)
            # TODO add protocluster?

        self.record_ids.append(record.id)
        self.record_subregions.append(subregions)
        self.record_protoclusters.append(protoclusters)
        for detector_label, meta in util.get_record_detector_meta(record).items():
            for k, v in meta.items():
                self.tool_meta['{}_{}'.format(detector_label, k)] = v

    def _get_cluster_classes_str(self, cluster, classifier_name):
        class_str_list = cluster.qualifiers.get(util.format_classification_column(classifier_name))
        return class_str_list[0] if class_str_list else 'no confident class'

    def _create_cluster_json(self, cluster, classifier_names):
        classes = {cls_name: self._get_cluster_classes_str(cluster, cls_name) for cls_name in classifier_names}
        tool_name = cluster.qualifiers.get('tool', ['unspecified'])[0]
        detector_name = cluster.qualifiers.get('detector', [tool_name])[0]
        score_column = util.format_bgc_score_column(detector_name)
        score = cluster.qualifiers.get(score_column)
        details = {
            'detector': detector_name,
        }
        details.update(classes)
        if score:
            details['score'] = score[0]
        return {
            'start': int(cluster.location.start),
            'end': int(cluster.location.end),
            'label': 'Putative BGC',
            'details': details
        }

    def _create_record_json(self, name, subregions, protoclusters):
        return {
            "name": name,
            "subregions": subregions,
            "protoclusters": protoclusters
        }

    def close(self):
        zipped_records = zip(self.record_ids, self.record_subregions, self.record_protoclusters)
        data = {
            "tool": {
                "name": "DeepBGC",
                "version": __version__,
                "description": "Putative BGCs predicted using DeepBGC",
                "configuration": self.tool_meta
            },
            "records": [self._create_record_json(name, sr, pc) for name, sr, pc in zipped_records]
        }
        with open(self.out_path, 'w') as f:
            json.dump(data, f, indent=2)
