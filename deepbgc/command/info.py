from __future__ import (
    print_function,
    absolute_import,
)

from deepbgc import util
from deepbgc.command.base import BaseCommand
import logging
from datetime import datetime
import os
from deepbgc.models.wrapper import SequenceModelWrapper


class InfoCommand(BaseCommand):
    command = 'info'
    help = """Show DeepBGC summary information about downloaded models and dependencies."""

    def add_arguments(self, parser):
        pass

    def print_model(self, name, model_path):
        logging.info("-"*80)
        logging.info('Model: %s', name)
        try:
            model = SequenceModelWrapper.load(model_path)
            logging.info('Type: %s', type(model.model).__name__)
            logging.info('Version: %s', model.version)
            logging.info('Timestamp: %s (%s)', model.timestamp, datetime.fromtimestamp(model.timestamp).isoformat())
        except Exception as e:
            logging.warning('Model not supported: %s', e)
            return False
        return True

    def run(self):
        ok = True
        custom_dir = os.environ.get(util.DEEPBGC_DOWNLOADS_DIR)
        if custom_dir:
            logging.info('Using custom downloads dir: %s', custom_dir)

        data_dir = util.get_downloads_dir(versioned=False)
        if not os.path.exists(data_dir):
            logging.warning('Data downloads directory does not exist yet: %s', data_dir)
            logging.warning('Run "deepbgc download" to download all dependencies or set %s env var', util.DEEPBGC_DOWNLOADS_DIR)
            ok = False
        else:
            logging.info('Available data files: %s', os.listdir(data_dir))

        versioned_dir = util.get_downloads_dir(versioned=True)
        if not os.path.exists(versioned_dir):
            logging.info('Downloads directory for current version does not exist yet: %s', versioned_dir)
            logging.info('Run "deepbgc download" to download current models')
            return

        detectors = util.get_available_models('detector')
        logging.info('='*80)
        logging.info('Available detectors: %s', detectors)

        if not detectors:
            logging.warning('Run "deepbgc download" to download current detector models')
            ok = False

        for name in detectors:
            model_path = util.get_model_path(name, 'detector')
            ok = self.print_model(name, model_path) and ok

        classifiers = util.get_available_models('classifier')
        logging.info('='*80)
        logging.info('Available classifiers: %s', classifiers)

        for name in classifiers:
            model_path = util.get_model_path(name, 'classifier')
            ok = self.print_model(name, model_path) and ok

        if not classifiers:
            logging.warning('Run "deepbgc download" to download current classifier models')
            ok = False

        logging.info('='*80)
        if ok:
            logging.info('All OK')
        else:
            logging.warning('Some warnings detected, check the output above')
