from __future__ import (
    print_function,
    division,
    absolute_import,
)

from deepbgc import util
from deepbgc.command.base import BaseCommand
from deepbgc.data import DOWNLOADS


class DownloadCommand(BaseCommand):
    command = 'download'
    help = """
    Download trained models and other file dependencies to the DeepBGC downloads directory.
    
    By default, files are downloaded to: '{}'
    Set {} env variable to specify a different downloads directory."
    """.format(util.get_default_downloads_dir(), util.DEEPBGC_DOWNLOADS_DIR)

    def add_arguments(self, parser):
        pass

    def run(self):
        util.download_files(DOWNLOADS)
