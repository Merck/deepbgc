#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import logging
import os
try:
    # If the deepbgc module is not accessible, we are running as a script so we need to add ".." to PATH.
    from deepbgc import __version__
except ImportError:
    import sys
    parent = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    print('DEV mode: Adding "{}" to PATH'.format(parent))
    sys.path.append(parent)
    from deepbgc import __version__

import argparse
import matplotlib
matplotlib.use('Agg')
from deepbgc.command.prepare import PrepareCommand
from deepbgc.command.download import DownloadCommand
from deepbgc.command.pipeline import PipelineCommand
from deepbgc.command.train import TrainCommand
from deepbgc.command.info import InfoCommand

import sys

def _fix_subparsers(subparsers):
    if sys.version_info[0] == 3:
        subparsers.required = True
        subparsers.dest = 'cmd'


class DeepBGCParser(argparse.ArgumentParser):

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            message = argparse._('unrecognized arguments: %s') % ' '.join(argv)
            if hasattr(args, 'cmd') and args.cmd:
                # Show help for specific command, catch exit and print message
                try:
                    super(DeepBGCParser, self).parse_args(args=[args.cmd, '--help'])
                except:
                    pass
            self.exit(2, "{}\n{}\n".format("="*80, message))
        return args

    def error(self, message):
        self.print_help()
        formatted_message = "{}\n{}\n".format("="*80, message) if message else None
        self.exit(2, formatted_message)


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.core.umath_tests is an internal NumPy module")
    warnings.filterwarnings("ignore", message="np.asscalar(a) is deprecated")
    warnings.filterwarnings("ignore", message="`wait_time` is not used anymore")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # Disable TensorFlow debug messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = DeepBGCParser(prog='deepbgc',
                           description='DeepBGC - Biosynthetic Gene Cluster detection and classification',
                           formatter_class=argparse.RawTextHelpFormatter)

    # Sub commands
    subparsers = parser.add_subparsers(
        title='Available Commands',
        metavar='COMMAND',
        dest='cmd',
        help='Use: deepbgc COMMAND --help for command-specific help')

    _fix_subparsers(subparsers)

    commands = [
        DownloadCommand(),
        PrepareCommand(),
        PipelineCommand(),
        TrainCommand(),
        InfoCommand()
    ]

    for command in commands:
        command.add_subparser(subparsers)

    args = parser.parse_args(argv)
    args_dict = {k: v for k, v in args.__dict__.items() if k not in ['cmd', 'func', 'debug']}

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)-7s %(asctime)s   %(message)s',
                        level=log_level, datefmt="%d/%m %H:%M:%S")

    logging.info("""
 _____                  ____    ____   ____ 
 |  _ \  ___  ___ ____ | __ )  / ___) / ___)
 | | \ \/ _ \/ _ \  _ \|  _ \ | |  _ | |    
 | |_/ /  __/  __/ |_) | |_) || |_| || |___ 
 |____/ \___|\___| ___/|____/  \____| \____)
=================|_|===== version """ + __version__ + " =====")

    args.func.run(**args_dict)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print(' Interrupted by the user')
        sys.exit(0)
    except Exception as e:
        message = e.args[0] if e.args else ''
        logging.exception(message)
        logging.error('='*80)
        logging.error('DeepBGC failed with %s%s', type(e).__name__, ': {}'.format(message) if message else '')
        if len(e.args) > 1:
            logging.error('='*80)
            for arg in e.args[1:]:
                logging.error(arg)
        logging.error('='*80)
        exit(1)
