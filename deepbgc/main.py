#!/usr/bin/env python
import argparse
from deepbgc.commands.detect import DetectCommand
from deepbgc.commands.pfam import PfamCommand
from deepbgc.commands.classify import ClassifyCommand
import sys

COMMANDS = [
    PfamCommand,
    DetectCommand,
    ClassifyCommand
]

def _fix_subparsers(subparsers):
    if sys.version_info[0] == 3:
        subparsers.required = True
        subparsers.dest = 'cmd'


class DeepBGCParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        self.exit(2, "{}\n".format(message))


def main(argv=None):
    parser = DeepBGCParser(prog='deepbgc',
                           description='DeepBGC - Biosynthetic Gene Cluster detection and classification.',
                           formatter_class=argparse.RawTextHelpFormatter)

    # Sub commands
    subparsers = parser.add_subparsers(
        title='Available Commands',
        metavar='COMMAND',
        dest='cmd',
        help='Use: deepbgc COMMAND --help for command-specific help.')

    _fix_subparsers(subparsers)

    for CommandClass in COMMANDS:
        CommandClass.add_subparser(subparsers)

    args = parser.parse_args(argv)

    # Initialize command object
    cmd = args.func(args)
    # Run command
    cmd.run()


if __name__ == '__main__':
    main(sys.argv[1:])
