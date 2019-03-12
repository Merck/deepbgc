import argparse


class BaseCommand(object):
    """
    Base abstract class for commands
    """
    command = ''
    help = ""

    def add_subparser(self, subparsers):
        parser = subparsers.add_parser(self.command, description=self.help, help=self.help,
                                       formatter_class=argparse.RawTextHelpFormatter)
        parser.set_defaults(func=self)
        parser.add_argument('--debug', action='store_true')
        self.add_arguments(parser)

    def add_arguments(self, parser):
        pass

    def run(self, *args):
        raise NotImplementedError()
