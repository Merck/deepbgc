from abc import ABC, abstractmethod
import argparse


class BaseCommand(ABC):
    """
    Base abstract class for commands
    """
    command = ''
    help = ""

    def __init__(self, args):
        self.args = args

    @classmethod
    def add_subparser(cls, subparsers):
        parser = subparsers.add_parser(cls.command, description=cls.help, help=cls.help,
                                       formatter_class=argparse.RawTextHelpFormatter)
        parser.set_defaults(func=cls)
        return parser

    @abstractmethod
    def run(self):
        raise NotImplementedError()
