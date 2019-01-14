from deepbgc.commands.base import BaseCommand
from deepbgc.converter import SequenceToPfamCSVConverter


class PfamCommand(BaseCommand):
    command = 'pfam'
    help = """Convert genomic BGCs sequence into a pfam domain CSV file by detecting proteins and pfam domains.
    
Examples:
    
  # Detect proteins and pfam domains in a FASTA sequence and save the result as csv file 
  deepbgc pfam --pfam Pfam-A.hmm inputSequence.fa outputPfamSequence.csv
  """

    def __init__(self, args):
        super().__init__(args)
        self.input_path = args.input
        self.output_path = args.output
        self.converter = SequenceToPfamCSVConverter(db_path=args.pfam)

    @classmethod
    def add_subparser(cls, subparsers):
        parser = super().add_subparser(subparsers)

        # parser.add_argument('--mode', default='auto', choices=['auto', 'nucl', 'prot', 'pfam'],
        #                     help="Input modes: \n"
        #                          "--mode auto: Automatic based on file extension.\n"
        #                          "--mode nucl: Nucleotide sequence without annotated genes. Will detect genes and pfam domains. \n"
        #                          "--mode prot: Protein sequence. Will detect pfam domains.)")
        parser.add_argument('-p', '--pfam', required=True, help="Pfam DB (Pfam-A.hmm) file path.")
        parser.add_argument(dest='input', help="Input sequence file path.")
        parser.add_argument(dest='output', help="Output pfam CSV file path.")

    def run(self):
        self.converter.convert(self.input_path, self.output_path)
        print()
        print('Saved Pfam CSV to: {}'.format(self.output_path))
