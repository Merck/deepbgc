from deepbgc.output.writer import OutputWriter
import sys
import os

class ReadmeWriter(OutputWriter):

    def __init__(self, out_path, root_path, writers):
        super(ReadmeWriter, self).__init__(out_path)
        self.root_path = root_path
        # Sort files in current directory first, then by filename
        self.writers = sorted(writers, key=lambda w: (not self._is_in_root_folder(w.out_path), w.out_path))

    def _is_in_root_folder(self, path):
        # get path relative to root folder
        rel_path = os.path.relpath(path, self.root_path)
        # file is in root folder if its path is the same as its basename (filename without folder prefix)
        return os.path.normpath(os.path.basename(rel_path)) == os.path.normpath(rel_path)

    @classmethod
    def get_description(cls):
        return None

    @classmethod
    def get_name(cls):
        return 'readme'

    def write(self, record):
        pass

    def close(self):
        readme = 'DeepBGC\n'
        readme += '=' * 80 + '\n'
        readme += ' '.join(sys.argv) + '\n'
        readme += '=' * 80 + '\n'
        readme += 'LOG.txt\tLog output of DeepBGC\n'
        for writer in self.writers:
            description = writer.get_description()
            if description:
                rel_path = os.path.relpath(writer.out_path, self.root_path)
                readme += rel_path + ' \t' + description + '\n'

        with open(self.out_path, 'w') as f:
            f.write(readme)
            f.write('\n')