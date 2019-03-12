
class OutputWriter(object):

    def __init__(self, out_path):
        self.out_path = out_path

    def write(self, record):
        raise NotImplementedError()

    def close(self):
        pass

    @classmethod
    def get_name(cls):
        raise NotImplementedError()

    @classmethod
    def get_description(cls):
        raise NotImplementedError()


class TSVWriter(OutputWriter):

    def __init__(self, out_path):
        super(TSVWriter, self).__init__(out_path)
        self.written = False

    def record_to_df(self, record):
        raise NotImplementedError()

    def write(self, record):
        df = self.record_to_df(record)
        if df.empty:
            return

        if self.written:
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True

        df.to_csv(self.out_path, mode=mode, header=header, index=False, sep='\t')
        self.written = True
