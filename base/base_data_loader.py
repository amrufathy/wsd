class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_data(self):
        raise NotImplementedError
