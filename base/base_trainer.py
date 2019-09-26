from tensorflow.python.keras import Model


class BaseTrainer(object):
    model: Model

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, *data):
        raise NotImplementedError
