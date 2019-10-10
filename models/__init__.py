from .model_trainer import ModelTrainer

import tensorflow as tf
Dataset = tf.compat.v2.data.Dataset

class Model:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.input_dim = [224,224] # TODO: remove dummy value here
    
    def train(self, dataset:Dataset):
        return NotImplementedError

    def test(self, dataset:Dataset):
        return NotImplementedError


class ClassicModel(Model):
    def __init__(self, output_dim):
        super().__init__(output_dim)
        return None

    def train(self, dataset:Dataset):
        return NotImplementedError

    def test(self, dataset:Dataset):
        return NotImplementedError


class CCSAModel(Model):
    def __init__(self, output_dim):
        super().__init__(output_dim)
        return None

    def train(self, dataset:Dataset):
        return NotImplementedError

    def test(self, dataset:Dataset):
        return NotImplementedError


class DSNEModel(Model):
    def __init__(self, output_dim):
        super().__init__(output_dim)
        return None

    def train(self, dataset:Dataset):
        return NotImplementedError

    def test(self, dataset:Dataset):
        return NotImplementedError

