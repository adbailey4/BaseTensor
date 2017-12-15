#!/usr/bin/env python3
"""Tensorflow utility functions"""
########################################################################
# File: utils.py
#  executable: utils.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################
import unittest
from basetensor.abstract import CreateTFDataset, CreateTFNetwork


class BaseTensorTests(unittest.TestCase):
    """Test the functions in all of basetensor"""

    @classmethod
    def setUpClass(cls):
        super(BaseTensorTests, cls).setUpClass()
        cls.dataset = PassDataset()

    def test_abstractfunctions(self):
        """test_CreateTFDataset"""
        with self.assertRaises(TypeError):
            DatasetFail()
            NetworkFail(self.dataset)

    def test_createnetwork(self):
        """test_createnetwork"""
        NetworkPass(self.dataset)
        with self.assertRaises(AssertionError):
            NetworkPass("somethingelse")


class PassDataset(CreateTFDataset):
    def __init__(self):
        super(PassDataset, self).__init__()

    def create_dataset(self):
        pass

    def create_iterator(self):
        pass

    def test(self):
        pass

    def load_training_data(self):
        pass

    def process_graph_output(self):
        pass

    def load_inference_data(self):
        pass


class DatasetFail(CreateTFDataset):
    def __init__(self):
        super(DatasetFail, self).__init__()

    def create_iterator(self):
        pass

    def test(self):
        pass

    def load_training_data(self):
        pass

    def process_graph_output(self):
        pass

    def load_inference_data(self):
        pass


class NetworkPass(CreateTFNetwork):
    def __init__(self, dataset):
        super(NetworkPass, self).__init__(dataset)

    def create_model(self):
        pass

    def create_train_ops(self):
        pass

    def create_inference_op(self):
        pass


class NetworkFail(CreateTFNetwork):
    def __init__(self, dataset):
        super(NetworkFail, self).__init__(dataset)

    def create_train_ops(self):
        pass

    def create_inference_op(self):
        pass


if __name__ == '__main__':
    unittest.main()
