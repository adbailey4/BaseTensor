#!/usr/bin/env python3
"""Test Tensorflow utility functions"""
########################################################################
# File: utils_test.py
#  executable: utils_test.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################
import unittest
from basetensor.utils import optimistic_restore, test_for_nvidia_gpu


class BaseTensorTests(unittest.TestCase):
    """Test the functions in all of basetensor"""

    def test_optimistic_restore(self):
        """test_optimistic_restore"""
        pass

    def test_test_for_nvidia_gpu(self):
        """test_for_nvidia_gpu"""
        pass



if __name__ == '__main__':
    unittest.main()
