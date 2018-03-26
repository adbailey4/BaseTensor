#!/usr/bin/env python3
"""Test Tensorflow utility functions"""
########################################################################
# File: utils_test.py
#  executable: utils_test.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################
import unittest
from basetensor.utils import optimistic_restore, check_for_nvidia_gpu


class BaseTensorTests(unittest.TestCase):
    """Test the functions in all of basetensor"""

    def test_optimistic_restore(self):
        """test_optimistic_restore"""
        # TODO still need to create a test model and make sure optimistic restore works
        pass

    def test_check_for_nvidia_gpu(self):
        """test check_for_nvidia_gpu"""
        self.assertFalse(check_for_nvidia_gpu(4))
        self.assertFalse(check_for_nvidia_gpu(4))


if __name__ == '__main__':
    unittest.main()
