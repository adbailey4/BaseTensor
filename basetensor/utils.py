#!/usr/bin/env python3
"""Tensorflow utility functions"""
########################################################################
# File: utils.py
#  executable: utils.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################
import tensorflow as tf
import re
import subprocess
import sys


def optimistic_restore(session, save_file):
    """ Implementation from: https://github.com/tensorflow/tensorflow/issues/312 """
    print('Restoring model from:', save_file)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def test_for_nvidia_gpu(num_gpu):
    assert type(num_gpu) is int, "num_gpu option must be integer"
    if num_gpu == 0:
        return False
    else:
        try:
            utilization = re.findall(r"Utilization.*?Gpu.*?(\d+).*?Memory.*?(\d+)",
                                     subprocess.check_output(["nvidia-smi", "-q"]).decode('utf-8'),
                                     flags=re.MULTILINE | re.DOTALL)
            indices = [i for i, x in enumerate(utilization) if x == ('0', '0')]
            assert len(indices) >= num_gpu, "Only {0} GPU's are available, change num_gpu parameter to {0}".format(
                len(indices))
            return indices[:num_gpu]
        except OSError:
            print("No GPU's found.", file=sys.stderr)
            return False


if __name__ == "__main__":
    print("This is a library file. Nothing to execute")
    raise SystemExit
