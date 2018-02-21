"""
Dummy module where global parameters are set and stored
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Args(object):
    def set_args(self, args):
        self.__dict__ = args.__dict__

args = Args()
