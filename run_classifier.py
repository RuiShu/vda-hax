import os
import sys
import argparse
from pprint import pprint
from codebase.args import args
from codebase.models.classifier import classifier
from codebase.train import train
from codebase import datasets
import tensorflow as tf

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr',     type=float, default=1e-3,      help="Learning rate")
parser.add_argument('--run',    type=int,   default=999,       help="Run index")
parser.add_argument('--logdir', type=str,   default='log',     help="Log directory")
args.set_args(parser.parse_args())
pprint(vars(args))

# Make model name
setup = [
    ('model={:s}',  'classifier'),
    ('src={:s}',  'mnist'),
    ('trg={:s}',  'svhn'),
    ('run={:04d}',   args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

M = classifier()
M.sess.run(tf.global_variables_initializer())
saver = None # tf.train.Saver()

src = datasets.Mnist(shape=(32, 32, 3))
trg = datasets.Svhn()

train(M, src, trg,
      saver=saver,
      model_name=model_name)
