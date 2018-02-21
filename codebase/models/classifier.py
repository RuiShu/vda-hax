import tensorflow as tf
import tensorbayes as tb
from extra_layers import basic_accuracy
from tensorbayes.layers import placeholder, constant
from tensorbayes.tfutils import get_getter
from codebase.args import args
from pprint import pprint
from designs import v1 as net
import numpy as np

softmax_xent = tf.nn.softmax_cross_entropy_with_logits

def classifier():
    T = tb.utils.TensorDict(dict(
    sess = tf.Session(config=tb.growth_config()),
        src_x = placeholder((None, 32, 32, 3)),
        src_y = placeholder((None, 10)),
        trg_x = placeholder((None, 32, 32, 3)),
        trg_y = placeholder((None, 10)),
        test_x = placeholder((None, 32, 32, 3)),
        test_y = placeholder((None, 10)),
        phase = placeholder((), tf.bool)
    ))

    # Supervised and conditional entropy minimization
    src_y = net.classifier(T.src_x, phase=True, internal_update=False)
    trg_y = net.classifier(T.trg_x, phase=True, internal_update=True, reuse=True)

    loss_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_y))

    # Evaluation (non-EMA)
    test_y = net.classifier(T.test_x, phase=False, scope='class', reuse=True)

    # Evaluation (EMA)
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    ema_op = ema.apply(tf.get_collection('trainable_variables', 'class/'))
    T.ema_y = net.classifier(T.test_x, phase=False, reuse=True, getter=get_getter(ema))

    src_acc = basic_accuracy(T.src_y, src_y)
    trg_acc = basic_accuracy(T.trg_y, trg_y)
    ema_acc = basic_accuracy(T.test_y, T.ema_y)
    fn_ema_acc = tb.function(T.sess, [T.test_x, T.test_y], ema_acc)

    # Optimizer
    loss_main = loss_class
    var_main = tf.get_collection('trainable_variables', 'class')
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)
    train_main = tf.group(train_main, ema_op)

    # Summarizations
    summary_main = [tf.summary.scalar('class/loss_class', loss_class),
                    tf.summary.scalar('acc/src_acc', src_acc),
                    tf.summary.scalar('acc/trg_acc', trg_acc)]
    summary_main = tf.summary.merge(summary_main)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('class'), loss_class]
    T.ops_main = [summary_main, train_main]
    T.fn_ema_acc = fn_ema_acc

    return T
