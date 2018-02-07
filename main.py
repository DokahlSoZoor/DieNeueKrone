import collections
import shutil

import tensorflow as tf
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn import learn_runner
# import tensorflow.contrib.metrics as metrics
from tensorflow.contrib import rnn

tf.logging.set_verbosity(tf.logging.INFO)

# N_OUTPUTS = 1
N_INPUTS = 10

def lstm_model(features, targets, mode, params):
    LSTM_SIZE = N_INPUTS//3  # number of hidden layers in each of the LSTM cells

    # 1. dynamic_rnn needs 3D shape: [BATCH_SIZE, N_INPUTS, 1]
    x = tf.reshape(features, [-1, N_INPUTS, 1])

    # 2. configure the RNN
    lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    outputs = outputs[:, (N_INPUTS-1):, :]  # last cell only

    # 3. flatten lstm output and pass through a dense layer
    lstm_flat = tf.reshape(outputs, [-1, lstm_cell.output_size])
    h1 = tf.layers.dense(lstm_flat, N_INPUTS//2, activation=tf.nn.relu)
    predictions = tf.layers.dense(h1, 1, activation=None) # (?, 1)

    # 2. loss function, training/eval ops
    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(targets, predictions)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.01,
            optimizer="SGD"
        )
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(targets, predictions)
        }
    else:
        loss = None
        train_op = None
        eval_metric_ops = None

    # 3. Create predictions
    predictions_dict = {"predicted": predictions}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        #   export_outputs={'predictions': tf.estimator.export.PredictOutput(predictions_dict)}
    )

def get_train():
  return read_dataset('train.csv', mode=tf.contrib.learn.ModeKeys.TRAIN)

def get_valid():
  return read_dataset('valid.csv', mode=tf.contrib.learn.ModeKeys.EVAL)

from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
def experiment_fn(output_dir):
    # run experiment
    return tflearn.Experiment(
        tflearn.Estimator(model_fn=lstm_model, model_dir=output_dir),
        train_input_fn=get_train(),
        eval_input_fn=get_valid(),
        eval_metrics={
            'rmse': tflearn.MetricSpec(
                metric_fn=tf.metrics.streaming_root_mean_squared_error
            )
        }
    )

shutil.rmtree('outputdir', ignore_errors=True) # start fresh each time
learn_runner.run(experiment_fn, 'outputdir')

#
#
# def simple_rnn(features, targets, mode):
#   # 0. Reformat input shape to become a sequence
#   x = tf.split(features[TIMESERIES_COL], N_INPUTS, 1)
#   #print 'x={}'.format(x)
#
#   # 1. configure the RNN
#   lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
#   outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#   # slice to keep only the last cell of the RNN
#   outputs = outputs[-1]
#   #print 'last outputs={}'.format(outputs)
#
#   # output is result of linear activation of last layer of RNN
#   weight = tf.Variable(tf.random_normal([LSTM_SIZE, N_OUTPUTS]))
#   bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
#   predictions = tf.matmul(outputs, weight) + bias
#
#   # 2. loss function, training/eval ops
#   if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
#      loss = tf.losses.mean_squared_error(targets, predictions)
#      train_op = tf.contrib.layers.optimize_loss(
#          loss=loss,
#          global_step=tf.contrib.framework.get_global_step(),
#          learning_rate=0.01,
#          optimizer="SGD")
#      eval_metric_ops = {
#       "rmse": tf.metrics.root_mean_squared_error(targets, predictions)
#      }
#   else:
#      loss = None
#      train_op = None
#      eval_metric_ops = None
#
#   # 3. Create predictions
#   predictions_dict = {"predicted": predictions}
#
#   # 4. return ModelFnOps
#   return tflearn.ModelFnOps(
#       mode=mode,
#       predictions=predictions_dict,
#       loss=loss,
#       train_op=train_op,
#       eval_metric_ops=eval_metric_ops)
