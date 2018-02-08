import collections
import random

import tensorflow as tf
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn import learn_runner
# import tensorflow.contrib.metrics as metrics
from tensorflow.contrib import rnn

tf.logging.set_verbosity(tf.logging.INFO)

# N_OUTPUTS = 1
N_INPUTS = 10

def lstm_model(features, labels, mode, params):
    LSTM_SIZE = N_INPUTS//3  # number of hidden layers in each of the LSTM cells

    # 1. dynamic_rnn needs 3D shape: [BATCH_SIZE, N_INPUTS, 1]
    x = tf.reshape(features, [-1, N_INPUTS, 1])

    # 2. configure the RNN
    lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    outputs = outputs[:, (N_INPUTS-1):, :]  # last cell only

    # 3. flatten lstm output and pass through a dense layer
    lstm_flat = tf.reshape(outputs, [-1, lstm_cell.output_size])
    h1 = tf.layers.dense(lstm_flat, N_INPUTS//2, activation=tf.nn.tanh)
    predictions = tf.layers.dense(h1, 1, activation=None) # (?, 1)

    # 2. loss function, training/eval ops
    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(labels, predictions)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=0.01,
            optimizer="SGD"
        )
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(labels, predictions)
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

def line(steps, gradient):
    if gradient > 0:
        return [[(i/steps)*gradient] for i in range(steps)]
    else:
        return [[1+(i/steps)*gradient] for i in range(steps)]

def read_dataset(filename=None):
    # generate data
    all_data = [line(N_INPUTS+1, random.uniform(-1,1)) for _ in range(2000)]

    # split data into training and evaluation (80%, 20%)
    train_data = all_data[:int(len(all_data)*0.8)]
    eval_data  = all_data[int(len(all_data)*0.8):]

    # transpose
    train_data = list(map(list, zip(*train_data)))
    eval_data = list(map(list, zip(*eval_data)))

    # make into tensors
    train_x = tf.concat(train_data[:-1], axis=1)
    # train_y = tf.concat(train_data[-1], axis=0)
    train_y = train_data[-1]
    eval_x  = tf.concat(eval_data[:-1], axis=1)
    # eval_y  = tf.concat(eval_data[-1], axis=0)
    eval_y = eval_data[-1]


    # and return
    return (train_x, train_y), (eval_x, eval_y)


model = tf.estimator.Estimator(
    model_fn=lstm_model,
    # params={
    #     'feature_columns': my_feature_columns,
    #     # Two hidden layers of 10 nodes each.
    #     'hidden_units': [10, 10],
    #     # The model must choose between 3 classes.
    #     'n_classes': 3,
    # }
)

def input_fn(features, labels, batch_size, repeat=-1):
    def _input_fn():
        """An input function for training"""

        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat(repeat).batch(batch_size)

        # Build the Iterator, and return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn


(train_x, train_y), (eval_x, eval_y) = read_dataset()

model.train(
    input_fn=input_fn(train_x, train_y, 20),
    steps=2000
)

pred = model.predict(
    input_fn=input_fn(eval_x, eval_y, 20, repeat=1),
)

result = model.evaluate(
    input_fn=input_fn(eval_x, eval_y, 20, repeat=1),
)

for metric, values in result.items():
    print(metric + ':', result)
