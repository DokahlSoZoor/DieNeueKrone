import os
import sys
import time
import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
tf.set_random_seed(0)

import util

# hyperparameters
SEQLEN = 40
BATCHSIZE = 200
ALPHASIZE = util.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout

data_dir = "data/bbc/*/*.txt"
# data_dir = "data/11-0.txt"

def load_data(dir=data_dir):
    codetext, valitext = util.read_data_files(data_dir, validation=True)

    # display some stats on the data
    epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
    print('data stats: training_len={}, validation_len={}, epoch_size={}'.format(len(codetext), len(valitext), epoch_size))

    return codetext, valitext

#
# the model (see FAQ in README.md)
#
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

# using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
# dynamic_rnn infers SEQLEN from the size of the inputs Xo

# How to properly apply dropout in RNNs: see README.md
cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
# "naive dropout" implementation
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

H = tf.identity(H, name='H')  # just to give it a name

# Softmax layer implementation:
# Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]
# then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
# From the readout point of view, a value coming from a sequence time step or a minibatch item is the same thing.

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# stats for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
# you can compare training and validation curves visually in Tensorboard.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)


def train(codetext, valitext):
    DISPLAY_FREQ = 50
    _50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN

    # init
    istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 0

    # training loop
    print('=== TRAINING ===')
    for x, y_, epoch in util.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=50):

        # train on one minibatch
        feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
        _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

        # log training data for Tensorboard display a mini-batch of sequences (every 50 batches)
        if step % _50_BATCHES == 0:
            feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCHSIZE}  # no dropout for validation
            y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
            print('\n\nstep {} (epoch {}):'.format(step, epoch))
            print('  training:   loss={:.5f}, accuracy={:.5f}'.format(bl, acc))
            summary_writer.add_summary(smm, step)

        # run a validation step every 50 batches
        if step % _50_BATCHES == 0 and len(valitext) > 0:
            l_loss = []
            l_acc = []
            vali_state = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
            for vali_x, vali_y, _ in util.rnn_minibatch_sequencer(valitext, BATCHSIZE, SEQLEN, 1):
                feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_state, pkeep: 1.0,  # no dropout for validation
                             batchsize: BATCHSIZE}
                ls, acc, ostate = sess.run([batchloss, accuracy, H], feed_dict=feed_dict)
                l_loss.append(ls)
                l_acc.append(acc)
                vali_state = ostate
            # calculate average
            avg_summary = tf.Summary(value=[
                tf.Summary.Value(tag="batch_loss", simple_value=np.mean(l_loss)),
                tf.Summary.Value(tag="batch_accuracy", simple_value=np.mean(l_acc)),
            ])

            print('  validation: loss={:.5f}, accuracy={:.5f}'.format(ls, acc))
            # save validation data for Tensorboard
            validation_writer.add_summary(avg_summary, step)

        # display a short text generated with the current weights and biases (every 150 batches)
        if step // 3 % _50_BATCHES == 0:
            print('--- generated sample ---')
            ry = np.array([[util.encode_text('<t>')[0]]])
            rh = np.zeros([1, INTERNALSIZE * NLAYERS])
            for k in range(2000):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
                rc = util.sample_from_probabilities(ryo, topn=3 if epoch <= 1 else 2)
                print(util.decode_character(rc), end="")
                ry = np.array([[rc]])
            print('\n--- end of generated sample ---'.format(step))

        # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
            print("Saved file: " + saved_file)

        print('.', end='', flush=True)

        # loop state around
        istate = ostate
        step += BATCHSIZE * SEQLEN

def generate_sample(checkpoint, length=5000):
    output = ''
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(checkpoint+'.meta')
        new_saver.restore(sess, checkpoint)
        x = util.encode_text('<t>')[0]
        x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

        # initial values
        y = x
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

        i = 0
        while (i < length) or (length == -1):
            yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

            c = util.sample_from_probabilities(yo, topn=2)
            y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
            c = util.decode_character(c)
            i += 1
            gen_input = yield c
            if gen_input is not None:
                raise StopIteration

def generate_articles(checkpoint, amount=-1):
    sample = generate_sample(checkpoint, length=-1)
    i = 0
    # skip over all the initial stuff so we start with a decent state
    def skip_to_title():
        while not next(sample) == '<t>':
            continue
    skip_to_title()
    # print('skipped initial stuff')
    # read title and body
    while (i < amount) or (amount == -1):
        title = ''
        body = ''
        # get title
        c = next(sample)
        while (not c == '</t>') and len(title) < 80:
            title += c
            c = next(sample)
        # print(len(title), title)
        # skip if title was too long/invalid
        if len(title) >= 80:
            skip_to_title()
            continue
        # get body
        c = next(sample)
        while (not c == '<t>'):
            body += c
            c = next(sample)
        # print(len(body))
        # reject article if the body contains invalid characters
        if '</t>' in body:
            # print('article rejected')
            continue
        # yield article as a title, body pair
        i += 1
        yield title.strip('\n'), body.strip('\n')
    # stop sample iterator
    sample.send('stop')

if __name__ == '__main__':
    if 'train' in sys.argv:
        codetext, valitext = load_data()
        train(codetext, valitext)
    if 'article' in sys.argv:
        for title, body in generate_articles('checkpoints/rnn_train_1519647475-248000000', amount=-1):
            print(title, '\n\n', body, '\n\n')
    else:
        for c in generate_sample('checkpoints/rnn_train_1519647475-248000000', length=20000):
            print(c, end='')
