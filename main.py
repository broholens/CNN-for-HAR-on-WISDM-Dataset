import warnings
from threading import Thread

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
N_TIME_STEPS = 90
N_FEATURES = 3
N_CLASSES = 6
N_HIDDEN_UNITS = 32
N_EPOCHS = 50
BATCH_SIZE = 64


class ModelGenerator:
    def __init__(self):
        self.columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
        self.filename = 'WISDM_ar_v1.1_raw.txt'

    def run(self):
        ts = [
            Thread(target=self.run_cnn),
            Thread(target=self.run_cnn_lstm),
            Thread(target=self.run_km_lstm)
        ]

        for t in ts:
            t.start()

        for t in ts:
            t.join()

    def _train_test_split(self):
        df = pd.read_csv(self.filename, names=self.columns, encoding="utf-8", error_bad_lines=False)
        df = df.dropna()
        step = 20
        segments = []
        labels = []
        for i in range(0, len(df) - N_TIME_STEPS, step):
            xs = df['x-axis'].values[i: i + N_TIME_STEPS]
            ys = df['y-axis'].values[i: i + N_TIME_STEPS]
            zs = df['z-axis'].values[i: i + N_TIME_STEPS]
            label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
            segments.append([xs, ys, zs])
            labels.append(label)

        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
        labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

        return train_test_split(reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

    def run_cnn(self):
        X_train, X_test, y_train, y_test = self._train_test_split()
        num_labels = 6
        LEARNING_RATE = 0.0025

        X = tf.placeholder(tf.float32, shape=[None, N_TIME_STEPS, N_FEATURES], name="input")
        Y = tf.placeholder(tf.float32, shape=[None, num_labels])
        conv1 = tf.layers.conv1d(inputs=X, filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=5, strides=2, padding='same')
        conv2 = tf.layers.conv1d(inputs=pool1, filters=32, kernel_size=5, strides=1, padding='same',
                                 activation=tf.nn.relu)
        flat = tf.layers.flatten(inputs=conv2)
        logits = tf.layers.dense(inputs=flat, units=6, activation=tf.nn.relu, name="y_")
        L2_LOSS = 0.0015

        l2 = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) + l2
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.train.Saver()

        history = dict(train_loss=[],
                       train_acc=[],
                       test_loss=[],
                       test_acc=[])

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        train_count = len(X_train)

        for i in range(1, N_EPOCHS + 1):
            for start, end in zip(range(0, train_count, BATCH_SIZE),
                                  range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
                sess.run(optimizer, feed_dict={X: X_train[start:end],
                                               Y: y_train[start:end]})

            _, acc_train, loss_train = sess.run([logits, accuracy, loss], feed_dict={
                X: X_train, Y: y_train})

            _, acc_test, loss_test = sess.run([logits, accuracy, loss], feed_dict={
                X: X_test, Y: y_test})

            history['train_loss'].append(loss_train)
            history['train_acc'].append(acc_train)
            history['test_loss'].append(loss_test)
            history['test_acc'].append(acc_test)

            print(f'[cnn]epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

        predictions, acc_final, loss_final = sess.run([logits, accuracy, loss], feed_dict={X: X_test, Y: y_test})
        print(f'[cnn]final results: accuracy: {acc_final} loss: {loss_final}')
        pickle.dump(predictions, open("cnn_predictions.p", "wb"))
        pickle.dump(history, open("cnn_history.p", "wb"))
        sess.close()

    def run_cnn_lstm(self):
        X_train, X_test, y_train, y_test = self._train_test_split()

        def create_cnn_lstm_model(inputs):
            W = {
                'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
                'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
            }
            biases = {
                'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
                'output': tf.Variable(tf.random_normal([N_CLASSES]))
            }

            X = tf.transpose(inputs, [1, 0, 2])
            X = tf.reshape(X, [-1, N_FEATURES])
            hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
            hidden = tf.split(hidden, N_TIME_STEPS, 0)

            # Stack 2 LSTM layers
            lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
            lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

            outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

            # Get output for the last time step
            lstm_last_output = outputs[-1]

            return tf.matmul(lstm_last_output, W['output']) + biases['output']

        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
        Y = tf.placeholder(tf.float32, [None, N_CLASSES])

        pred_Y = create_cnn_lstm_model(X)

        pred_softmax = tf.nn.softmax(pred_Y, name="y_")

        L2_LOSS = 0.0015

        l2 = L2_LOSS * \
             sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2

        LEARNING_RATE = 0.0025

        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

        history = dict(train_loss=[],
                       train_acc=[],
                       test_loss=[],
                       test_acc=[])

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        train_count = len(X_train)

        for i in range(1, N_EPOCHS + 1):
            for start, end in zip(range(0, train_count, BATCH_SIZE),
                                  range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
                sess.run(optimizer, feed_dict={X: X_train[start:end],
                                               Y: y_train[start:end]})

            _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
                X: X_train, Y: y_train})

            _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
                X: X_test, Y: y_test})

            history['train_loss'].append(loss_train)
            history['train_acc'].append(acc_train)
            history['test_loss'].append(loss_test)
            history['test_acc'].append(acc_test)

            print(f'[cnn-lstm]epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

        predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})
        print(f'[cnn-lstm]final results: accuracy: {acc_final} loss: {loss_final}')

        pickle.dump(predictions, open("cnn-lstm_predictions.p", "wb"))
        pickle.dump(history, open("cnn-lstm_history.p", "wb"))
        sess.close()

    def run_km_lstm(self):
        X_train, X_test, y_train, y_test = self._train_test_split()

        def create_km_lstm_model(input):
            conv1 = tf.layers.conv1d(inputs=input, filters=32, kernel_size=5, strides=1, padding='same',
                                     activation=tf.nn.relu)
            conv2 = tf.layers.conv1d(inputs=conv1, filters=32, kernel_size=5, strides=1, padding='same',
                                     activation=tf.nn.relu)
            n_ch = 32
            lstm_in = tf.transpose(conv2, [1, 0, 2])  # reshape into (seq_len, batch, channels)
            lstm_in = tf.reshape(lstm_in, [-1, n_ch])  # Now (seq_len*batch, n_channels)
            # To cells
            lstm_in = tf.layers.dense(lstm_in, N_HIDDEN_UNITS,
                                      activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?

            # Open up the tensor into a list of seq_len pieces
            lstm_in = tf.split(lstm_in, N_TIME_STEPS, 0)

            # Add LSTM layers
            lstm = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
            cell = tf.contrib.rnn.MultiRNNCell(lstm)
            outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32)

            # We only need the last output tensor to pass into a classifier
            logits = tf.layers.dense(outputs[-1], N_CLASSES)
            return logits

        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
        Y = tf.placeholder(tf.float32, [None, N_CLASSES])

        pred_Y = create_km_lstm_model(X)
        pred_softmax = tf.nn.softmax(pred_Y, name="y_")

        L2_LOSS = 0.0015

        l2 = L2_LOSS * \
             sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2

        LEARNING_RATE = 0.0025

        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

        history = dict(train_loss=[],
                       train_acc=[],
                       test_loss=[],
                       test_acc=[])

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        train_count = len(X_train)

        for i in range(1, N_EPOCHS + 1):
            for start, end in zip(range(0, train_count, BATCH_SIZE),
                                  range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
                sess.run(optimizer, feed_dict={X: X_train[start:end],
                                               Y: y_train[start:end]})

            _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
                X: X_train, Y: y_train})

            _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
                X: X_test, Y: y_test})

            history['train_loss'].append(loss_train)
            history['train_acc'].append(acc_train)
            history['test_loss'].append(loss_test)
            history['test_acc'].append(acc_test)

            print(f'[km-lstm]epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

        predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})
        print(f'[km-lstm]final results: accuracy: {acc_final} loss: {loss_final}')

        pickle.dump(predictions, open("km-lstm_predictions.p", "wb"))
        pickle.dump(history, open("km-lstm_history.p", "wb"))
        sess.close()


class ModelComparator:
    def __init__(self):
        self.cnn_history = pickle.load(open("cnn_history.p", "rb"))
        self.cnn_lstm_history = pickle.load(open("cnn-lstm_history.p", "rb"))
        self.km_lstm_history = pickle.load(open("km-lstm_history.p", "rb"))

    def draw_test_accuracy(self):
        plt.figure(figsize=(12, 8))

        plt.plot(np.array(self.cnn_history['test_acc']), "r-", label="CNN")
        plt.plot(np.array(self.cnn_lstm_history['test_acc']), "g-", label="CNN+LSTM")
        plt.plot(np.array(self.km_lstm_history['test_acc']), "b-", label="KM+LSTM")

        plt.title("Test accuracy")
        plt.legend(loc='lower right', shadow=True)
        plt.ylabel('Training Progress (Accuracy values)')
        plt.xlabel('Training Epoch')
        plt.ylim(0.7)

        plt.savefig('test_accuracy_compare.png', bbox_inches='tight')

    def draw_test_lose(self):
        plt.figure(figsize=(12, 8))

        plt.plot(np.array(self.cnn_history['test_loss']), "r-", label="CNN")
        plt.plot(np.array(self.cnn_lstm_history['test_loss']), "g-", label="CNN+LSTM")
        plt.plot(np.array(self.km_lstm_history['test_loss']), "b-", label="KM+LSTM")

        plt.title("Test lose")
        plt.legend(loc='higher right', shadow=True)
        plt.ylabel('Training Progress (Loss)')
        plt.xlabel('Training Epoch')
        plt.ylim(0)

        plt.savefig('test_lose_compare.png', bbox_inches='tight')

    def run(self):
        self.draw_test_lose()
        self.draw_test_accuracy()


if __name__ == '__main__':
    ModelGenerator().run()
    ModelComparator().run()

