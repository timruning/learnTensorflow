import numpy as np
import pandas as pd
import tensorflow as tf


def check(x: list):
    n = 0
    for v in x:
        if v % 2 == 1:
            n += 1
    if n % 2 == 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    print(10)
    x = tf.placeholder(shape=[None, 4], dtype=tf.int32)
    y = tf.placeholder(shape=[None], dtype=tf.float32)
    embedding = tf.Variable(tf.random_uniform((10, 8), -1, 1), name=f'embedding', trainable=False)
    layer = tf.Variable(tf.random_uniform([8, 1], -1, 1), name="layer", trainable=True)
    emb = tf.nn.embedding_lookup(embedding, x)
    cell = tf.keras.layers.SimpleRNNCell(8)
    emb2 = tf.keras.layers.LSTM(8)(emb)
    k = tf.matmul(emb2, layer)
    pred = tf.squeeze(tf.nn.sigmoid(tf.matmul(emb2, layer)))
    loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(y_true=y, y_pred=pred))
    op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000000):
            x1 = np.random.randint(10, size=[256, 4])
            y1 = [check(v) for v in x1]
            _loss, _emb2, _pred, _op = sess.run([loss, emb2, pred, op], feed_dict={
                x: x1, y: y1
            })
            if i % 5000 == 1:
                print(i, x1[0], y1[0], _pred[0])
                print(_emb2[0])
                print(_loss)
