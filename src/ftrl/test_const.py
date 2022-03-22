import tensorflow as tf

if __name__ == '__main__':
    with tf.Session() as sess:
        w = tf.constant([[1, 2, 3, 4, 5]], name='w', dtype=tf.float32)
        b = tf.constant(1, name='b', dtype=tf.float32)
        x = tf.constant([1, 1, 1, 1, 1], name='x', dtype=tf.float32)
        y = tf.constant([1.0], name='y', dtype=tf.float32)
        pre = tf.reduce_sum(w * x, axis=1)
        pre = pre + b
        pre_pro = tf.sigmoid(pre)
        print("pre: ", pre)
        loss = tf.losses.log_loss(labels=y, predictions=pre_pro)
        loss2 = tf.losses.log_loss(labels=[1.0], predictions=[0.5])
        _loss, _pre_pro, _y, _loss2 = sess.run([loss, pre_pro, y, loss2])
        print(_y)
        print(_pre_pro)
        print(_loss2)
        print(_loss)

        tf.train.FtrlOptimizer()

