import tensorflow as tf
from tensorflow.contrib.losses import metric_learning

label = tf.constant([1, 0, 1], dtype=tf.int32)
embed = tf.constant([[1.0, 0., 1.], [3., 0., 3.], [1, 1, 1]], dtype=tf.float32)
loss = metric_learning.triplet_semihard_loss(labels=label, embeddings=embed,margin=100.0)

with tf.Session() as sess:
    loss_ = sess.run(loss)
    print(loss_)
