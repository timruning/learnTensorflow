import tensorflow as tf

if __name__ == '__main__':
    w = tf.Variable(initial_value=tf.truncated_normal(shape=[8], stddev=0.1), name='weight')
    b = tf.Variable(initial_value=tf.zeros(1), name='bias')
    x = tf.placeholder(shape=[8], name='x', dtype=tf.float32)
    y = tf.placeholder(shape=[1], name='y', dtype=tf.int32)
    pre = tf.reduce_sum(w * x)
    print(pre)
    pre = pre + b
    print(pre)
