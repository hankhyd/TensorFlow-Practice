import tensorflow as tf
import numpy as np

#create data
x = np.random.rand(200).astype(np.float32) #np.random.rand(shape) will return an array with given shape and populate it with random samples from a uniform distribution over [0, 1).
y_target = x * 0.8 + 0.3

#building the model
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))
y_output = tf.multiply(x, Weights) + bias
loss = tf.reduce_mean(tf.square(y_target - y_output))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(201):
    sess.run(train)
    if i % 10 == 0:
        print(i, sess.run(Weights), sess.run(bias))