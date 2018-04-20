import tensorflow as tf
import numpy as np

#creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.7 + 0.4

#tensorflow structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Bias = tf.Variable(tf.zeros([1]))

y = Weights*x_data + Bias

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


init = tf.initialize_all_variables()


#initialize
sess = tf.Session()
sess.run(init)


for step in range(500):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(Bias))


sate = tf.Variable()