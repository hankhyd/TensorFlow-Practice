import tensorflow as tf
import numpy as np

def layer(inputs, row, column, activation_function = None):
    Weights = tf.Variable(tf.random_normal([row, column]))
    bias = tf.Variable(tf.zeros([1, column]) + 0.1)
    func = tf.matmul(inputs, Weights) + bias
    if activation_function is None:
        outputs = func
    else:
        outputs = activation_function(func)
    return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
            # np.linspace(start, end, num=, endpoint=true, restep=False, dtype=None)
            # Will return evenly spaced numbers over a specified interval
            # clearly, the parameters of start and end defining the specified interval
            # num means the number of samples will be generate
            # for np.newaxis is tying to conver a 1-D vector into either row vectro or column vector
            # [:, np.newaxis] will convert a 1-D vector into a column vector
            # [np.newaxis, :] will convert a 1-D vector into a row vector
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
            # the reason I defined a noise here is becasue i am trying to create a dataset that is more close to reality
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

layer_1 = layer(xs, 1, 10, activation_function = tf.nn.relu)
outputs = layer(layer_1, 10, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - outputs), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range (1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        print(i, sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

