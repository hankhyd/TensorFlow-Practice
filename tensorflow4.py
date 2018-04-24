import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def layer(inputs, row, column, activation_function = None):
    Weights = tf.Variable(tf.random_normal([row, column]))
    bias = tf.Variable(tf.zeros([1, column]) + 0.1)
    func = tf.matmul(inputs, Weights) + bias
    if activation_function is None:
        outputs = func
    else:
        outputs = activation_function(func)
    return outputs

# creating some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# input training data x and y
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# building model
layer_1 = layer(xs, 1, 10, activation_function = tf.nn.relu)
outputs = layer(layer_1, 10, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - outputs), axis=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range (1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 20 == 0:
        #print(i, sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        outputs_value = sess.run(outputs, feed_dict={xs: x_data})
        lines = ax.plot(x_data, outputs_value, 'r-',lw=5)
        plt.pause(0.1)

