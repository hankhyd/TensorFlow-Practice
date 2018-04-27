import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def layer(inputs, row, column, n_layer, activation_function = None):
    layer_name = 'layer%s' % n_layer # string formatting
    with tf.name_scope(layer_name):
        with tf.name_scope('Weight'):
            Weights = tf.Variable(tf.random_normal([row, column]))
            tf.summary.histogram(layer_name + '-weight', Weights) #tf.summary.histogram()
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, column]) + 0.1)
            tf.summary.histogram(layer_name + '-bias', bias)
        with tf.name_scope('function'):
            func = tf.matmul(inputs, Weights) + bias
        if activation_function is None:
            outputs = func
        else:
            outputs = activation_function(func)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

layer_1 = layer(xs, 1, 10, n_layer=1, activation_function = tf.nn.relu)
outputs = layer(layer_1, 10, 1, n_layer=2, activation_function = None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - outputs), axis=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
merge = tf.summary.merge_all() # since we want to visual more than one feature for the processing, so we need to merge all of them together
writer = tf.summary.FileWriter('tensorboarda/', sess.graph)

sess.run(tf.global_variables_initializer())


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range (1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        outputs_value = sess.run(outputs, feed_dict={xs: x_data})
        lines = ax.plot(x_data, outputs_value, 'r-',lw=5)
        plt.pause(0.1)
        # to show the processing in histograms
        result = sess.run(merge, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)

# string formatting
# string formation functions kind same as '+', the string concatenation operator
# string formation works with integers by specifying %d instead of %s


# tf.summary.histogram(name, values, collection = None, family = None)
# it's adding a histogram summary makes it possible to visualize our data's distribution in Tensorboard
