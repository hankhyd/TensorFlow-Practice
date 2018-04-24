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

# creating some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)# I'm creating fake data close to reality by adding some noise with it
y_data = np.square(x_data) - 0.5 + noise

# input training data x and y
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# Building Model
# There're only 3 layers in this simple model, the model structure is looking like 1-10-1,
# where the numbers mean how many neurons in each layer
layer_1 = layer(xs, 1, 10, activation_function = tf.nn.relu)
outputs = layer(layer_1, 10, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - outputs), axis=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range (1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        print(i, sess.run(loss, feed_dict={xs: x_data, ys: y_data})) # since i'm using placeholder for input data,
                                                                     # but the way placeholder input data is by using feed_dict
                                                                     # and its a dictionary, that's why i am using {}


# np.linspace(start, end, num=, endpoint=true, restep=False, dtype=None)
# will return evenly spaced numbers over a specified interval
# clearly, the parameters of start and end defining the specified interval
# num means the number of samples will be generate
# for np.newaxis is tying to conver a 1-D vector into either row vector or column vector
# [:, np.newaxis] will convert a 1-D vector into a column vector
# [np.newaxis, :] will convert a 1-D vector into a row vector


# tf.reduce_sum(input_tensor, axis, keepdims, name)
# is trying to first of all, compute sum of elements across dimensions of a tensor
# then it will reduce the dimension of input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of tensor is reduced by 1 for each entry in axis.
# if keepdims is true, the returned value will maintain the same dimension as the original input_tensor
