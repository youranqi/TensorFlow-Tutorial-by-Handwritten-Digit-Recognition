import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import partial
import os
from tensorflow.examples.tutorials.mnist import input_data
np.set_printoptions(precision=2, suppress=True, threshold=10000000)

path_data = "/Path/To/Data/"
path_save = "/Path/To/Result/"
n_inputs = 28 * 28  # number of features
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
float_type = tf.float32
int_type = tf.int64
learning_rate = 0.01
n_epochs = 40
batch_size = 50
threshold_earlystopping = 10
scale = 0.001  # penalty parameter for l1 and l2 regularizations

# Function ############################################################################################################

def neuron_layer(X, n_neurons, name, activation=None):  # Good Practice: modularity
    with tf.name_scope(name):  # Good Practice: use name scope
        n_inputs = int(X.get_shape()[1])
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=2 / np.sqrt(n_inputs + n_neurons))
        # Good Practice: use variable scope. The name_scope is ignored by the variables created by tf.get_variable
        # We recommend to always use tf.get_variable. tf.Variable is low level and used when tf.get_variable doesn't appear
        # Since tensorflow 1.4, you can use reuse=tf.AUTO_REUSE
        with tf.variable_scope(name, reuse=False):
            W = tf.get_variable("kernel", initializer=init, regularizer=tf.contrib.layers.l1_regularizer(scale))
            b = tf.get_variable("bias", initializer=tf.zeros((n_neurons, )), regularizer=tf.contrib.layers.l1_regularizer(scale))
        Z = tf.matmul(X, W) + b  # Z has shape (number of instances, number of features)
        if activation is not None:
            return activation(Z)
        else:
            return Z

# Construction #########################################################################################################

tf.reset_default_graph()
tf.set_random_seed(0)  # Good practice: always set the graph-level random seed of tensorflow

global_step = tf.get_variable("global_step", shape=(), initializer=tf.constant_initializer(0.0), trainable=False)

# Don't forget to set "training" to True in the training session!
training = tf.placeholder_with_default(False, shape=(), name="training")
X = tf.placeholder(float_type, shape=(None, n_inputs), name="X")
y = tf.placeholder(int_type, shape=(None, ), name="y")

my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1")
    bn1 = my_batch_norm_layer(hidden1)
    bn1_act = tf.nn.relu(bn1)
    hidden2 = neuron_layer(bn1_act, n_hidden2, name="hidden2")
    bn2 = my_batch_norm_layer(hidden2)
    bn2_act = tf.nn.relu(bn2)
    logits_before_bn = neuron_layer(bn2_act, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    # The sparse_softmax_cross_entropy_with_logits outputs 0, 1, 2, ... , 9 etc. and handles the numerical issues
    # Another function softmax_cross_entropy_with_logits is similar to it and the only difference is that it outputs one-hot vectors
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="base_loss")
    # After you add "regularizer" in tf.get_variable, tf has automatically created the nodes and put them in tf.GraphKeys.REGULARIZATION_LOSSES
    # Don't forget to fetch these nodes and add them to the base_loss
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, float_type))

for i in (X, logits):
    tf.add_to_collection("for_testing", i)  # Save the handles of the tensors (or ops) that will be used for testing.

init = tf.global_variables_initializer()  # Good practice: use global_variables_initializer()
saver = tf.train.Saver()
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "{}/run-{}/".format(path_save + "tf_logs", now)
loss_summary = tf.summary.scalar("Loss", loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())




# A Tensor is a symbolic handle to one of the outputs of an operation.
# It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow tf.Session.
# It has two primary purposes:
# 1. A Tensor can be passed as an input to another operation. This builds a dataflow connection between operations, which enables TensorFlow to execute an entire Graph that represents a large, multi-step computation.
# 2. After the graph has been launched in a session, the value of the Tensor can be computed by passing it to tf.Session.run. t.eval() is a shortcut for calling tf.get_default_session().run(t).

# X is a tensor and its name is "X:0"
print(X.name)
print(tf.get_default_graph().get_tensor_by_name("X:0"))
# X.op is an op and its name is "X"
print(X.op.name)
print(tf.get_default_graph().get_operation_by_name("X"))

# A variable is a tensor with additional capability and utility.
# You can specify a variable as trainable (the default, actually), meaning that your optimizer will adjust it in an effort to minimize your cost function.
# You can specify where the variable resides on a distributed system.
# You can easily save and restore variables and graphs.
# https://stackoverflow.com/questions/44167134/whats-the-difference-between-tensor-and-variable-in-tensorflow