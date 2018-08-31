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

# Function #############################################################################################################

# Good Practice: define function to achieve modularity
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):  # Good Practice: use name scope
        n_inputs = int(X.get_shape()[1])
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=2 / np.sqrt(n_inputs + n_neurons))
        # Good Practice: use variable scope.
        # We recommend to use tf.get_variable since tf.Variable is a low level function
        # The name_scope is ignored by the variables created by tf.get_variable
        # Since tensorflow 1.4, you can use reuse=tf.AUTO_REUSE
        with tf.variable_scope(name, reuse=False):
            W = tf.get_variable("kernel", initializer=init, regularizer=tf.contrib.layers.l1_regularizer(scale))
            b = tf.get_variable("bias", initializer=tf.zeros((n_neurons, )),
                                regularizer=tf.contrib.layers.l1_regularizer(scale))
        # Z has shape (number of instances, number of features)
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

# Construction #########################################################################################################

tf.reset_default_graph()
# Good practice: always set the graph-level random seed of tensorflow
tf.set_random_seed(0)

global_step = tf.get_variable("global_step", shape=(), initializer=tf.constant_initializer(0.0), trainable=False)
# Don't forget to set "training" to True in the training session
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
    # The softmax_cross_entropy_with_logits is similar to it and the only difference is that it outputs one-hot vectors
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="base_loss")
    # After you add "regularizer" in tf.get_variable, tf has automatically created the nodes and put them in
    # tf.GraphKeys.REGULARIZATION_LOSSES
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
    # Save the handles of the tensors (or ops) that will be used for testing
    tf.add_to_collection("for_testing", i)

# Good practice: use global_variables_initializer()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "{}/run-{}/".format(path_save + "tf_logs", now)
loss_summary = tf.summary.scalar("Loss", loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Execution ############################################################################################################

mnist = input_data.read_data_sets(path_data)

# Run the following line only after you added all batch_normalization ops to tf.GraphKeys.UPDATE_OPS!
# Don't forget to run extra_update_ops in the training session!
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
best_acc_val = -np.inf
unimproved_steps = 0
stop_training = False

# Good practice: use "with" to start a session
with tf.Session() as sess:
    init.run()
    n_batches = mnist.train.num_examples // batch_size
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            # You may need to write your own function to fetch the batch
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # Good practice: run the ops in a single run (if allowed) to avoid repeated computing
            sess.run([training_op, extra_update_ops], feed_dict={X: X_batch, y: y_batch, training: True})
            if batch_index % 10 == 0:
                summary_str = loss_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            if batch_index % 50 == 0:
                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
                print("Epoch: ", epoch, " Number of updates: ", global_step.eval(), " Train accuracy: ", acc_train,
                      " Val accuracy: ", acc_val)
                unimproved_steps += 1
                if acc_val > best_acc_val:
                    unimproved_steps = 0
                    best_acc_val = acc_val
                    save_path = saver.save(sess, path_save + "epoch" + str(epoch) + "_n" + str(global_step.eval()) +
                                           "_acctrain" + str(acc_train) + "_accval" + str(acc_val) + ".ckpt")
                if unimproved_steps > threshold_earlystopping:
                    stop_training = True
                    break
        if stop_training:
            print("Stop at epoch ", epoch, " with a total of ", global_step.eval(), "updates")
            break
    save_path = saver.save(sess, path_save + "my_model_final.ckpt")

file_writer.close()

# Run TensorBoard if you want
os.system('~/anaconda/bin/python -m tensorboard.main --port 6004 --logdir ' + logdir)

# Prediction ###########################################################################################################

mnist = input_data.read_data_sets(path_data)

# tf.train.import_meta_graph will add the loaded stuff to the existing graph so you want to clear the existing graph
# by tf.reset_default_graph() before you do tf.train.import_meta_graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph(path_save + "my_model_final.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, path_save + "my_model_final.ckpt")
    # tf.add_to_collection and tf.get_collection are useful to save and then fetch a bunch of tensors and ops
    X, logits = tf.get_collection("for_testing")
    Z = logits.eval(feed_dict={X: mnist.test.images})
    y_pred = np.argmax(Z, axis=1)

np.sum(y_pred != mnist.test.labels) / y_pred.shape[0]
y_pred.shape = (-1, 1)
mnist.test.labels.shape = (-1, 1)
print(np.hstack((y_pred, mnist.test.labels)))

# Some Concepts ########################################################################################################

# A Tensor is a symbolic handle to one of the outputs of an operation
# It does not hold the values of that operation's output, but instead provides a means of computing those values
# in a TensorFlow tf.Session

# It has two primary purposes:
# 1. A Tensor can be passed as an input to another operation. This builds a dataflow connection between operations,
#    which enables TensorFlow to execute an entire Graph that represents a large, multi-step computation
# 2. After the graph has been launched in a session, the value of the tensor can be computed by passing it to
#    tf.Session.run, or by t.eval(), which is a shortcut for calling tf.get_default_session().run(t)

# X is a tensor and its name is "X:0"
print(X.name)
print(tf.get_default_graph().get_tensor_by_name("X:0"))
# X.op is an op and its name is "X"
print(X.op.name)
print(tf.get_default_graph().get_operation_by_name("X"))

# A variable is a tensor with additional capability and utility:
# 1. You can specify a variable as trainable (the default, actually), meaning that your optimizer will adjust it
#    in an effort to minimize your cost function
# 2. You can specify where the variable resides on a distributed system
# 3. You can easily save and restore variables and graphs
# https://stackoverflow.com/questions/44167134/whats-the-difference-between-tensor-and-variable-in-tensorflow
