import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.examples.tutorials.mnist import input_data
save_path = "./save_model/mnist/"
# MNIST image dimension: (28, 28)
class Machine:
    def __init__(self, args):
        self.image_dimension = args.image_dimension
        self.n_class = args.n_class
        self.n_noise = args.n_noise
        self.network()

    def network(self):
        self.Y = tf.placeholder(tf.float32, [None, self.n_class])
        self.G_input = tf.placeholder(tf.float32, [None, self.n_noise])
        self.generated_image = self.generator(self.G_input, self.Y)
        self.D_g = self.discriminator(self.generated_image, self.Y)

        self.real_image = tf.placeholder(tf.float32, [None, self.image_dimension])
        self.D_r = self.discriminator(self.real_image, self.Y, True)

        self.loss_D = tf.reduce_mean(tf.log(tf.maximum(self.D_r,1e-10)) + tf.log(tf.maximum(1-self.D_g,1e-10)))
        self.loss_G = tf.reduce_mean(tf.log(tf.maximum(self.D_g,1e-10)))
        D_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        G_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        self.train_D = tf.train.AdamOptimizer(0.0001).minimize(-self.loss_D, var_list=D_weights)
        self.train_G = tf.train.AdamOptimizer(0.0001).minimize(-self.loss_G, var_list=G_weights)

    def generator(self, input, labels):
        with tf.variable_scope('generator'):
            inputs = tf.concat([input, labels], 1) #(?, 138)
            hidden = tf.contrib.layers.fully_connected(inputs, 256, activation_fn = tf.nn.relu)
            output = tf.contrib.layers.fully_connected(hidden, self.image_dimension, activation_fn = tf.nn.sigmoid)
        return output

    def discriminator(self, input, labels, reuse=None):
        with tf.variable_scope('discriminator', reuse = reuse):
            inputs = tf.concat([input, labels], 1)
            hidden = tf.contrib.layers.fully_connected(inputs, 256, activation_fn = tf.nn.relu)
            output = tf.contrib.layers.fully_connected(hidden, 1, activation_fn = tf.nn.sigmoid)
        return output

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

def train(args):
    GAN = Machine(args)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
    total_batch = int(mnist.train.num_examples / args.batch_size)

    for epoch in range(args.total_epoch):
        avg_loss_D = 0
        avg_loss_G = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            noise = get_noise(args.batch_size, args.n_noise)
            _, loss_val_D = sess.run([GAN.train_D, GAN.loss_D], feed_dict={GAN.real_image: batch_xs, GAN.G_input: noise, GAN.Y: batch_ys})
            _, loss_val_G = sess.run([GAN.train_G, GAN.loss_G], feed_dict={GAN.G_input: noise, GAN.Y: batch_ys})
            avg_loss_D += loss_val_D
            avg_loss_G += loss_val_G
        print('Epoch:', '%03d' % epoch, 'D loss: {:.4}'.format(avg_loss_D/total_batch), 'G loss: {:.4}'.format(avg_loss_G/total_batch))
        if epoch == 0 or (epoch + 1) % 50 == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if args.save == "True":
                print("Saving model...")
                saver.save(sess, save_path + str(epoch) + ".cptk")

def generate(args):
    import matplotlib.pyplot as plt
    GAN = Machine(args)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)
    model_name = args.n_model + ".cptk"
    saver.restore(sess, save_path +model_name)

    n_samples = 10
    noise = get_noise(n_samples, args.n_noise)
    targets = np.array([[0,1,2,3,4,5,6,7,8,9]]).reshape(-1)
    Y = np.eye(args.n_class)[targets]

    samples = sess.run(GAN.generated_image, feed_dict={GAN.G_input: noise, GAN.Y: Y})
    fig, ax = plt.subplots(1, n_samples, figsize=(n_samples, 1))
    for i in range(n_samples):
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(samples[i], (28, 28)))
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", default = "generate", type = str)
    parser.add_argument("--total_epoch", default = 200, type = int)
    parser.add_argument("--batch_size", default = 100, type = int)
    parser.add_argument("--learning_rate", default = 0.0002, type = float)
    parser.add_argument("--n_noise", default = 128, type = int)
    parser.add_argument("--n_class", default = 10, type = int)
    parser.add_argument("--image_dimension", default = 28*28, type = int)
    parser.add_argument("--n_model", default = "199", type = str, help="0, 49, 99")
    parser.add_argument("--save", default = "False", type = str)

    args = parser.parse_args()
    if args.todo == "train":
        train(args)
    elif args.todo == "generate":
        generate(args)
if __name__ == "__main__":
    main()
