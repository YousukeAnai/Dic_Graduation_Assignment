import numpy as np
# import os
import tensorflow as tf
# from PIL import Image
# import utility as Utility
# import argparse


class BiGAN():
    def __init__(self, noise_unit_num, img_channel, seed, base_channel, keep_prob):
        self.NOISE_UNIT_NUM = noise_unit_num # 200
        self.IMG_CHANNEL = img_channel # 1
        self.SEED = seed
        np.random.seed(seed=self.SEED)
        self.BASE_CHANNEL = base_channel # 32
        self.KEEP_PROB = keep_prob

                                
    def leaky_relu(self, x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


    def gaussian_noise(self, input, std): #used at discriminator
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32, seed=self.SEED)
        return input + noise

    def conv2d(self, input, in_channel, out_channel, k_size, stride, seed):
        w = tf.get_variable('w', [k_size, k_size, in_channel, out_channel],
                              initializer=tf.random_normal_initializer
                              (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding="SAME", name='conv') + b
        return conv

    def conv2d_transpose(self, input, in_channel, out_channel, k_size, stride, seed):
        w = tf.get_variable('w', [k_size, k_size, out_channel, in_channel],
                              initializer=tf.random_normal_initializer
                              (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        out_shape = tf.stack(
                        [tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, tf.constant(out_channel)])
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=out_shape, strides=[1, stride, stride, 1],
                                                         padding="SAME") + b
        return deconv

    def batch_norm(self, input):
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        scale = tf.get_variable('scale', [n_out], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0))
        batch_mean, batch_var = tf.nn.moments(input, [0])
        bn = tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.0001, name='batch_norm')
        return bn

    def fully_connect(self, input, in_num, out_num, seed):
        w = tf.get_variable('w', [in_num, out_num], initializer=tf.random_normal_initializer
        (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_num], initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input, w, name='fc') + b
        return fc

    def encoder(self, x, reuse=False, is_training=False): #x is expected [n, 100, 100, 1]
        with tf.variable_scope('encoder', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer1 conv nx100x100x1 -> nx50x50x32
                conv1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL, 3, 2, self.SEED)

            with tf.variable_scope("layer2"):  # layer2 conv nx50x50x32 -> nx25x25x64
                conv2 = self.conv2d(conv1, self.BASE_CHANNEL, self.BASE_CHANNEL*2, 3, 2, self.SEED)
                bn2 = self.batch_norm(conv2)
                lr2 = self.leaky_relu(bn2, alpha=0.1)

            with tf.variable_scope("layer3"):  # layer3 conv nx25x25x64 -> nx13x13x128
                conv3 = self.conv2d(lr2, self.BASE_CHANNEL*2, self.BASE_CHANNEL*4, 3, 2, self.SEED)
                bn3 = self.batch_norm(conv3)
                lr3 = self.leaky_relu(bn3, alpha=0.1)

            with tf.variable_scope("layer4"):  # layer4 fc nx13x13x128 -> nx200
                shape = tf.shape(lr3)
                print(shape[1])
                reshape4 = tf.reshape(lr3, [shape[0], shape[1]*shape[2]*shape[3]])
                fc4 = self.fully_connect(reshape4, 21632, self.NOISE_UNIT_NUM, self.SEED)

        return fc4

    def decoder(self, z, reuse=False, is_training=False):  # z is expected [n, 200]
        with tf.variable_scope('decoder', reuse=reuse):
            with tf.variable_scope("layer1"):  # layer1 fc nx200 -> nx1024
                fc1 = self.fully_connect(z, self.NOISE_UNIT_NUM, 1024, self.SEED)
                bn1 = self.batch_norm(fc1)
                rl1 = tf.nn.relu(bn1)

            with tf.variable_scope("layer2"):  # layer2 fc nx1024 -> nx80000
                fc2 = self.fully_connect(rl1, 1024, 25*25*self.BASE_CHANNEL*4, self.SEED)
                bn2 = self.batch_norm(fc2)
                rl2 = tf.nn.relu(bn2)

            with tf.variable_scope("layer3"):  # layer3 deconv nx80000 -> nx25x25x128 -> nx50x50x64
                shape = tf.shape(rl2)
                reshape3 = tf.reshape(rl2, [shape[0], 25, 25, 128])
                deconv3 = self.conv2d_transpose(reshape3, self.BASE_CHANNEL*4, self.BASE_CHANNEL*2, 4, 2, self.SEED)
                bn3 = self.batch_norm(deconv3)
                rl3 = tf.nn.relu(bn3)

            with tf.variable_scope("layer4"):  # layer3 deconv nx50x50x64 -> nx100x100x1
                deconv4 = self.conv2d_transpose(rl3, self.BASE_CHANNEL*2, self.IMG_CHANNEL, 4, 2, self.SEED)
                tanh4 = tf.tanh(deconv4)

        return tanh4


    def discriminator(self, x, z, reuse=False, is_training=True): #z[n, 200], x[n, 100, 100, 1]
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope("x_layer1"):  # layer x1 conv [n, 100, 100, 1] -> [n, 50, 50, 64]
                convx1 = self.conv2d(x, self.IMG_CHANNEL, self.BASE_CHANNEL*2, 4, 2, self.SEED)
                lrx1 = self.leaky_relu(convx1, alpha=0.1)
                dropx1 = tf.layers.dropout(lrx1, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("x_layer2"):  # layer x2 conv [n, 50, 50, 64] -> [n, 25, 25, 64] -> [n, 40000]
                convx2 = self.conv2d(dropx1, self.BASE_CHANNEL*2, self.BASE_CHANNEL*2, 4, 2, self.SEED)
                bnx2 = self.batch_norm(convx2)
                lrx2 = self.leaky_relu(bnx2, alpha=0.1)
                dropx2 = tf.layers.dropout(lrx2, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)
                shapex2 = tf.shape(dropx2)
                reshape3 = tf.reshape(dropx2, [shapex2[0], shapex2[1]*shapex2[2]*shapex2[3]])

            with tf.variable_scope("z_layer1"):  # layer1 fc [n, 200] -> [n, 512]
                fcz1 = self.fully_connect(z, self.NOISE_UNIT_NUM, 512, self.SEED)
                lrz1 = self.leaky_relu(fcz1, alpha=0.1)
                dropz1 = tf.layers.dropout(lrz1, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("y_layer3"):  # layer1 fc [n, 40512], [n, 1024]
                con3 = tf.concat([reshape3, dropz1], axis=1)
                fc3 = self.fully_connect(con3, 40000+512, 1024, self.SEED)
                lr3 = self.leaky_relu(fc3, alpha=0.1)
                self.drop3 = tf.layers.dropout(lr3, rate=1.0 - self.KEEP_PROB, name='dropout', training=is_training)

            with tf.variable_scope("y_fc_logits"):
                self.logits = self.fully_connect(self.drop3, 1024, 1, self.SEED)

        return self.drop3, self.logits



