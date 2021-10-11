"""
Created on Wed Sep 8 2021

model: pix2pix

@author: Ray
"""
import os, sys, random, time
import tensorflow.compat.v1 as tf
import numpy as np
from utils import *
from PIL import Image

class pix2pix():
    
    def __init__(self, sess, args):
        self.sess = sess
        self.mode = args.mode.lower()
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.do_resize = args.do_resize
        self.k_initilizer = tf.random_normal_initializer(0, 0.02)
        self.b_initilizer = tf.random_normal_initializer(1, 0.02)
        self.bulid_model()
    
    def bulid_model(self):
        """
        init model
        """
        # init variable
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='x')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='y')

        # generator
        self.g_ = self.generator(self.x_)
        

        # discriminator
        self.real = self.discriminator(self.x_, self.y_, reuse=False)
        self.fake = self.discriminator(self.x_, self.g_, reuse=True)

        # loss
        self.loss_g, self.loss_d = self.loss(self.real, self.fake, self.y_, self.g_)

        # summary
        tf.summary.scalar("loss_g", self.loss_g)
        tf.summary.scalar("loss_d", self.loss_d)
        self.merged = tf.summary.merge_all()
        
        # vars
        self.vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

        # saver
        self.saver = tf.train.Saver()

    def generator(self, x, reuse=False):
        """
        encoder: (conv-batchnorm-leakyReLU)
        first layer (64): without batch norm
        other layers (128 256 512 512 512 512 512)
        -------
        UNet decoder: (conv-batchnorm-dropout-ReLU)
        all layers (512 1024 1024 1024 512 256 128 64)
        -------
        par x: input image
        return: generated image
        """
        with tf.variable_scope('generator'):

            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            layers = []
            en_h = self.g_encoder_block(x, 64, name='C64')
            layers.append(en_h)
            
            for i, filters in enumerate([128, 256, 512, 512, 512, 512, 512]):
                en_h = self.g_encoder_block(en_h, filters, name=f'dC{filters}_{i}')
                layers.append(en_h)
                
            encode_layers_num = len(layers)

            for i, filters in enumerate([512, 1024, 1024, 1024, 512, 256, 128, 64, 3]):
                skip_layer = encode_layers_num - i 
                
                if i in [0, 1, 2, 3, 8]:
                    inputs = layers[-1]
                    name = 'output' if i == 8 else f'CD{filters}_{i}'
                else:
                    inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3, name=f'concatenate_{i}')
                    name = f'uC{filters*2}_{i}'
                de_h = self.g_decoder_block(inputs, filters, i, name=name)
                layers.append(de_h)
            
            return de_h
        
    def discriminator(self, x, y, reuse=False):
        """
        70x70 discriminator (64 128 256 512 1)
        first layer (64): without batch norm
        output layer: sigmoid func.
        """
        with tf.variable_scope('discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            x = tf.concat([x, y], axis=3, name='D_concatenate')
            h = self.d_block(x, 64, 2, name='C64')
            h = self.d_block(h, 128, 2, name='C128')
            h = self.d_block(h, 256, 2, name='C256')
            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            h = self.d_block(h, 512, 1, padding='valid', name='C512')
            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            h = self.d_block(h, 1, 1, padding='valid', name='output')
            
        return h
        
    def g_encoder_block(self, inputs, filters, name='g_encoder_block'):
        """
        encoder block
        """
        with tf.variable_scope(name):
            h = tf.layers.conv2d(inputs,
                                 kernel_size=4,
                                 filters=filters,
                                 strides=2,
                                 padding='same',
                                 kernel_initializer=self.k_initilizer)
            if filters == 64:
                h = tf.nn.leaky_relu(h, alpha=0.2)
                
            else:
                h = tf.layers.batch_normalization(h,
                                                  epsilon=1e-5,
                                                  momentum=0.1,
                                                  training=True,
                                                  gamma_initializer=self.b_initilizer)
                h = tf.nn.leaky_relu(h, alpha=0.2)
        return h 
    
    def g_decoder_block(self, inputs, filters, i, name='g_decoder_block'):
        """
        Unet decoder
        """
        strides = 1 if i == 0 else 2
        with tf.variable_scope(name):
            h = tf.layers.conv2d_transpose(inputs,
                                           kernel_size=4,
                                           filters=filters,
                                           strides=strides,
                                           padding='same',
                                           kernel_initializer=self.k_initilizer)
            if filters == 3:
                h = tf.nn.tanh(h)
                
            else:
                h = tf.layers.batch_normalization(h,
                                                  epsilon=1e-5,
                                                  momentum=0.1,
                                                  training=True,
                                                  gamma_initializer=self.b_initilizer)
                if i < 3:
                    h = tf.nn.dropout(h, keep_prob=0.5)
                    h = tf.nn.relu(h)
                else:
                    h = tf.nn.relu(h)
        return h
        
    def d_block(self, inputs, filters, strides, padding='same', name='d_block'):
        """
        PatchGAN discriminator
        """
        with tf.variable_scope(name):
            h = tf.layers.conv2d(inputs,
                                 kernel_size=4,
                                 filters=filters,
                                 strides=strides,
                                 padding=padding,
                                 kernel_initializer=self.k_initilizer)
            if filters == 1:
                h = tf.nn.sigmoid(h)
                
            elif filters == 64:
                h = tf.nn.leaky_relu(h, alpha=0.2)
                
            else:
                h = tf.layers.batch_normalization(h,
                                                  epsilon=1e-5,
                                                  momentum=0.1,
                                                  training=True,
                                                  gamma_initializer=self.b_initilizer)
                h = tf.nn.leaky_relu(h, alpha=0.2)
        return h
    
    def loss(self, real, fake, y, g, L1_lambda=100):
        """
        discriminator loss, generator loss
        """
        loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        loss_d = loss_d_real + loss_d_fake

        loss_g_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
        loss_g_l1 = tf.reduce_mean(tf.abs(y - g))
        loss_g = loss_g_gan + loss_g_l1 * L1_lambda
        return loss_g, loss_d
    
    def train(self, image_lists):
        """
        par image_lists: list of image paths
        par epochs: epochs
        par batch_size: batch size
        """
        # Optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_step_d = tf.train.AdamOptimizer(learning_rate=0.0002,
                                              beta1=0.5).minimize(self.loss_d, var_list=self.vars_d)
        train_step_g = tf.train.AdamOptimizer(learning_rate=0.0002,
                                              beta1=0.5).minimize(self.loss_g, var_list=self.vars_g)

        # init variable
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./log", self.sess.graph)
        #self.saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))
        
        # Training
        for i in range(self.epochs):
            batch_num = int(np.ceil(len(image_lists) / self.batch_size))
            batch_list = np.array_split(image_lists, batch_num)
            random.shuffle(batch_list)
            t1 = time.time()

            loss_ds = []
            loss_gs = []
            for j in range(len(batch_list)):

                batch_x, batch_y = load_data(batch_list[j], self.mode)
                _, loss_d = self.sess.run([train_step_d, self.loss_d],
                                          feed_dict={self.x_: batch_x, self.y_: batch_y})

                _, loss_g = self.sess.run([train_step_g, self.loss_g],
                                            feed_dict={self.x_: batch_x, self.y_: batch_y})
                loss_ds.append(loss_d)
                loss_gs.append(loss_g)
            loss_ds = sum(loss_ds) / len(batch_list)
            loss_gs = sum(loss_gs) / len(batch_list)
                
            print("Epoch: %d/%d: discriminator loss: %.4f; generator loss: %.4f; training time: %.1fs" % ((i + 1), self.epochs, loss_ds, loss_gs, time.time()-t1))
        
        # Save loss
            summary = self.sess.run(self.merged, feed_dict={self.x_: batch_x, self.y_: batch_y})
            self.writer.add_summary(summary, global_step=i)
            
        # Save model
            if (i + 1) % 10 == 0:
                self.saver.save(self.sess, './checkpoint/epoch_%d.ckpt' % (i + 1))
            
    def test(self, images):
        """
        par images: list of image path
        par save_path: save path
        """
        # init variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        # load model
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))
        
        im = []
        for j in range(len(images)):
            batch_x, _ = load_data(np.array([images[j]]), self.mode)
            g = self.sess.run(self.g_, feed_dict={self.x_: batch_x})
            g = (np.array(g[0]) + 1) * 127.5

            if self.do_resize:
                g = resize(images[j], g).eval()
                
            g = Image.fromarray(np.uint8(g))
            im.append(g)

        return im, images                